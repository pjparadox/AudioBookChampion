#!/usr/bin/env python
"""
dialogue_live_gui.py

This live GUI lets you manually review and adjust speaker attributions for dialogue segments,
split combined dialogue lines, and view/edit the ebook text in a right‐hand pane.
It accepts a saved state JSON file (from forced_alignment.py) and an ebook file (DOCX/PDF/TXT)
via command‑line arguments so that you can resume processing without re‑processing audio.
"""

import os
import sys
import json
import glob
import re
import difflib
import shutil
from datetime import datetime
import argparse

from PyQt5 import QtWidgets, QtCore, QtGui

# Import the audio processing export function from forced_alignment.py
from audio_bchampion.forced_alignment import process_audiobook_alternate, load_ebook_text

# Global known characters dictionary.
KNOWN_CHARACTERS = {
    "George": "male",
    "Lennie": "male",
    "Curley's Wife": "female",
    "Candy": "male",
    "Crooks": "male",
    "Slim": "male",
    "Curley": "male",
    "Carlson": "male",
    "Boss": "male",
    "The Boss": "male",
    "Narrator": "unknown",
    "Unknown": "unknown"
}

# Default directories and global assignments file
DEFAULT_OUTPUT_DIR = "Z:/AudioBChampion/data/output"
DEFAULT_EBOOK_DIR = "Z:/AudioBChampion/data/pdf"
DEFAULT_INPUT_DIR = "Z:/AudioBChampion/data/input"
GLOBAL_ASSIGNMENTS_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "global_speaker_assignments.json")
SIMILARITY_THRESHOLD = 0.75  # for matching dialogue segments

def load_global_assignments():
    """
    Load the global speaker assignment cache from a JSON file.

    Returns:
        dict: A dictionary of dialogue patterns to speaker assignments.
    """
    if os.path.exists(GLOBAL_ASSIGNMENTS_FILE):
        try:
            with open(GLOBAL_ASSIGNMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading global assignments:", e)
    return {}

def save_global_assignments(assignments):
    """
    Save the global speaker assignment cache to a JSON file.

    Args:
        assignments (dict): The dictionary of dialogue patterns to speaker assignments.
    """
    try:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        with open(GLOBAL_ASSIGNMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(assignments, f, indent=4)
    except Exception as e:
        print("Error saving global assignments:", e)

def apply_global_assignments(state_data, global_assignments):
    """
    Apply cached global speaker assignments to the current session's state data.

    Matches dialogue text against the global cache using string similarity.

    Args:
        state_data (dict): The current processing state including dialogue subtitles.
        global_assignments (dict): The global cache of assignments.

    Returns:
        dict: The updated state data with applied speaker assignments.
    """
    if "dialogue_subtitles" not in state_data or "speaker_assignments" not in state_data:
        return state_data
    for idx, seg in enumerate(state_data["dialogue_subtitles"]):
        dialogue_text = seg[2]
        best_match = None
        best_ratio = 0.0
        for text_pattern, speaker in global_assignments.items():
            ratio = difflib.SequenceMatcher(None, dialogue_text, text_pattern).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = speaker
        if best_ratio >= SIMILARITY_THRESHOLD:
            state_data["speaker_assignments"][str(idx)] = best_match
    return state_data

class SplitDialogueDialog(QtWidgets.QDialog):
    """
    A modal dialog for splitting a dialogue line into two segments.
    """
    def __init__(self, dialogue_text, parent=None):
        """
        Initialize the SplitDialogueDialog.

        Args:
            dialogue_text (str): The text of the dialogue line to split.
            parent (QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Split Dialogue Line")
        self.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self)
        self.text_edit = QtWidgets.QTextEdit(self)
        self.text_edit.setPlainText(dialogue_text)
        layout.addWidget(self.text_edit)
        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK", self)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QtWidgets.QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
    def get_split_index(self):
        """
        Get the cursor position where the split should occur.

        Returns:
            int: The character index of the split.
        """
        return self.text_edit.textCursor().position()

class DialogueLiveGUI(QtWidgets.QWidget):
    """
    The main widget for the Live Dialogue Editor GUI.

    Allows users to:
    - View dialogue segments in a table.
    - Reassign speakers.
    - Split dialogue segments.
    - View and search the ebook text context.
    - Export the final audio tracks.
    """
    def __init__(self, state_file, ebook_file):
        """
        Initialize the GUI with the state and ebook files.

        Args:
            state_file (str): Path to the JSON state file.
            ebook_file (str): Path to the ebook file.
        """
        super().__init__()
        self.state_file = state_file
        self.ebook_file = ebook_file
        self.state_data = None      # Loaded state JSON for current file
        self.audiobook_file = None  # Will be loaded from state data if available
        self.global_assignments = load_global_assignments()
        self.current_font_size = 14  # Ebook display font size
        self.table_font_size = 14    # Dialogue table font size
        self.load_state()
        self.init_ui()

    def init_ui(self):
        """Set up the UI layout and widgets."""
        self.setWindowTitle("Dialogue Live GUI – Review & Adjust Speaker Attributions")
        self.resize(1200, 700)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(splitter)

        # Left pane: Controls and dialogue table.
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        # Top controls.
        top_layout = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save Updated State", self)
        self.save_btn.clicked.connect(self.save_state)
        top_layout.addWidget(self.save_btn)
        self.split_btn = QtWidgets.QPushButton("Split Selected Line", self)
        self.split_btn.clicked.connect(self.split_selected_line)
        top_layout.addWidget(self.split_btn)
        self.search_btn = QtWidgets.QPushButton("Search in Ebook", self)
        self.search_btn.clicked.connect(self.search_in_ebook)
        top_layout.addWidget(self.search_btn)
        self.export_btn = QtWidgets.QPushButton("Export Audio Tracks", self)
        self.export_btn.clicked.connect(self.export_audio_tracks)
        top_layout.addWidget(self.export_btn)
        left_layout.addLayout(top_layout)

        # Dialogue table.
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Index", "Start Time", "End Time", "Dialogue Text", "Speaker"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self.update_context_display)
        left_layout.addWidget(self.table)

        # Table zoom controls.
        table_zoom_layout = QtWidgets.QHBoxLayout()
        self.table_zoom_in_btn = QtWidgets.QPushButton("Table Zoom In", self)
        self.table_zoom_in_btn.clicked.connect(self.zoom_table_in)
        table_zoom_layout.addWidget(self.table_zoom_in_btn)
        self.table_zoom_out_btn = QtWidgets.QPushButton("Table Zoom Out", self)
        self.table_zoom_out_btn.clicked.connect(self.zoom_table_out)
        table_zoom_layout.addWidget(self.table_zoom_out_btn)
        left_layout.addLayout(table_zoom_layout)

        # Right pane: Ebook display with in-place editing.
        right_layout = QtWidgets.QVBoxLayout()
        zoom_layout = QtWidgets.QHBoxLayout()
        self.zoom_in_btn = QtWidgets.QPushButton("Ebook Zoom In", self)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QtWidgets.QPushButton("Ebook Zoom Out", self)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        right_layout.addLayout(zoom_layout)
        self.ebook_display = QtWidgets.QTextEdit(self)
        ebook_font = QtGui.QFont()
        ebook_font.setPointSize(self.current_font_size)
        self.ebook_display.setFont(ebook_font)
        # Allow editing so you can fix ebook formatting on the fly.
        self.ebook_display.setReadOnly(False)
        right_layout.addWidget(self.ebook_display)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])
        self.populate_table()

    def zoom_in(self):
        """Increase the font size of the ebook display."""
        self.current_font_size += 2
        font = self.ebook_display.font()
        font.setPointSize(self.current_font_size)
        self.ebook_display.setFont(font)

    def zoom_out(self):
        """Decrease the font size of the ebook display."""
        self.current_font_size = max(6, self.current_font_size - 2)
        font = self.ebook_display.font()
        font.setPointSize(self.current_font_size)
        self.ebook_display.setFont(font)

    def zoom_table_in(self):
        """Increase the font size of the dialogue table."""
        self.table_font_size += 2
        self.update_table_font()

    def zoom_table_out(self):
        """Decrease the font size of the dialogue table."""
        self.table_font_size = max(6, self.table_font_size - 2)
        self.update_table_font()

    def update_table_font(self):
        """Apply the current table font size to all items in the table."""
        font = QtGui.QFont()
        font.setPointSize(self.table_font_size)
        self.table.setFont(font)
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setFont(font)
                widget = self.table.cellWidget(row, col)
                if widget:
                    widget.setFont(font)
        header_font = QtGui.QFont()
        header_font.setPointSize(self.table_font_size)
        self.table.horizontalHeader().setFont(header_font)
        self.table.verticalHeader().setFont(header_font)
        self.table.resizeRowsToContents()

    def load_state(self):
        """Load the processing state from the JSON file."""
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.state_data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Loading State", str(e))
            sys.exit(1)
        if "audiobook" in self.state_data:
            self.audiobook_file = self.state_data["audiobook"]
        if "ebook_text" in self.state_data:
            # If ebook_text was saved, use it; otherwise, load from file.
            self.ebook_display.setPlainText(self.state_data["ebook_text"])
        else:
            try:
                text = load_ebook_text(self.ebook_file)
                self.ebook_display.setPlainText(text)
            except Exception as e:
                self.ebook_display.setPlainText(f"Error loading ebook: {e}")

    def populate_table(self):
        """Populate the dialogue table with data from the loaded state."""
        if not self.state_data or "dialogue_subtitles" not in self.state_data or "speaker_assignments" not in self.state_data:
            QtWidgets.QMessageBox.critical(self, "Invalid State", "The state file does not contain required keys.")
            return
        dialogue_subtitles = self.state_data["dialogue_subtitles"]
        speaker_assignments = self.state_data.get("speaker_assignments", {})
        self.table.setRowCount(len(dialogue_subtitles))
        font = QtGui.QFont()
        font.setPointSize(self.table_font_size)
        for idx, seg in enumerate(dialogue_subtitles):
            start, end, text = seg
            idx_item = QtWidgets.QTableWidgetItem(str(idx))
            idx_item.setFont(font)
            start_item = QtWidgets.QTableWidgetItem(f"{start:.2f}")
            start_item.setFont(font)
            end_item = QtWidgets.QTableWidgetItem(f"{end:.2f}")
            end_item.setFont(font)
            dialogue_item = QtWidgets.QTableWidgetItem(text)
            dialogue_item.setFont(font)
            self.table.setItem(idx, 0, idx_item)
            self.table.setItem(idx, 1, start_item)
            self.table.setItem(idx, 2, end_item)
            self.table.setItem(idx, 3, dialogue_item)
            combo = QtWidgets.QComboBox(self)
            combo.setEditable(True)
            combo.setFont(font)
            for name in sorted(KNOWN_CHARACTERS.keys()):
                combo.addItem(name)
            combo.addItem("Other")
            current = speaker_assignments.get(str(idx), "Unknown unknown")
            current_name = current.strip() if isinstance(current, str) and current.strip() else "Unknown"
            index_in_combo = combo.findText(current_name)
            if index_in_combo >= 0:
                combo.setCurrentIndex(index_in_combo)
            else:
                combo.setCurrentIndex(combo.findText("Other"))
            self.table.setCellWidget(idx, 4, combo)
        self.table.resizeRowsToContents()

    def update_context_display(self):
        """Highlight the corresponding text in the ebook display when a table row is selected."""
        selected_rows = self.table.selectionModel().selectedRows()
        row = selected_rows[0].row() if selected_rows else self.table.currentRow()
        if row is not None and row >= 0:
            dialogue_text = self.table.item(row, 3).text().strip()
            self.ebook_display.setExtraSelections([])
            if dialogue_text:
                doc = self.ebook_display.document()
                cursor = doc.find(dialogue_text)
                if cursor.isNull():
                    # If not found exactly, try to find the closest matching paragraph.
                    paragraphs = self.ebook_display.toPlainText().split('\n\n')
                    best_ratio = 0.0
                    best_para = ""
                    for para in paragraphs:
                        ratio = difflib.SequenceMatcher(None, dialogue_text, para).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_para = para
                    if best_para:
                        cursor = doc.find(best_para)
                if not cursor.isNull():
                    selection = QtWidgets.QTextEdit.ExtraSelection()
                    selection.format.setBackground(QtGui.QColor("#00008B"))
                    selection.format.setForeground(QtGui.QColor("white"))
                    selection.cursor = cursor
                    self.ebook_display.setExtraSelections([selection])
                    self.ebook_display.setTextCursor(cursor)
                    self.ebook_display.ensureCursorVisible()
                else:
                    self.ebook_display.setExtraSelections([])
        else:
            self.ebook_display.setExtraSelections([])

    def search_in_ebook(self):
        """Search for the selected dialogue text within the ebook display."""
        selected_rows = self.table.selectionModel().selectedRows()
        row = selected_rows[0].row() if selected_rows else self.table.currentRow()
        if row is None or row < 0:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a dialogue row to search for.")
            return
        dialogue_text = self.table.item(row, 3).text().strip()
        if not dialogue_text:
            QtWidgets.QMessageBox.warning(self, "Empty Dialogue", "The selected dialogue is empty.")
            return
        doc = self.ebook_display.document()
        cursor = doc.find(dialogue_text)
        if cursor.isNull():
            paragraphs = self.ebook_display.toPlainText().split('\n\n')
            best_ratio = 0.0
            best_para = ""
            for para in paragraphs:
                ratio = difflib.SequenceMatcher(None, dialogue_text, para).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_para = para
            if best_para:
                cursor = doc.find(best_para)
        if not cursor.isNull():
            selection = QtWidgets.QTextEdit.ExtraSelection()
            selection.format.setBackground(QtGui.QColor("#00008B"))
            selection.format.setForeground(QtGui.QColor("white"))
            selection.cursor = cursor
            self.ebook_display.setExtraSelections([selection])
            self.ebook_display.setTextCursor(cursor)
            self.ebook_display.ensureCursorVisible()
        else:
            QtWidgets.QMessageBox.information(self, "Not Found", "No close match found in the ebook.")

    def split_selected_line(self):
        """Open the split dialog to divide the selected dialogue line."""
        selected_rows = self.table.selectionModel().selectedRows()
        row = selected_rows[0].row() if selected_rows else self.table.currentRow()
        if row is None or row < 0:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a dialogue row to split.")
            return
        dialogue_text = self.table.item(row, 3).text()
        if not dialogue_text:
            QtWidgets.QMessageBox.warning(self, "Empty Dialogue", "The selected dialogue is empty.")
            return
        try:
            start_time = float(self.table.item(row, 1).text())
            end_time = float(self.table.item(row, 2).text())
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid Time", "Could not parse start or end times.")
            return
        split_dialog = SplitDialogueDialog(dialogue_text, self)
        if split_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        split_index = split_dialog.get_split_index()
        if split_index <= 0 or split_index >= len(dialogue_text):
            QtWidgets.QMessageBox.warning(self, "Invalid Split", "Split index must be between 0 and the dialogue length.")
            return
        text1 = dialogue_text[:split_index].strip()
        text2 = dialogue_text[split_index:].strip()
        if not text1 or not text2:
            QtWidgets.QMessageBox.warning(self, "Invalid Split", "Cannot split into empty segments.")
            return
        total_duration = end_time - start_time
        ratio = split_index / len(dialogue_text)
        mid_time = start_time + total_duration * ratio
        dialogue_subtitles = self.state_data["dialogue_subtitles"]
        dialogue_subtitles[row] = [start_time, mid_time, text1]
        dialogue_subtitles.insert(row + 1, [mid_time, end_time, text2])
        speaker_assignments = self.state_data.get("speaker_assignments", {})
        original = speaker_assignments.get(str(row), "Unknown unknown")
        speaker_assignments[str(row)] = original
        new_assignments = {}
        for i, v in enumerate(speaker_assignments.values()):
            new_assignments[str(i)] = v
        self.state_data["speaker_assignments"] = new_assignments
        self.populate_table()
        self.save_state()  # Auto-save after splitting.

    def save_state(self):
        """Save the current state (assignments, ebook text) to the JSON file."""
        if not self.state_data or not self.state_file:
            return
        try:
            backup_file = self.state_file + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
            shutil.copy2(self.state_file, backup_file)
            print(f"Backup created: {backup_file}")
        except Exception as e:
            print("Error creating backup:", e)
        updated_assignments = {}
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 4)
            if widget:
                selected = widget.currentText().strip()
                if selected.lower() == "other":
                    name, ok1 = QtWidgets.QInputDialog.getText(self, "Enter Speaker Name", f"Enter speaker name for row {row}:")
                    if not (ok1 and name.strip()):
                        updated_assignments[str(row)] = "Unknown unknown"
                        continue
                    gender, ok2 = QtWidgets.QInputDialog.getText(self, "Enter Speaker Gender", f"Enter gender for {name} (male/female):")
                    if not (ok2 and gender.strip()):
                        updated_assignments[str(row)] = f"{name.strip()} unknown"
                        continue
                    new_name = name.strip()
                    new_gender = gender.strip().lower()
                    updated_assignments[str(row)] = f"{new_name} {new_gender}"
                    if new_name not in KNOWN_CHARACTERS:
                        KNOWN_CHARACTERS[new_name] = new_gender
                else:
                    updated_assignments[str(row)] = selected
            else:
                updated_assignments[str(row)] = "Unknown unknown"
        self.state_data["speaker_assignments"] = updated_assignments
        if self.audiobook_file:
            self.state_data["audiobook"] = self.audiobook_file
        # Also save the current ebook text (in case you made edits)
        self.state_data["ebook_text"] = self.ebook_display.toPlainText()
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state_data, f, indent=4)
            QtWidgets.QMessageBox.information(self, "Saved", f"Updated state saved to {self.state_file}")
            # Update global assignments using dialogue text as key.
            for row, assignment in updated_assignments.items():
                dialogue = self.state_data["dialogue_subtitles"][int(row)][2]
                self.global_assignments[dialogue] = assignment
            save_global_assignments(self.global_assignments)
            self.populate_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Saving State", str(e))

    def export_audio_tracks(self):
        """Export the final audio tracks based on current assignments."""
        self.save_state()
        if not self.audiobook_file:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Audiobook File", "", "Audio Files (*.mp3 *.wav *.m4a)")
            if file_path:
                self.audiobook_file = file_path
            else:
                QtWidgets.QMessageBox.warning(self, "No Audiobook", "No audiobook file selected.")
                return
        export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        dialogue_subtitles = self.state_data.get("dialogue_subtitles", [])
        narration_subtitles = self.state_data.get("narration_subtitles", [])
        speaker_assignments = self.state_data.get("speaker_assignments", {})
        try:
            process_audiobook_alternate(self.audiobook_file, dialogue_subtitles, narration_subtitles, speaker_assignments, export_dir)
            QtWidgets.QMessageBox.information(self, "Export Complete", "Audio tracks exported successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

def main():
    """Main entry point for the Dialogue Live GUI."""
    parser = argparse.ArgumentParser(description="Resume Dialogue Live GUI for speaker reassignment")
    parser.add_argument("--state", required=True, help="Path to the processing state JSON file")
    parser.add_argument("--ebook", required=True, help="Path to the ebook file (DOCX, PDF, or TXT)")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    gui = DialogueLiveGUI(state_file=args.state, ebook_file=args.ebook)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
