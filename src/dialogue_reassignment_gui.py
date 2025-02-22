#!/usr/bin/env python
"""
dialogue_reassignment_gui.py

This GUI is designed for re‑assigning speaker labels in an already processed state.
It loads a JSON state file and the ebook, displays the dialogue segments in a table,
and allows manual reassignment. Changes are saved to the state file.
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

DEFAULT_OUTPUT_DIR = "Z:/AudioBChampion/data/output"

def load_global_assignments():
    assignments_file = os.path.join(DEFAULT_OUTPUT_DIR, "global_speaker_assignments.json")
    if os.path.exists(assignments_file):
        try:
            with open(assignments_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading global assignments:", e)
    return {}

def save_global_assignments(assignments):
    assignments_file = os.path.join(DEFAULT_OUTPUT_DIR, "global_speaker_assignments.json")
    try:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        with open(assignments_file, "w", encoding="utf-8") as f:
            json.dump(assignments, f, indent=4)
    except Exception as e:
        print("Error saving global assignments:", e)

class DialogueReassignmentGUI(QtWidgets.QWidget):
    def __init__(self, state_file, ebook_file):
        super().__init__()
        self.state_file = state_file
        self.ebook_file = ebook_file
        self.state_data = None
        self.global_assignments = load_global_assignments()
        self.current_font_size = 14  # Ebook display font size
        self.table_font_size = 14    # Dialogue table font size
        self.load_state()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Dialogue Reassignment GUI")
        self.resize(1200, 700)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(splitter)

        # Left pane: Table and controls.
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        top_layout = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save Updated State", self)
        self.save_btn.clicked.connect(self.save_state)
        top_layout.addWidget(self.save_btn)
        left_layout.addLayout(top_layout)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Index", "Start Time", "End Time", "Dialogue Text", "Speaker"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table)

        table_zoom_layout = QtWidgets.QHBoxLayout()
        self.table_zoom_in_btn = QtWidgets.QPushButton("Table Zoom In", self)
        self.table_zoom_in_btn.clicked.connect(self.zoom_table_in)
        table_zoom_layout.addWidget(self.table_zoom_in_btn)
        self.table_zoom_out_btn = QtWidgets.QPushButton("Table Zoom Out", self)
        self.table_zoom_out_btn.clicked.connect(self.zoom_table_out)
        table_zoom_layout.addWidget(self.table_zoom_out_btn)
        left_layout.addLayout(table_zoom_layout)

        # Right pane: Ebook display (editable).
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
        # Allow editing in case you need to fix formatting.
        self.ebook_display.setReadOnly(False)
        right_layout.addWidget(self.ebook_display)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])
        self.populate_table()

    def zoom_in(self):
        self.current_font_size += 2
        font = self.ebook_display.font()
        font.setPointSize(self.current_font_size)
        self.ebook_display.setFont(font)

    def zoom_out(self):
        self.current_font_size = max(6, self.current_font_size - 2)
        font = self.ebook_display.font()
        font.setPointSize(self.current_font_size)
        self.ebook_display.setFont(font)

    def zoom_table_in(self):
        self.table_font_size += 2
        self.update_table_font()

    def zoom_table_out(self):
        self.table_font_size = max(6, self.table_font_size - 2)
        self.update_table_font()

    def update_table_font(self):
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
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.state_data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Loading State", str(e))
            sys.exit(1)
        if "ebook_text" in self.state_data:
            self.ebook_display.setPlainText(self.state_data["ebook_text"])
        else:
            try:
                text = load_ebook_text(self.ebook_file)
                self.ebook_display.setPlainText(text)
            except Exception as e:
                self.ebook_display.setPlainText(f"Error loading ebook: {e}")

    def populate_table(self):
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

    def save_state(self):
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
        self.state_data["ebook_text"] = self.ebook_display.toPlainText()
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state_data, f, indent=4)
            QtWidgets.QMessageBox.information(self, "Saved", f"Updated state saved to {self.state_file}")
            for row, assignment in updated_assignments.items():
                dialogue = self.state_data["dialogue_subtitles"][int(row)][2]
                self.global_assignments[dialogue] = assignment
            save_global_assignments(self.global_assignments)
            self.populate_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Saving State", str(e))

    def export_audio_tracks(self):
        self.save_state()
        audiobook_file = self.state_data.get("audiobook", "")
        if not audiobook_file:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Audiobook File", "", "Audio Files (*.mp3 *.wav *.m4a)")
            if file_path:
                audiobook_file = file_path
                self.state_data["audiobook"] = audiobook_file
            else:
                QtWidgets.QMessageBox.warning(self, "No Audiobook", "No audiobook file selected.")
                return
        export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        dialogue_subtitles = self.state_data.get("dialogue_subtitles", [])
        narration_subtitles = self.state_data.get("narration_subtitles", [])
        speaker_assignments = self.state_data.get("speaker_assignments", {})
        from audio_bchampion.forced_alignment import process_audiobook_alternate
        try:
            process_audiobook_alternate(audiobook_file, dialogue_subtitles, narration_subtitles, speaker_assignments, export_dir)
            QtWidgets.QMessageBox.information(self, "Export Complete", "Audio tracks exported successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

def main():
    parser = argparse.ArgumentParser(description="Dialogue Reassignment GUI")
    parser.add_argument("--state", required=True, help="Path to the processing state JSON file")
    parser.add_argument("--ebook", required=True, help="Path to the ebook file (DOCX, PDF, or TXT)")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    gui = DialogueReassignmentGUI(state_file=args.state, ebook_file=args.ebook)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
