import sys
import subprocess
import os
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QCheckBox, QMessageBox, QTabWidget, QProgressDialog
)

class ProcessWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        try:
            # Run the command and wait until it finishes.
            subprocess.run(self.cmd, check=True)
            self.finished.emit()
        except subprocess.CalledProcessError as e:
            self.error.emit(str(e))
            self.finished.emit()

class AudioBChampionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioBChampion")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # Create tab widget for the two main functions
        self.tabs = QTabWidget()
        self.tab_separation = QWidget()
        self.tab_normalization = QWidget()
        self.tabs.addTab(self.tab_separation, "Separate Tracks")
        self.tabs.addTab(self.tab_normalization, "Pitch Normalization")
        
        self.init_separation_tab()
        self.init_normalization_tab()
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def init_separation_tab(self):
        layout = QVBoxLayout()
        
        # Audiobook file selection
        self.audiobook_label = QLabel("Audiobook File:")
        self.audiobook_line = QLineEdit()
        self.audiobook_browse = QPushButton("Browse")
        self.audiobook_browse.clicked.connect(self.browse_audiobook)
        layout.addWidget(self.audiobook_label)
        layout.addWidget(self.audiobook_line)
        layout.addWidget(self.audiobook_browse)
        
        # Ebook file selection (optional)
        self.ebook_label = QLabel("Ebook File (Optional):")
        self.ebook_line = QLineEdit()
        self.ebook_browse = QPushButton("Browse")
        self.ebook_browse.clicked.connect(self.browse_ebook)
        layout.addWidget(self.ebook_label)
        layout.addWidget(self.ebook_line)
        layout.addWidget(self.ebook_browse)
        
        # Output directory
        self.output_label = QLabel("Output Directory:")
        self.output_line = QLineEdit()
        self.output_browse = QPushButton("Browse")
        self.output_browse.clicked.connect(self.browse_output)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_line)
        layout.addWidget(self.output_browse)
        
        # Noise reduction checkbox
        self.noise_checkbox = QCheckBox("Apply Noise Reduction")
        layout.addWidget(self.noise_checkbox)
        
        # Process button
        self.process_button = QPushButton("Process Audiobook")
        self.process_button.clicked.connect(self.process_audiobook)
        layout.addWidget(self.process_button)
        
        self.tab_separation.setLayout(layout)
    
    def init_normalization_tab(self):
        layout = QVBoxLayout()
        
        # Input files selection
        self.inputs_label = QLabel("Input Audio Files (separated by semicolons):")
        self.inputs_line = QLineEdit()
        self.inputs_browse = QPushButton("Browse")
        self.inputs_browse.clicked.connect(self.browse_inputs)
        layout.addWidget(self.inputs_label)
        layout.addWidget(self.inputs_line)
        layout.addWidget(self.inputs_browse)
        
        # Output directory
        self.norm_output_label = QLabel("Output Directory:")
        self.norm_output_line = QLineEdit()
        self.norm_output_browse = QPushButton("Browse")
        self.norm_output_browse.clicked.connect(self.browse_norm_output)
        layout.addWidget(self.norm_output_label)
        layout.addWidget(self.norm_output_line)
        layout.addWidget(self.norm_output_browse)
        
        # Process button
        self.norm_process_button = QPushButton("Normalize Pitch")
        self.norm_process_button.clicked.connect(self.normalize_pitch)
        layout.addWidget(self.norm_process_button)
        
        self.tab_normalization.setLayout(layout)
    
    def browse_audiobook(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Audiobook File")
        if file:
            self.audiobook_line.setText(file)
    
    def browse_ebook(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Ebook File", filter="Text Files (*.txt);;PDF Files (*.pdf)")
        if file:
            self.ebook_line.setText(file)
    
    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_line.setText(folder)
    
    def browse_inputs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Input Audio Files")
        if files:
            self.inputs_line.setText(";".join(files))
    
    def browse_norm_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.norm_output_line.setText(folder)
    
    def process_audiobook(self):
        audiobook = self.audiobook_line.text()
        ebook = self.ebook_line.text()
        output_dir = self.output_line.text()
        noise_reduce = self.noise_checkbox.isChecked()
        
        if not audiobook or not output_dir:
            QMessageBox.warning(self, "Error", "Please select at least an audiobook file and an output directory.")
            return
        
        # Build command line arguments for the separation command.
        cmd = ["python", "audio_bchampion/main.py", "separate", "--audiobook", audiobook, "--output", output_dir]
        if ebook:
            cmd.extend(["--ebook", ebook])
        if noise_reduce:
            cmd.append("--noise_reduce")
        
        print("Running command:", " ".join(cmd))
        self.run_subprocess_command(cmd, "Processing audiobook...")
    
    def normalize_pitch(self):
        inputs = self.inputs_line.text().split(";")
        output_dir = self.norm_output_line.text()
        if not inputs or not output_dir:
            QMessageBox.warning(self, "Error", "Please select input files and an output directory.")
            return
        
        cmd = ["python", "audio_bchampion/main.py", "normalize", "--inputs"] + inputs + ["--output", output_dir]
        print("Running command:", " ".join(cmd))
        self.run_subprocess_command(cmd, "Normalizing pitch...")
    
    def run_subprocess_command(self, cmd, progress_text):
        # Create an indeterminate progress dialog.
        progress_dialog = QProgressDialog(progress_text, "Cancel", 0, 0, self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()
        
        # Run the command in a background thread.
        self.worker = ProcessWorker(cmd)
        self.worker.finished.connect(lambda: self.on_command_finished(progress_dialog))
        self.worker.error.connect(lambda err: self.on_command_error(err, progress_dialog))
        self.worker.start()
    
    def on_command_finished(self, progress_dialog):
        progress_dialog.cancel()
        QMessageBox.information(self, "Done", "Processing completed.")
    
    def on_command_error(self, error_msg, progress_dialog):
        progress_dialog.cancel()
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioBChampionGUI()
    window.show()
    sys.exit(app.exec_())
