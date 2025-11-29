import os
import sys
import re
import difflib
import subprocess
import argparse
import torch
import whisper
import PyPDF2  # fallback for PDF ebooks
from pydub import AudioSegment

try:
    from docx import Document
except ImportError:
    sys.exit("Please install python-docx (pip install python-docx) to read DOCX files.")

# For the GUI
from PyQt5 import QtWidgets, QtCore

# ---------------------------
# Global Debug Log for Speaker Attribution
# ---------------------------
DEBUG_LOG = []  # Each entry logs dialogue text, context window size, raw DeepSeek output, text-based result, speech style, fallback, etc.

# ---------------------------
# Helper Functions
# ---------------------------
def remove_ansi_escape(text):
    """
    Remove ANSI escape sequences from text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with ANSI codes removed.
    """
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def remove_think_tags(text):
    """
    Remove <think>...</think> tags produced by reasoning models.

    Args:
        text (str): Input text.

    Returns:
        str: Text with tags removed.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def sanitize_speaker_name(name):
    """
    Remove special characters from a speaker name, keeping only alphanumerics and underscores.

    Args:
        name (str): Raw speaker name.

    Returns:
        str: Sanitized speaker name.
    """
    return re.sub(r'[^A-Za-z0-9_]', '', name)

def format_timestamp(seconds):
    """
    Format time in seconds to HH:MM:SS,mmm string.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted timestamp.
    """
    millis = int((seconds - int(seconds)) * 1000)
    s = int(seconds)
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def export_srt(subtitles, srt_path):
    """
    Write subtitles to an SRT file.

    Args:
        subtitles (list of tuple): List of (start, end, text) tuples.
        srt_path (str): Output file path.
    """
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(subtitles, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")
    print(f"SRT file exported to: {srt_path}")

# ---------------------------
# Transcription Using Whisper
# ---------------------------
def transcribe_audio(audio_file):
    """
    Transcribe audio using Whisper.

    Args:
        audio_file (str): Path to audio file.

    Returns:
        dict: Whisper transcription result.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    return model.transcribe(audio_file)

def merge_transcription(result):
    """
    Merge transcription segments into full text and boundaries.

    Args:
        result (dict): Whisper result.

    Returns:
        tuple: (full_text_str, list_of_boundaries)
    """
    full_text = ""
    boundaries = []
    for seg in result.get("segments", []):
        boundaries.append((seg["start"], seg["end"], seg["text"].strip()))
        full_text += " " + seg["text"].strip()
    return full_text.strip(), boundaries

# ---------------------------
# Ebook Loading and Book Title Extraction
# ---------------------------
def load_ebook_text(ebook_file):
    """
    Load text from an ebook file.

    Args:
        ebook_file (str): Path to ebook.

    Returns:
        str: Extracted text.
    """
    ebook_text = ""
    if ebook_file.lower().endswith('.docx'):
        try:
            doc = Document(ebook_file)
            for para in doc.paragraphs:
                ebook_text += para.text + "\n"
        except Exception as e:
            print("Error reading DOCX ebook:", e)
    elif ebook_file.lower().endswith('.pdf'):
        try:
            with open(ebook_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        ebook_text += text + " "
        except Exception as e:
            print("Error reading PDF ebook:", e)
    else:
        try:
            with open(ebook_file, "r", encoding="utf-8") as f:
                ebook_text = f.read()
        except Exception as e:
            print("Error reading ebook:", e)
    return ebook_text

def extract_book_title(ebook_text):
    """
    Extract the first line of the ebook as the title.

    Args:
        ebook_text (str): Ebook text.

    Returns:
        str: Title string.
    """
    for line in ebook_text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""

def align_transcript_to_ebook(full_transcript, ebook_text):
    """
    Check alignment between transcript and ebook.

    Args:
        full_transcript (str): Audio transcript.
        ebook_text (str): Ebook text.

    Returns:
        str: The ebook text.
    """
    ratio = difflib.SequenceMatcher(None, full_transcript, ebook_text).ratio()
    print(f"Global alignment similarity: {ratio:.2f}")
    return ebook_text  # Use ebook text as the gold standard

# ---------------------------
# Dialogue Extraction from Ebook Text
# ---------------------------
def extract_dialogue_intervals(ebook_text, audio_boundaries, gap_threshold=10):
    """
    Identify dialogue intervals by mapping ebook quotes to audio timestamps.

    Args:
        ebook_text (str): Ebook text.
        audio_boundaries (list): Audio segment boundaries.
        gap_threshold (int): Max chars between grouped quotes.

    Returns:
        list of tuple: Dialogue segments (start, end, text).
    """
    dialogue_matches = list(re.finditer(r'[“"](.+?)[”"]', ebook_text, re.DOTALL))
    grouped_matches = []
    if dialogue_matches:
        current_group = [dialogue_matches[0]]
        for m in dialogue_matches[1:]:
            prev_end = current_group[-1].end()
            in_between = ebook_text[prev_end:m.start()]
            if len(in_between.strip()) < gap_threshold and "\n" not in in_between:
                current_group.append(m)
            else:
                grouped_matches.append(current_group)
                current_group = [m]
        grouped_matches.append(current_group)
    else:
        grouped_matches = []
    dialogue_subtitles = []
    total_chars = len(ebook_text)
    if not audio_boundaries:
        return dialogue_subtitles
    audio_start = audio_boundaries[0][0]
    audio_end = audio_boundaries[-1][1]
    duration = audio_end - audio_start
    for group in grouped_matches:
        group_text = " ".join(m.group(1).strip() for m in group)
        group_start = group[0].start()
        group_end = group[-1].end()
        start_frac = group_start / total_chars
        end_frac = group_end / total_chars
        start_time = audio_start + start_frac * duration
        end_time = audio_start + end_frac * duration
        dialogue_subtitles.append((start_time, end_time, group_text))
    return dialogue_subtitles

def extract_narration_intervals(boundaries, dialogue_subtitles):
    """
    Identify narration intervals (non-dialogue).

    Args:
        boundaries (list): Audio segments.
        dialogue_subtitles (list): Dialogue segments.

    Returns:
        list of tuple: Narration segments.
    """
    narration_subtitles = []
    for b in boundaries:
        b_start, b_end, b_text = b
        if not any(not (b_end <= d_start or b_start >= d_end) for d_start, d_end, _ in dialogue_subtitles):
            narration_subtitles.append((b_start, b_end, b_text))
    return narration_subtitles

# ---------------------------
# Known Characters (from "Of Mice and Men")
# ---------------------------
known_characters = {
    "George": "male",
    "Lennie": "male",
    "Curley's Wife": "female",
    "Candy": "male",
    "Crooks": "male",
    "Slim": "male",
    "Curley": "male",
    "Carlson": "male",
    "Boss": "male",
    "The Boss": "male"
}

# ---------------------------
# Speaker Determination via DeepSeek (Text-Based)
# ---------------------------
def get_speaker_for_dialogue(dialogue_text, ebook_text, seg_start, seg_end, prev_speaker=None, book_title=""):
    """
    Determine speaker for a dialogue line using LLM (Ollama).

    Args:
        dialogue_text (str): The dialogue.
        ebook_text (str): Ebook context.
        seg_start (float): Start time (unused).
        seg_end (float): End time (unused).
        prev_speaker (str): Previous speaker name.
        book_title (str): Title of the book.

    Returns:
        str: Speaker assignment string.
    """
    model_id = "deepseek-r1:8b"
    window_size = 2000
    max_window = 10000
    best_result = None
    final_window_used = window_size
    while window_size <= max_window:
        idx = ebook_text.find(dialogue_text)
        if idx != -1:
            start_ctx = max(0, idx - window_size)
            end_ctx = idx + len(dialogue_text) + window_size
            context_window = ebook_text[start_ctx:end_ctx]
        else:
            context_window = ebook_text[:window_size]
        if book_title:
            context_window = f"Book: {book_title}\n" + context_window
        prompt = """You are an expert assistant in classic literature with deep knowledge of "Of Mice and Men". Your task is to determine the speaker of the dialogue excerpt below and describe their speech style. Follow these steps exactly:
1. Read the dialogue excerpt carefully.
2. Look in the provided context for explicit dialogue tags (e.g., "George said", "Lennie cried"). If found, use that name.
3. If no explicit tag is present, analyze the surrounding narrative to:
   a. Identify which character the paragraph primarily describes.
   b. Look for action descriptions and pronoun references that indicate who is acting.
   c. IMPORTANT: If the dialogue begins with an addressed name (e.g., "Look, George, ..."), ignore that name unless the dialogue is clearly self-directed.
   d. Consider connected dialogue lines (separated by short gaps or action tags) as one block.
4. Use your detailed knowledge of "Of Mice and Men" to decide.
5. If no clear name is evident, assign a descriptive tag (e.g., "Unnamed1").
6. On a new line, output exactly:
Answer: <Name> <Gender>
On the next line, output exactly:
SpeechStyle: <Style>
where <Name> is the speaker’s name (without extra words), <Gender> is either "male" or "female", and <Style> is a brief description of their speech characteristics (e.g., "gruff and impatient", "soft and gentle"). Do not include any extra commentary.

Dialogue: {dialogue_text}

Context: {context_window}

Speaker?"""
        prompt = prompt.format(dialogue_text=dialogue_text, context_window=context_window)
        command = ["ollama", "run", model_id]
        try:
            result = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True
            )
        except subprocess.CalledProcessError as e:
            print("DeepSeek speaker determination error:", e.stderr)
            window_size += 2000
            continue
        output = remove_think_tags(remove_ansi_escape(result.stdout)).strip()
        print("DEBUG: Window size", window_size, "Raw DeepSeek output:", output)
        answer_line = None
        style_line = None
        for line in output.splitlines()[::-1]:
            if line.strip().lower().startswith("answer:"):
                answer_line = line.strip()
            if line.strip().lower().startswith("speechstyle:"):
                style_line = line.strip()
            if answer_line and style_line:
                break
        if answer_line:
            match = re.search(r'Answer:\s*([A-Za-z]+)\s+(male|female)\b', answer_line, re.IGNORECASE)
            if match:
                speaker_name = sanitize_speaker_name(match.group(1))
                gender = match.group(2).lower()
                best_result = f"{speaker_name} {gender}"
        final_window_used = window_size
        if best_result is not None:
            break
        window_size += 2000

    if best_result is None or best_result.lower().startswith("unknown"):
        best_result = fallback_speaker(dialogue_text, prev_speaker)
    fallback_result = fallback_speaker(dialogue_text, prev_speaker)
    DEBUG_LOG.append({
        "dialogue": dialogue_text,
        "context_window_size": final_window_used,
        "raw_output": output,
        "text_based_result": best_result,
        "speech_style": style_line if style_line else "N/A",
        "fallback_result": fallback_result,
        "final_result": best_result
    })
    return best_result

def fallback_speaker(dialogue_text, prev_speaker):
    """
    Keyword-based fallback speaker attribution.

    Args:
        dialogue_text (str): Dialogue text.
        prev_speaker (str): Previous speaker.

    Returns:
        str: Guessed speaker.
    """
    found = []
    lower_text = dialogue_text.lower()
    for name, gender in known_characters.items():
        if name.lower() in lower_text:
            found.append((name, gender))
    if len(found) == 1:
        return f"{found[0][0]} {found[0][1]}"
    elif prev_speaker:
        for name, gender in known_characters.items():
            if name.lower() in lower_text and name.lower() not in prev_speaker.lower():
                return f"{name} {gender}"
    return "Unnamed1 unknown"

def update_descriptive_assignments(speaker_assignments, dialogue_subtitles, threshold=0.6):
    """
    Propagate assignments for 'unnamed' speakers using text similarity.

    Args:
        speaker_assignments (dict): Assignment map.
        dialogue_subtitles (list): Dialogue list.
        threshold (float): Similarity threshold.

    Returns:
        dict: Updated assignments.
    """
    for i in range(len(dialogue_subtitles)):
        if speaker_assignments[i].lower().startswith("unnamed"):
            for j in range(len(dialogue_subtitles)):
                if not speaker_assignments[j].lower().startswith("unnamed"):
                    sim = difflib.SequenceMatcher(None, dialogue_subtitles[i][2].lower(), dialogue_subtitles[j][2].lower()).ratio()
                    if sim > threshold:
                        speaker_assignments[i] = speaker_assignments[j]
                        break
    return speaker_assignments

# ---------------------------
# Audio Processing & SRT Generation
# ---------------------------
def merge_intervals(intervals):
    """
    Merge overlapping intervals.

    Args:
        intervals (list): List of (start, end) tuples.

    Returns:
        list: Merged intervals.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def mute_segments(audio, intervals):
    """
    Mute specified intervals in audio.

    Args:
        audio (AudioSegment): Audio data.
        intervals (list): List of (start, end) tuples.

    Returns:
        AudioSegment: Audio with muted intervals.
    """
    output = audio
    for start, end in intervals:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        silence = AudioSegment.silent(duration=(end_ms - start_ms))
        output = output[:start_ms] + silence + output[end_ms:]
    return output

def export_srt(subtitles, srt_path):
    """
    Export to SRT file (duplicate of helper function for consistency).

    Args:
        subtitles (list): List of subtitles.
        srt_path (str): Output path.
    """
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(subtitles, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")
    print(f"SRT file exported to: {srt_path}")

# ---------------------------
# Global Processing & Output Generation
# ---------------------------
def process_transcription_alternate(audio_file, ebook_file):
    """
    Main processing pipeline for a single file.

    Args:
        audio_file (str): Audio file path.
        ebook_file (str): Ebook file path.

    Returns:
        tuple: (dialogue_subtitles, narration_subtitles, duration, speaker_assignments)
    """
    result = transcribe_audio(audio_file)
    full_transcript, boundaries = merge_transcription(result)
    ebook_text = load_ebook_text(ebook_file)
    global_ebook = align_transcript_to_ebook(full_transcript, ebook_text)
    book_title = extract_book_title(ebook_text)
    if boundaries:
        audio_start = boundaries[0][0]
        audio_end = boundaries[-1][1]
    else:
        audio_start, audio_end = 0.0, 0.0
    dialogue_subtitles = extract_dialogue_intervals(global_ebook, boundaries)
    narration_subtitles = []
    for b in boundaries:
        b_start, b_end, b_text = b
        if not any(not (b_end <= d_start or b_start >= d_end) for d_start, d_end, _ in dialogue_subtitles):
            narration_subtitles.append((b_start, b_end, b_text))
    speaker_assignments = {}
    prev_speaker = None
    for idx, (d_start, d_end, dialogue_text) in enumerate(dialogue_subtitles):
        speaker = get_speaker_for_dialogue(dialogue_text, global_ebook, d_start, d_end, prev_speaker=prev_speaker, book_title=book_title)
        if speaker is None or speaker.lower().startswith("narrator"):
            speaker = "Unknown unknown"
        speaker_assignments[idx] = speaker
        prev_speaker = speaker
        print(f"Dialogue '{dialogue_text}' from {d_start:.2f}-{d_end:.2f} assigned to {speaker}.")
    speaker_assignments = update_descriptive_assignments(speaker_assignments, dialogue_subtitles, threshold=0.6)
    total_duration = result.get("duration", 0.0)
    return dialogue_subtitles, narration_subtitles, total_duration, speaker_assignments

def process_audiobook_alternate(audio_file, dialogue_subtitles, narration_subtitles, speaker_assignments, output_dir):
    """
    Generate output audio files (narrator and per-speaker tracks).

    Args:
        audio_file (str): Input audio file.
        dialogue_subtitles (list): Dialogue segments.
        narration_subtitles (list): Narration segments.
        speaker_assignments (dict): Speaker assignments.
        output_dir (str): Output folder.
    """
    print(f"Loading audiobook file: {audio_file}")
    audio = AudioSegment.from_file(audio_file)
    os.makedirs(output_dir, exist_ok=True)
    orig_name = os.path.basename(audio_file)

    # Create narrator track by muting dialogue intervals.
    all_dialogue = merge_intervals([(s, e) for (s, e, _) in dialogue_subtitles])
    narrator_audio = mute_segments(audio, all_dialogue)
    narrator_path = os.path.join(output_dir, f"narrator_{orig_name}")
    narrator_audio.export(narrator_path, format="wav")
    print(f"Narrator track saved to: {narrator_path}")
    narrator_srt = os.path.join(output_dir, f"narrator_{orig_name.rsplit('.',1)[0]}.srt")
    export_srt(narration_subtitles, narrator_srt)

    # Create individual speaker tracks by muting non-dialogue intervals.
    speaker_intervals = {}
    for idx, (s, e, txt) in enumerate(dialogue_subtitles):
        speaker = speaker_assignments.get(idx, "Unknown unknown")
        speaker_intervals.setdefault(speaker, []).append((s, e, txt))

    for speaker, intervals in speaker_intervals.items():
        merged_int = merge_intervals([(s, e) for (s, e, _) in intervals])
        non_speaker = []
        current = 0.0
        for s, e in merged_int:
            if s > current:
                non_speaker.append((current, s))
            current = e
        if current < audio.duration_seconds:
            non_speaker.append((current, audio.duration_seconds))
        speaker_audio = mute_segments(audio, non_speaker)
        parts = speaker.split()
        if len(parts) >= 2:
            sp_name, sp_gender = parts[0], parts[1]
        else:
            sp_name, sp_gender = speaker, "unknown"
        speaker_filename = f"{sp_name}_{sp_gender}_{orig_name}"
        speaker_path = os.path.join(output_dir, speaker_filename)
        speaker_audio.export(speaker_path, format="wav")
        print(f"{speaker} track saved to: {speaker_path}")
        speaker_srt = os.path.join(output_dir, f"{sp_name}_{sp_gender}_{orig_name.rsplit('.',1)[0]}.srt")
        export_srt(intervals, speaker_srt)

# ---------------------------
# GUI for Manual Dialogue Verification and Reassignment
# ---------------------------
class DialogueEditor(QtWidgets.QWidget):
    """
    GUI for verifying and correcting dialogue assignments.
    """
    def __init__(self, dialogue_subtitles, speaker_assignments, known_chars):
        """
        Initialize the DialogueEditor.

        Args:
            dialogue_subtitles (list): List of dialogue segments.
            speaker_assignments (dict): Initial speaker assignments.
            known_chars (dict): Dictionary of known characters.
        """
        super().__init__()
        self.dialogue_subtitles = dialogue_subtitles  # List of tuples (start, end, dialogue_text)
        self.speaker_assignments = speaker_assignments  # Dict: index -> speaker string
        self.known_chars = known_chars  # Dict of known characters
        self.init_ui()

    def init_ui(self):
        """Setup UI elements."""
        self.setWindowTitle("Dialogue Speaker Reassignment")
        layout = QtWidgets.QVBoxLayout(self)

        # Table to list dialogue segments
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Index", "Time", "Dialogue", "Speaker"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.populate_table()
        layout.addWidget(self.table)

        # Save button
        save_btn = QtWidgets.QPushButton("Save Assignments and Process Audio", self)
        save_btn.clicked.connect(self.save_and_process)
        layout.addWidget(save_btn)

    def populate_table(self):
        """Populate table with dialogue data."""
        self.table.setRowCount(len(self.dialogue_subtitles))
        for idx, (start, end, text) in enumerate(self.dialogue_subtitles):
            index_item = QtWidgets.QTableWidgetItem(str(idx))
            time_item = QtWidgets.QTableWidgetItem(f"{start:.2f}-{end:.2f}")
            dialogue_item = QtWidgets.QTableWidgetItem(text)
            self.table.setItem(idx, 0, index_item)
            self.table.setItem(idx, 1, time_item)
            self.table.setItem(idx, 2, dialogue_item)
            # Create dropdown for speaker selection
            combo = QtWidgets.QComboBox(self)
            for name in self.known_chars.keys():
                combo.addItem(name)
            combo.addItem("Unknown")
            current = self.speaker_assignments.get(idx, "Unknown")
            current_name = current.split()[0] if current != "Unknown unknown" else "Unknown"
            index_in_combo = combo.findText(current_name)
            if index_in_combo >= 0:
                combo.setCurrentIndex(index_in_combo)
            else:
                combo.setCurrentIndex(combo.findText("Unknown"))
            self.table.setCellWidget(idx, 3, combo)

    def save_and_process(self):
        """Save changes and close the window."""
        for idx in range(self.table.rowCount()):
            widget = self.table.cellWidget(idx, 3)
            if widget:
                selected = widget.currentText()
                gender = self.known_chars.get(selected, "unknown")
                self.speaker_assignments[idx] = f"{selected} {gender}"
        self.close()

# ---------------------------
# Main Program (CLI + GUI for Batch Processing)
# ---------------------------
def main():
    """
    Main function to run the CLI/GUI hybrid tool.

    Parses arguments, processes audio files, and launches the verification GUI for each.
    """
    parser = argparse.ArgumentParser(description="Forced Alignment & Manual Dialogue Verification GUI (Batch Mode)")
    parser.add_argument("--input_folder", required=True, help="Folder containing audiobook files")
    parser.add_argument("--ebook_folder", required=True, help="Folder containing the ebook DOCX file (only one expected)")
    parser.add_argument("--output_folder", required=True, help="Directory to save output files")
    args = parser.parse_args()

    ebook_file = None
    for file in os.listdir(args.ebook_folder):
        if file.lower().endswith('.docx'):
            ebook_file = os.path.join(args.ebook_folder, file)
            break
    if ebook_file is None:
        sys.exit("No DOCX ebook found in the specified folder.")

    audio_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)
                   if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not audio_files:
        sys.exit("No audio files found in the specified input folder.")

    for audio_file in audio_files:
        print(f"Processing file: {audio_file}")
        dialogue_subtitles, narration_subtitles, total_duration, speaker_assignments = process_transcription_alternate(audio_file, ebook_file=ebook_file)

        print("Initial Dialogue Intervals and Speaker Assignments:")
        for idx, (s, e, txt) in enumerate(dialogue_subtitles):
            print(f"  {idx}: {s:.2f}-{e:.2f}: {txt}")
            print(f"       Assigned: {speaker_assignments.get(idx, 'Unknown unknown')}")

        # Launch GUI for manual verification and reassignment for this file.
        app = QtWidgets.QApplication(sys.argv)
        editor = DialogueEditor(dialogue_subtitles, speaker_assignments, known_characters)
        editor.resize(800, 600)
        editor.show()
        app.exec_()

        print("Final Speaker Assignments after manual editing:")
        for idx, sp in speaker_assignments.items():
            print(f"  Dialogue index {idx}: {sp}")

        process_audiobook_alternate(audio_file, dialogue_subtitles, narration_subtitles, speaker_assignments, args.output_folder)

        # Export debug log for this file.
        debug_log_path = os.path.join(args.output_folder, f"{os.path.basename(audio_file).rsplit('.',1)[0]}_speaker_debug_log.txt")
        with open(debug_log_path, "w", encoding="utf-8") as f:
            for entry in DEBUG_LOG:
                f.write("Dialogue: " + entry["dialogue"] + "\n")
                f.write("Context window size: " + str(entry["context_window_size"]) + "\n")
                f.write("Raw DeepSeek output:\n" + entry["raw_output"] + "\n")
                f.write("Text-based result: " + str(entry["text_based_result"]) + "\n")
                f.write("SpeechStyle: " + str(entry["speech_style"]) + "\n")
                f.write("Fallback result: " + str(entry["fallback_result"]) + "\n")
                f.write("Final assigned: " + str(entry["final_result"]) + "\n")
                f.write("-----\n")
        print(f"Speaker debug log exported to: {debug_log_path}")

if __name__ == "__main__":
    main()
