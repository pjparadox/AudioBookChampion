import os
import sys
import re
import difflib
import subprocess
import argparse
import json

import torch
import whisper
import PyPDF2  # For PDF ebook fallback
from pydub import AudioSegment
from tqdm import tqdm

try:
    from docx import Document
except ImportError:
    sys.exit("Please install python-docx (pip install python-docx) to read DOCX files.")

# ---------------------------
# Global Debug Log for Speaker Attribution
# ---------------------------
DEBUG_LOG = []  # This will collect detailed debug info for each dialogue block

# Global variables for model and processing version selection
SELECTED_OLLAMA_MODEL = None
SELECTED_PROCESSING_VERSION = None

# Global flag for full chain-of-thought debug output (default: False)
SHOW_DEBUG = False

# ---------------------------
# Helper Functions
# ---------------------------
def remove_ansi_escape(text):
    """
    Remove ANSI escape sequences from a string.

    Args:
        text (str): The text containing ANSI codes.

    Returns:
        str: The cleaned text.
    """
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def remove_think_tags(text):
    """
    Remove <think>...</think> tags and their content from the text.
    These tags are often produced by reasoning models.

    Args:
        text (str): The text containing think tags.

    Returns:
        str: The text with think tags removed.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def sanitize_speaker_name(name):
    """
    Sanitize a speaker name by removing non-alphanumeric characters (except underscores).

    Args:
        name (str): The raw speaker name.

    Returns:
        str: The sanitized speaker name.
    """
    return re.sub(r'[^A-Za-z0-9_]', '', name)

def format_timestamp(seconds):
    """
    Format seconds into an SRT timestamp string (HH:MM:SS,mmm).

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The formatted timestamp string.
    """
    millis = int((seconds - int(seconds)) * 1000)
    s = int(seconds)
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

def export_srt(subtitles, srt_path):
    """
    Export subtitles to an SRT file.

    Args:
        subtitles (list of tuple): A list of (start_time, end_time, text) tuples.
        srt_path (str): The file path to save the SRT file.

    Returns:
        None
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
    Transcribe an audio file using OpenAI's Whisper model.

    Loads the 'base' model on CUDA if available, otherwise CPU.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        dict: The transcription result containing 'segments' and other metadata.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    return model.transcribe(audio_file)

def merge_transcription(result):
    """
    Merge transcription segments into a single text block and a list of boundaries.

    Args:
        result (dict): The result dictionary from `transcribe_audio`.

    Returns:
        tuple:
            - full_text (str): The concatenated transcription text.
            - boundaries (list of tuple): List of (start, end, text) for each segment.
    """
    full_text = ""
    boundaries = []
    for seg in result.get("segments", []):
        boundaries.append((seg["start"], seg["end"], seg["text"].strip()))
        full_text += " " + seg["text"].strip()
    return full_text.strip(), boundaries

# ---------------------------
# Ebook Loading and Title Extraction
# ---------------------------
def load_ebook_text(ebook_file):
    """
    Load text from an ebook file (.docx, .pdf, or plain text).

    Args:
        ebook_file (str): Path to the ebook file.

    Returns:
        str: The extracted text from the ebook.
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
    Attempt to extract the book title from the first non-empty line of the text.

    Args:
        ebook_text (str): The full text of the ebook.

    Returns:
        str: The extracted title, or an empty string if not found.
    """
    for line in ebook_text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""

def align_transcript_to_ebook(full_transcript, ebook_text):
    """
    Compute alignment similarity between the transcript and the ebook text.

    Args:
        full_transcript (str): The full audio transcription.
        ebook_text (str): The full ebook text.

    Returns:
        str: The ebook text (currently just returns it, but calculates ratio for logging).
    """
    ratio = difflib.SequenceMatcher(None, full_transcript, ebook_text).ratio()
    print(f"Global alignment similarity: {ratio:.2f}")
    return ebook_text

def determine_local_ebook_range(full_transcript, ebook_text):
    """
    Determine the subset of the ebook text that corresponds to the given transcript.

    This attempts to find the start and end of the transcript within the ebook
    to avoid searching the entire book for every dialogue line.

    Args:
        full_transcript (str): The full audio transcription.
        ebook_text (str): The full ebook text.

    Returns:
        str: A substring of the ebook text that roughly matches the transcript coverage.
    """
    header = full_transcript[:300].strip()
    footer = full_transcript[-300:].strip()
    start_idx = ebook_text.lower().find(header.lower())
    if start_idx == -1:
        paragraphs = re.split(r'\n\s*\n', ebook_text)
        for para in paragraphs:
            if not re.search(r'[“"]', para):
                start_idx = ebook_text.find(para)
                break
        else:
            start_idx = 0
    end_idx = ebook_text.lower().find(footer.lower())
    if end_idx == -1 or end_idx < start_idx:
        end_idx = len(ebook_text)
    end_idx = min(end_idx + 500, len(ebook_text))
    local_text = ebook_text[start_idx:end_idx]
    print(f"[DEBUG] Local ebook range determined: start={start_idx}, end={end_idx}, length={len(local_text)}")
    return local_text

# ---------------------------
# Dialogue Extraction from Ebook Text
# ---------------------------
def extract_dialogue_intervals(ebook_text, audio_boundaries, gap_threshold=10):
    """
    Extract dialogue intervals by mapping quoted text in the ebook to audio time boundaries.

    It finds all quoted text in the ebook, groups adjacent quotes, and then maps these groups
    to the audio timeline based on character position ratios.

    Args:
        ebook_text (str): The ebook text.
        audio_boundaries (list of tuple): List of (start, end, text) from the audio transcription.
        gap_threshold (int): Max characters between quotes to consider them part of the same group.

    Returns:
        list of tuple: A list of (start_time, end_time, dialogue_text) for each dialogue segment.
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
    Identify narration intervals by checking audio boundaries against dialogue intervals.

    Args:
        boundaries (list of tuple): List of (start, end, text) from audio transcription.
        dialogue_subtitles (list of tuple): List of (start, end, text) identified as dialogue.

    Returns:
        list of tuple: A list of (start, end, text) that are considered narration.
    """
    narration_subtitles = []
    for b in boundaries:
        b_start, b_end, b_text = b
        if not re.search(r'[“"]', b_text):
            narration_subtitles.append((b_start, b_end, b_text))
        elif not any(not (b_end <= d_start or b_start >= d_end) for d_start, d_end, _ in dialogue_subtitles):
            narration_subtitles.append((b_start, b_end, b_text))
    return narration_subtitles

# ---------------------------
# Known Characters (from Of Mice and Men)
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
    "The Boss": "male",
    "Narrator": "unknown"
}

# ---------------------------
# Processing Version Selection Prompt
# ---------------------------
def select_processing_version():
    """
    Prompt the user via CLI to select a processing version (A or B).

    Version A: Immediate Narration Assignment if no quotes.
    Version B: Extended context window search before assigning Narration.

    Returns:
        str: "A" or "B".
    """
    global SELECTED_PROCESSING_VERSION
    if SELECTED_PROCESSING_VERSION is not None:
        return SELECTED_PROCESSING_VERSION
    print("Please select a processing version:")
    print("1: Version A - Immediate Narration Assignment: If the provided context contains no quotation marks, immediately assign 'Narrator unknown'.")
    print("2: Version B - Extended Narration Assignment: If no quotation marks exist in the initial 200-character context, expand the context window by 3000 characters and if still no quotes are found, assign 'Narrator unknown'; otherwise, process dialogue normally.")
    selection = input("Enter 1 for Version A, or 2 for Version B: ").strip()
    if selection == "1":
        SELECTED_PROCESSING_VERSION = "A"
    elif selection == "2":
        SELECTED_PROCESSING_VERSION = "B"
    else:
        print("Invalid selection, defaulting to Version A.")
        SELECTED_PROCESSING_VERSION = "A"
    print(f"Selected Processing Version: {SELECTED_PROCESSING_VERSION}")
    return SELECTED_PROCESSING_VERSION

# ---------------------------
# Ollama Model Selection
# ---------------------------
def select_ollama_model():
    """
    List available Ollama models and prompt the user to select one.

    Returns:
        str: The name of the selected model (e.g., "deepseek-r1:8b").
    """
    global SELECTED_OLLAMA_MODEL
    if SELECTED_OLLAMA_MODEL is not None:
        return SELECTED_OLLAMA_MODEL
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        lines = output.splitlines()
        if len(lines) < 2:
            print("No models found in ollama list output. Defaulting to deepseek-r1:8b.")
            SELECTED_OLLAMA_MODEL = "deepseek-r1:8b"
            return SELECTED_OLLAMA_MODEL
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        print("Available Ollama models:")
        for idx, model in enumerate(models, start=1):
            print(f"{idx}: {model}")
        selection = input("Enter the number corresponding to the model you want to use: ").strip()
        try:
            sel_index = int(selection) - 1
            if sel_index < 0 or sel_index >= len(models):
                print("Invalid selection. Defaulting to deepseek-r1:8b.")
                SELECTED_OLLAMA_MODEL = "deepseek-r1:8b"
            else:
                SELECTED_OLLAMA_MODEL = models[sel_index]
        except ValueError:
            print("Invalid input. Defaulting to deepseek-r1:8b.")
            SELECTED_OLLAMA_MODEL = "deepseek-r1:8b"
    except subprocess.CalledProcessError as e:
        print("Error listing ollama models:", e.stderr)
        SELECTED_OLLAMA_MODEL = "deepseek-r1:8b"
    print(f"Selected model: {SELECTED_OLLAMA_MODEL}")
    return SELECTED_OLLAMA_MODEL

# ---------------------------
# DeepSeek Prompt Template for Blocks
# ---------------------------
DEEPSEEK_PROMPT_BLOCK = (
    "You are an expert assistant in classic literature with deep knowledge of 'Of Mice and Men'.\n"
    "Your task is to determine the speaker for each line in the dialogue block below and describe their speech style.\n"
    "Follow these steps exactly:\n"
  "You are an expert assistant in classic literature with deep knowledge of 'Of Mice and Men'.\n"
            "Your task is to determine the speaker of the dialogue excerpt below and describe their speech style. Follow these steps exactly:\n"
            "Step 1: If the provided context contains no quotation marks, assign 'Narrator unknown'.\n"
            "Step 2: Otherwise, check for explicit dialogue tags (e.g., 'said George' or 'George said') in the context and use that information, unless a name is used solely as an address.\n"
            "Step 3: Consider the subject of the paragraph and the flow of conversation. Use logical deduction to decide the speaker.\n"
            "Step 4: Examine the surrounding narrative in detail, to determine the identity of the speaker of the dialogue sample.\n"
            "Step 5: Treat all quoted text within a contiguous block (without line breaks) as coming from the same speaker, even if interrupted by dialogue tags or action descriptions.\n"
            "Step 6: If a line break appears immediately after a dialogue segment, consider it as a potential indicator of a new speaker, but not definitive.\n"
            "Step 7: Assume that dialogue included in paragraphs about a specific character is spoken by that character unless explicitly indicated otherwise.\n"
            "Step 8: If a name appears within quotation marks, assume it is NOT the speaker unless they are introducing themselves or talking to themselves.\n"
            "Step 9: As a final check, rely on your deep knowledge of 'Of Mice and Men' by Steinbeck to ensure that the attributed speaker makes sense within the narrative.\n"
    "Step 10: If a name appears solely as an address (e.g., 'Look, George, ...'), do not assign that name as the speaker unless the dialogue is self-directed.\n"
    "Step 11: Treat all quoted text within a contiguous block (without line breaks) as coming from the same speaker.\n"
    "Step 12: Return exactly one JSON object mapping each line number (as a string starting at '1') to an object with keys \"speaker\" and \"gender\".\n"
    "For example: {{\"1\": {{\"speaker\": \"George\", \"gender\": \"male\"}}, \"2\": {{\"speaker\": \"Lennie\", \"gender\": \"male\"}}}}\n"
    "Do not include any extra commentary.\n\n"
    "Dialogue Block:\n{block_text}\n\n"
    "Context (excerpt from ebook):\n{context_window}\n\n"
    "Book Title: {book_title}\n"
    "Previous Speaker: {prev_speaker}\n"
    "Speaker?"
)

# ---------------------------
# Speaker Determination via DeepSeek (Block-Based)
# ---------------------------
def get_speakers_for_block(block, ebook_text, book_title, prev_speaker=None):
    """
    Determine speakers for a block of dialogue using an LLM (Ollama).

    Args:
        block (list): A list of dialogue segments.
        ebook_text (str): The text of the ebook.
        book_title (str): The title of the book.
        prev_speaker (str, optional): The name of the previous speaker.

    Returns:
        dict: A dictionary mapping index strings ('1', '2', etc.) to speaker assignment strings (e.g., "George male").
    """
    block_text = ""
    for i, (_, _, dialogue_text) in enumerate(block):
        block_text += f"{i+1}. {dialogue_text}\n"
    window_size = 200  # starting with 200 characters
    max_window = 7000
    best_result = None
    output = ""
    while window_size <= max_window:
        context_window = ebook_text[:window_size]
        if book_title:
            context_window = f"Book: {book_title}\n" + context_window
        try:
            full_prompt = DEEPSEEK_PROMPT_BLOCK.format(
                block_text=block_text,
                context_window=context_window,
                book_title=book_title,
                prev_speaker=prev_speaker if prev_speaker else "None"
            )
        except KeyError as e:
            print("KeyError in prompt formatting:", e)
            return {}
        model_id = select_ollama_model()
        command = ["ollama", "run", model_id]
        try:
            result = subprocess.run(
                command,
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=True
            )
        except subprocess.CalledProcessError as e:
            print("DeepSeek block speaker determination error:", e.stderr)
            window_size += 3000
            print(f"Expanding context window to {window_size} characters.")
            continue
        output = remove_think_tags(remove_ansi_escape(result.stdout)).strip()
        if SHOW_DEBUG:
            print("DEBUG (block): Window size", window_size, "Raw DeepSeek output:", output)
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                best_result = {}
                for k, v in parsed.items():
                    if isinstance(v, dict):
                        best_result[k] = f"{v.get('speaker', 'Unknown')} {v.get('gender', 'unknown')}"
                break
        except Exception as e:
            print(f"Expanding context window to {window_size + 3000} characters (JSON parse error: {e}).")
            window_size += 3000
            continue
    if best_result is None:
        best_result = {}
        for i, (_, _, dialogue_text) in enumerate(block):
            best_result[str(i+1)] = get_speaker_for_dialogue(dialogue_text, ebook_text, 0, 0, prev_speaker, book_title)
    for key, val in best_result.items():
        DEBUG_LOG.append({
            "dialogue_line_number": key,
            "block_result": val
        })
    return best_result

# ---------------------------
# Speaker Determination via DeepSeek (Single Dialogue)
# ---------------------------
def get_speaker_for_dialogue(dialogue_text, ebook_text, seg_start, seg_end, prev_speaker=None, book_title=""):
    """
    Determine the speaker for a single dialogue line using an LLM.

    Args:
        dialogue_text (str): The dialogue text.
        ebook_text (str): The ebook text (local context).
        seg_start (float): Start time of segment (unused).
        seg_end (float): End time of segment (unused).
        prev_speaker (str, optional): The name of the previous speaker.
        book_title (str, optional): The title of the book.

    Returns:
        str: The assigned speaker string (e.g., "George male").
    """
    model_id = select_ollama_model()
    window_size = 200  # starting at 200 characters
    max_window = 7000
    best_result = None
    output = ""
    style_line = None

    # If the dialogue text appears in a block without any quotes, return Narrator unknown immediately.
    paragraphs = re.split(r'\n\s*\n', ebook_text)
    for para in paragraphs:
        if dialogue_text in para:
            if not re.search(r'[“"]', para):
                return "Narrator unknown"
            break

    local_ebook = determine_local_ebook_range(full_transcript=ebook_text, ebook_text=ebook_text)
    while window_size <= max_window:
        idx = local_ebook.find(dialogue_text)
        if idx != -1:
            start_ctx = max(0, idx - window_size)
            end_ctx = idx + len(dialogue_text) + window_size
            context_window = local_ebook[start_ctx:end_ctx]
        else:
            context_window = local_ebook[:window_size]
        if book_title:
            context_window = f"Book: {book_title}\n" + context_window

        # Immediate narration assignment if no quotes are present:
        if not re.search(r'[“"]', context_window):
            if SELECTED_PROCESSING_VERSION == "A":
                best_result = "Narrator unknown"
                break
            elif SELECTED_PROCESSING_VERSION == "B":
                # In Version B, expand the context window by 3000 characters
                new_window_size = window_size + 3000
                expanded_context = local_ebook[:new_window_size]
                if not re.search(r'[“"]', expanded_context):
                    best_result = "Narrator unknown"
                    break
                else:
                    context_window = expanded_context

        prompt = (
            "You are an expert assistant in classic literature with deep knowledge of 'Of Mice and Men'.\n"
            "Your task is to determine the speaker of the dialogue excerpt below and describe their speech style. Follow these steps exactly:\n"
            "Step 1: If the provided context contains no quotation marks, assign 'Narrator unknown'.\n"
            "Step 2: Otherwise, check for explicit dialogue tags (e.g., 'said George' or 'George said') in the context and use that information, unless a name is used solely as an address.\n"
            "Step 3: Consider the subject of the paragraph and the flow of conversation. Use logical deduction to decide the speaker.\n"
            "Step 4: Examine the surrounding narrative in detail, to determine the identity of the speaker of the dialogue sample.\n"
            "Step 5: Treat all quoted text within a contiguous block (without line breaks) as coming from the same speaker, even if interrupted by dialogue tags or action descriptions.\n"
            "Step 6: If a line break appears immediately after a dialogue segment, consider it as a potential indicator of a new speaker, but not definitive.\n"
            "Step 7: Assume that dialogue included in paragraphs about a specific character is spoken by that character unless explicitly indicated otherwise.\n"
            "Step 8: If a name appears within quotation marks, assume it is NOT the speaker unless they are introducing themselves or talking to themselves.\n"
            "Step 9: As a final check, rely on your deep knowledge of 'Of Mice and Men' by Steinbeck to ensure that the attributed speaker makes sense within the narrative.\n"
            "Return exactly two lines:\n"
            "Answer: <Name> <Gender>\n"
            "SpeechStyle: <Style>\n"
            "Do not include any extra commentary.\n\n"
            "Dialogue: {dialogue_text}\n\n"
            "Context: {context_window}\n\n"
            "Speaker?"
        )
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
            window_size += 3000
            print(f"Expanding context window to {window_size} characters for dialogue segment.")
            continue
        output = remove_think_tags(remove_ansi_escape(result.stdout)).strip()
        if SHOW_DEBUG:
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
            match = re.search(r'Answer:\s*([A-Za-z\'\-]+)\s+(male|female)\b', answer_line, re.IGNORECASE)
            if match:
                speaker_name = sanitize_speaker_name(match.group(1))
                gender = match.group(2).lower()
                best_result = f"{speaker_name} {gender}"
                break
        window_size += 3000
        print(f"Expanding context window to {window_size} characters for dialogue segment.")
    if best_result is None or best_result.lower().startswith("unknown"):
        best_result = fallback_speaker(dialogue_text, prev_speaker)
    fallback_result = fallback_speaker(dialogue_text, prev_speaker)
    DEBUG_LOG.append({
        "dialogue": dialogue_text,
        "context_window_size": window_size,
        "raw_output": output,
        "text_based_result": best_result,
        "speech_style": style_line if style_line else "N/A",
        "fallback_result": fallback_result,
        "final_result": best_result
    })
    return best_result

def fallback_speaker(dialogue_text, prev_speaker):
    """
    Attempt to identify a speaker using simple keyword matching if the LLM fails.

    Args:
        dialogue_text (str): The dialogue text.
        prev_speaker (str): The previous speaker.

    Returns:
        str: A guessed speaker string or "Unnamed1 unknown".
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
    Propagate assignments from known speakers to "unnamed" ones if the text is similar.

    Args:
        speaker_assignments (dict): The current speaker assignment map.
        dialogue_subtitles (list): The list of dialogue subtitles.
        threshold (float): Similarity threshold for text matching.

    Returns:
        dict: The updated speaker assignments.
    """
    for i in range(len(dialogue_subtitles)):
        current = speaker_assignments.get(str(i), "")
        if current.lower().startswith("unnamed"):
            for j in range(len(dialogue_subtitles)):
                other = speaker_assignments.get(str(j), "")
                if not other.lower().startswith("unnamed"):
                    sim = difflib.SequenceMatcher(None, dialogue_subtitles[i][2].lower(), dialogue_subtitles[j][2].lower()).ratio()
                    if sim > threshold:
                        speaker_assignments[str(i)] = speaker_assignments[str(j)]
                        break
    return speaker_assignments

def merge_intervals(intervals):
    """
    Merge overlapping or adjacent time intervals.

    Args:
        intervals (list of tuple): List of (start, end) or (start, end, ...) tuples.

    Returns:
        list of tuple: Merged list of (start, end, ...) tuples.
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
    Mute specified intervals in an audio segment.

    Args:
        audio (AudioSegment): The audio segment.
        intervals (list of tuple): List of (start, end) tuples.

    Returns:
        AudioSegment: The audio segment with muted intervals.
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
    Export subtitles to an SRT file.

    Args:
        subtitles (list of tuple): List of (start, end, text) tuples.
        srt_path (str): The output file path.

    Returns:
        None
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
    Main orchestration function for processing an audiobook file.

    It performs transcription, alignment, dialogue extraction, and speaker attribution.

    Args:
        audio_file (str): Path to the audio file.
        ebook_file (str): Path to the ebook file.

    Returns:
        tuple:
            - dialogue_subtitles (list)
            - narration_subtitles (list)
            - total_duration (float)
            - speaker_assignments (dict)
    """
    select_processing_version()
    result = transcribe_audio(audio_file)
    full_transcript, boundaries = merge_transcription(result)
    ebook_text = load_ebook_text(ebook_file)
    local_ebook = determine_local_ebook_range(full_transcript, ebook_text)
    global_ebook = local_ebook
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
        if not re.search(r'[“"]', b_text):
            narration_subtitles.append((b_start, b_end, b_text))
        elif not any(not (b_end <= d_start or b_start >= d_end) for d_start, d_end, _ in dialogue_subtitles):
            narration_subtitles.append((b_start, b_end, b_text))
    speaker_assignments = {}
    prev_speaker = None
    block_size = 40  # processing 40 segments at a time
    for block_start in range(0, len(dialogue_subtitles), block_size):
        block = dialogue_subtitles[block_start:block_start+block_size]
        block_speakers = get_speakers_for_block(block, global_ebook, book_title, prev_speaker=prev_speaker)
        for i, seg in enumerate(block):
            global_idx = block_start + i
            speaker = block_speakers.get(str(i+1), "Narrator unknown")
            if speaker is None or (isinstance(speaker, str) and speaker.lower().startswith("narrator")):
                speaker = "Narrator unknown"
            speaker_assignments[global_idx] = speaker
            prev_speaker = speaker
            d_start, d_end, dialogue_text = seg
            print(f"Dialogue '{dialogue_text}' from {d_start:.2f}-{d_end:.2f} assigned to {speaker}.")
    speaker_assignments = update_descriptive_assignments(speaker_assignments, dialogue_subtitles, threshold=0.6)
    total_duration = result.get("duration", 0.0)
    state = {
        "dialogue_subtitles": dialogue_subtitles,
        "narration_subtitles": narration_subtitles,
        "speaker_assignments": {str(k): v for k, v in speaker_assignments.items()},
        "duration": total_duration,
        "audiobook": audio_file
    }
    state_filename = os.path.splitext(os.path.basename(audio_file))[0] + "_processing_state.json"
    output_dir = os.path.join(os.path.dirname(ebook_file), "output")
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, state_filename)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    print(f"[INFO] Processing state saved to: {state_path}")
    return dialogue_subtitles, narration_subtitles, total_duration, speaker_assignments

def process_audiobook_alternate(audio_file, dialogue_subtitles, narration_subtitles, speaker_assignments, output_dir):
    """
    Generate output audio files based on the processed subtitles and assignments.

    Args:
        audio_file (str): Path to the source audio file.
        dialogue_subtitles (list): List of dialogue segments.
        narration_subtitles (list): List of narration segments.
        speaker_assignments (dict): Speaker assignment map.
        output_dir (str): Directory to save output files.

    Returns:
        None
    """
    print(f"Loading audiobook file: {audio_file}")
    audio = AudioSegment.from_file(audio_file)
    os.makedirs(output_dir, exist_ok=True)
    orig_name = os.path.basename(audio_file)

    # Create narrator track by muting dialogue intervals.
    all_dialogue = merge_intervals([(s, e) for (s, e, _) in dialogue_subtitles])
    narrator_audio = mute_segments(audio, all_dialogue)
    narrator_path = os.path.join(output_dir, f"narrator_{orig_name}")
    narrator_audio.export(narrator_path, format="mp3", bitrate="320k")
    print(f"Narrator track saved to: {narrator_path}")
    narrator_srt = os.path.join(output_dir, f"narrator_{os.path.splitext(orig_name)[0]}.srt")
    export_srt(narration_subtitles, narrator_srt)

    # Create individual speaker tracks by muting non-dialogue parts.
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
        speaker_audio.export(speaker_path, format="mp3", bitrate="320k")
        print(f"{speaker} track saved to: {speaker_path}")
        speaker_srt = os.path.join(output_dir, f"{sp_name}_{sp_gender}_{os.path.splitext(orig_name)[0]}.srt")
        export_srt(intervals, speaker_srt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forced Alignment & Speaker Attribution")
    parser.add_argument("--audiobook", required=True, help="Path to audiobook file")
    parser.add_argument("--ebook", required=True, help="Path to ebook file (DOCX or PDF)")
    args = parser.parse_args()

    dialogue_subtitles, narration_subtitles, total_duration, speaker_assignments = process_transcription_alternate(args.audiobook, ebook_file=args.ebook)
    print("Dialogue Intervals:")
    for idx, (s, e, txt) in enumerate(dialogue_subtitles):
        print(f"{idx}: {s:.2f}-{e:.2f}: {txt}")
        print(f"     Assigned: {speaker_assignments.get(str(idx), 'Unknown unknown')}")
