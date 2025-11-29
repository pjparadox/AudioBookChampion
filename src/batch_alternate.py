import os
import sys
import json
import argparse
import subprocess
import traceback
from tqdm import tqdm

from audio_bchampion.forced_alignment import process_transcription_alternate

STATE_FILENAME = "processing_state.json"

def save_state(state, output_folder, audio_filename):
    """
    Save the processing state to a JSON file.

    Args:
        state (dict): The state dictionary to save.
        output_folder (str): The folder where the state file will be saved.
        audio_filename (str): The name of the audio file associated with this state.

    Returns:
        str: The path to the saved state file.
    """
    base = os.path.basename(audio_filename).rsplit('.', 1)[0]
    state_path = os.path.join(output_folder, f"{base}_{STATE_FILENAME}")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"[INFO] Processing state saved to: {state_path}")
    return state_path

def load_state(state_path):
    """
    Load the processing state from a JSON file.

    Args:
        state_path (str): The path to the state file.

    Returns:
        dict: The loaded state dictionary.
    """
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    print(f"[INFO] Loaded processing state from: {state_path}")
    return state

def process_file(audio_file, ebook_file, output_folder, resume=False):
    """
    Process a single audio file to extract dialogue and assign speakers.

    Args:
        audio_file (str): Path to the audio file.
        ebook_file (str): Path to the ebook file.
        output_folder (str): Folder to save output.
        resume (bool): If True, attempts to load existing state.

    Returns:
        dict or None: The processing state if successful, None otherwise.
    """
    print(f"\n[INFO] Processing file: {audio_file}")
    base = os.path.basename(audio_file).rsplit('.', 1)[0]
    state_path = os.path.join(output_folder, f"{base}_{STATE_FILENAME}")
    if resume and os.path.exists(state_path):
        state = load_state(state_path)
    else:
        try:
            dialogue_subtitles, narration_subtitles, total_duration, speaker_assignments = process_transcription_alternate(
                audio_file, ebook_file=ebook_file
            )
            state = {
                "dialogue_subtitles": dialogue_subtitles,
                "narration_subtitles": narration_subtitles,
                "speaker_assignments": speaker_assignments,
                "total_duration": total_duration,
                "audio_file": audio_file
            }
            save_state(state, output_folder, audio_file)
        except Exception as e:
            print(f"[ERROR] Error processing {audio_file}:\n{traceback.format_exc()}")
            state = None
    return state

def launch_gui(state_path):
    """
    Launch the Live Dialogue GUI for a specific state file.

    Args:
        state_path (str): Path to the state file to open in the GUI.

    Returns:
        None
    """
    cmd = [
        sys.executable,
        "-m", "audio_bchampion.dialogue_live_gui",
        "--state", state_path
    ]
    print(f"[INFO] Launching GUI for state file: {state_path}")
    try:
        subprocess.Popen(cmd)
    except Exception as e:
        print(f"[ERROR] Error launching GUI for {state_path}: {e}")

def main():
    """
    Main entry point for batch processing audio files.

    Scans the input folder for audio files, processes them sequentially,
    and launches the manual verification GUI for each processed file.
    """
    parser = argparse.ArgumentParser(
        description="Batch Processing of Audio Files with Live GUI for Manual Dialogue Speaker Reassignment"
    )
    parser.add_argument("--input_folder", required=True, help="Folder containing audiobook files")
    parser.add_argument("--ebook_folder", required=True, help="Folder containing the ebook DOCX file (only one expected)")
    parser.add_argument("--output_folder", required=True, help="Directory to save output files and state")
    parser.add_argument("--resume", action="store_true", help="Resume processing from saved state (if available)")
    args = parser.parse_args()

    ebook_file = None
    for file in os.listdir(args.ebook_folder):
        if file.lower().endswith('.docx'):
            ebook_file = os.path.join(args.ebook_folder, file)
            break
    if ebook_file is None:
        sys.exit("[FATAL] No DOCX ebook found in the specified folder.")

    audio_files = sorted(
        [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)
         if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    )
    if not audio_files:
        sys.exit("[FATAL] No audio files found in the specified input folder.")

    print("[INFO] Starting sequential processing of audio files...\n")
    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        state = process_file(audio_file, ebook_file, args.output_folder, resume=args.resume)
        if state is None:
            print(f"[WARN] Skipping {audio_file} due to processing error.")
            continue
        base = os.path.basename(audio_file).rsplit('.', 1)[0]
        state_path = os.path.join(args.output_folder, f"{base}_{STATE_FILENAME}")
        launch_gui(state_path)
    print("[INFO] All files have been processed. GUIs for manual editing have been launched for each file.")

if __name__ == "__main__":
    main()
