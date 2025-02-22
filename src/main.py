import os
import argparse

# Base mode functions are imported only if needed.
def process_folder_base(input_folder, ebook, output_folder):
    from audio_bchampion.forced_alignment import transcribe_audio, load_ebook_text, get_dialogue_segments
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg')):
            audio_path = os.path.join(input_folder, file)
            out_dir = os.path.join(output_folder, os.path.splitext(file)[0])
            os.makedirs(out_dir, exist_ok=True)
            print(f"Processing (base mode): {audio_path}")
            dialogue_intervals = get_dialogue_segments(audio_path, ebook_file=ebook)
            transcription = transcribe_audio(audio_path)
            total_duration = transcription.get("duration", 0.0)
            narration_intervals = []
            current = 0.0
            for d_start, d_end in sorted(dialogue_intervals):
                if d_start > current:
                    narration_intervals.append((current, d_start))
                current = d_end
            if current < total_duration:
                narration_intervals.append((current, total_duration))
            from audio_bchampion.audio_processing import process_audiobook
            process_audiobook(audio_path, dialogue_intervals, narration_intervals, out_dir)

# Alternate mode functions: Import alternate functions only.
def process_folder_alternate(input_folder, ebook, output_folder):
    from audio_bchampion.forced_alignment import process_transcription_alternate, process_audiobook_alternate
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg')):
            audio_path = os.path.join(input_folder, file)
            out_dir = os.path.join(output_folder, os.path.splitext(file)[0])
            os.makedirs(out_dir, exist_ok=True)
            print(f"Processing (alternate mode): {audio_path}")
            speaker_dict, narration_intervals, duration = process_transcription_alternate(audio_path, ebook_file=ebook)
            print("Speaker intervals:", speaker_dict)
            print("Narration intervals:", narration_intervals)
            process_audiobook_alternate(
                audio_path, 
                {sp: [(s, e, txt) for (s, e, txt) in intervals] for sp, intervals in speaker_dict.items()},
                narration_intervals,
                out_dir
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audiobook Processing Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # Base mode for batch processing.
    base_parser = subparsers.add_parser("separate_folder", help="Process all audio files in input folder in base mode")
    base_parser.add_argument("--input_folder", default="Z:/AudioBChampion/data/input", help="Input folder (default: Z:/AudioBChampion/data/input)")
    base_parser.add_argument("--ebook", required=False, help="Path to the ebook file (PDF or text)")
    base_parser.add_argument("--output_folder", default="Z:/AudioBChampion/data/output", help="Output folder (default: Z:/AudioBChampion/data/output)")

    # Alternate mode for batch processing.
    alt_parser = subparsers.add_parser("separate_folder_alt", help="Process all audio files in input folder in alternate (speaker separation) mode")
    alt_parser.add_argument("--input_folder", default="Z:/AudioBChampion/data/input", help="Input folder (default: Z:/AudioBChampion/data/input)")
    alt_parser.add_argument("--ebook", required=False, help="Path to the ebook file (PDF or text)")
    alt_parser.add_argument("--output_folder", default="Z:/AudioBChampion/data/output", help="Output folder (default: Z:/AudioBChampion/data/output)")

    args = parser.parse_args()

    if args.command == "separate_folder":
        process_folder_base(args.input_folder, args.ebook, args.output_folder)
    elif args.command == "separate_folder_alt":
        process_folder_alternate(args.input_folder, args.ebook, args.output_folder)
