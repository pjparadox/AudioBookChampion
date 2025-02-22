import os
from pydub import AudioSegment

def mute_segments(audio, segments):
    """
    Given an AudioSegment and a list of segments (start, end in seconds), mute these segments.
    Returns a new AudioSegment with the specified segments replaced by silence.
    """
    output = audio
    for start, end in segments:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        silence = AudioSegment.silent(duration=(end_ms - start_ms))
        output = output[:start_ms] + silence + output[end_ms:]
    return output

def get_non_dialogue_segments(total_duration, dialogue_segments):
    """
    Given the total duration (in seconds) and a list of dialogue segments,
    compute and return the complementary segments as a list of (start, end) tuples.
    """
    non_dialogue_segments = []
    current = 0.0
    for seg in sorted(dialogue_segments):
        if seg[0] > current:
            non_dialogue_segments.append((current, seg[0]))
        current = seg[1]
    if current < total_duration:
        non_dialogue_segments.append((current, total_duration))
    return non_dialogue_segments

def process_audiobook(audiobook_file, dialogue_segments, output_dir):
    """
    Process the audiobook file to produce:
      - Narrator track (with dialogue muted)
      - Dialogue track (with narration muted)
    The output files will have the same duration as the original.
    
    Files are saved as "narrator_track.wav" and "dialogue_track.wav" in output_dir.
    If a file exists, a numeric suffix is added to avoid overwriting.
    """
    print(f"Loading audiobook file: {audiobook_file}")
    audio = AudioSegment.from_file(audiobook_file)
    duration_sec = audio.duration_seconds
    
    # For narrator track: mute dialogue segments.
    print("Creating narrator track (muting dialogue)...")
    narrator_audio = mute_segments(audio, dialogue_segments)
    
    # For dialogue track: mute non-dialogue segments.
    non_dialogue_segments = get_non_dialogue_segments(duration_sec, dialogue_segments)
    print("Creating dialogue track (muting narration)...")
    dialogue_audio = mute_segments(audio, non_dialogue_segments)
    
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    narrator_output_path = os.path.join(output_dir, "narrator_track.wav")
    dialogue_output_path = os.path.join(output_dir, "dialogue_track.wav")
    
    # Ensure filenames are unique (avoid overwriting)
    narrator_output_path = _ensure_unique_filename(narrator_output_path)
    dialogue_output_path = _ensure_unique_filename(dialogue_output_path)
    
    print(f"Exporting narrator track to: {narrator_output_path}")
    narrator_audio.export(narrator_output_path, format="wav")
    print(f"Exporting dialogue track to: {dialogue_output_path}")
    dialogue_audio.export(dialogue_output_path, format="wav")
    
    print("Audiobook processing complete.")

def _ensure_unique_filename(filepath):
    """
    If a file already exists at filepath, append a numeric suffix to avoid overwriting.
    """
    base, ext = os.path.splitext(filepath)
    counter = 1
    unique_path = filepath
    while os.path.exists(unique_path):
        unique_path = f"{base}_{counter}{ext}"
        counter += 1
    return unique_path
