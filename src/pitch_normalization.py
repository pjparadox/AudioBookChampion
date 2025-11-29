import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm  # Added for CLI progress bar

def normalize_pitch_files(input_files, output_dir):
    """
    Normalize the pitch of multiple audio files to a common reference pitch.

    This function estimates the median pitch (using librosa.pyin) for each file in the
    input list. It then calculates a global reference pitch (median of all file pitches)
    and shifts each file so that its pitch matches the reference.

    Args:
        input_files (list of str): A list of file paths to the audio files to process.
        output_dir (str): The directory where the normalized files will be saved.
                          Files will be saved with a "normalized_" prefix.

    Returns:
        None
    """
    pitches = []

    print("Estimating pitches for input files...")
    # Estimate pitch for each file.
    for file in tqdm(input_files, desc="Estimating pitch"):
        print(f"Processing {file}...")
        y, sr = librosa.load(file, sr=None)
        pitch, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch = pitch[~np.isnan(pitch)]
        estimated_pitch = np.median(pitch) if len(pitch) > 0 else 0
        print(f"Estimated pitch for {file}: {estimated_pitch:.2f} Hz")
        pitches.append(estimated_pitch)

    valid_pitches = [p for p in pitches if p > 0]
    if not valid_pitches:
        print("No valid pitch estimates found. Aborting pitch normalization.")
        return
    reference_pitch = np.median(valid_pitches)
    print(f"Reference pitch set to: {reference_pitch:.2f} Hz")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize each file, showing progress with tqdm.
    for file, file_pitch in tqdm(zip(input_files, pitches), total=len(input_files), desc="Normalizing files"):
        if file_pitch == 0:
            shift = 0
        else:
            shift = 12 * np.log2(reference_pitch / file_pitch)
        print(f"Shifting {file} by {shift:.2f} semitones...")
        y, sr = librosa.load(file, sr=None)
        y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=shift)
        base_name = os.path.basename(file)
        output_file = os.path.join(output_dir, f"normalized_{base_name}")
        sf.write(output_file, y_shifted, sr)
        print(f"Saved normalized file: {output_file}")

    print("Pitch normalization complete.")
