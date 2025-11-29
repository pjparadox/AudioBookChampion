import numpy as np
import noisereduce as nr
from pydub import AudioSegment

def reduce_noise(audio_segment):
    """
    Reduce background noise from a pydub AudioSegment using spectral gating.

    This function converts the audio segment to a numpy array, applies noise reduction
    using the `noisereduce` library, and then converts it back to a pydub AudioSegment.
    If the audio is stereo, it averages the channels before processing (converting to mono).

    Args:
        audio_segment (AudioSegment): The pydub AudioSegment to process.

    Returns:
        AudioSegment: A new AudioSegment with noise reduced.
    """
    print("Reducing noise on audio segment...")
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

    # If stereo, average the channels.
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
        samples = samples.mean(axis=1)

    reduced = nr.reduce_noise(y=samples, sr=audio_segment.frame_rate)

    reduced_int16 = np.int16(reduced)
    new_audio = AudioSegment(
        reduced_int16.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=1
    )
    return new_audio
