# AudioBChampion

AudioBChampion is a professional audiobook processing tool that:

- **Separates an audiobook** into two tracks:
  - A **narrator track** (with dialogue muted)
  - A **dialogue track** (with narration muted)
- Uses forced alignment with an ebook (text or PDF) to detect dialogue segments.
- If no ebook is available, falls back to automatically transcribing the audio (using OpenAI Whisper) and detecting dialogue based on punctuation (quotation marks).
- Provides **pitch normalization** across multiple audio files.
- Offers bonus features such as background noise reduction.
- Provides both a **GUI** (built with PyQt5) and a **CLI** mode.
- Leverages your NVIDIA RTX 4090 (via PyTorch and Whisper) for GPU‑accelerated processing.

## Directory Structure

Place the project in:  
`Z:\AudioBChampion`

