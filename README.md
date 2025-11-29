# AudioBChampion

AudioBChampion is a professional audiobook processing tool designed to automate the separation of narration and dialogue, and attribute dialogue to specific speakers using advanced AI techniques.

## Features

- **Audiobook Separation**: Splits an audiobook into two distinct tracks:
  - **Narrator Track**: Contains only the narration (dialogue is muted).
  - **Dialogue Track**: Contains only the dialogue (narration is muted).
- **Speaker Attribution**: Uses LLMs (via Ollama/DeepSeek) and text analysis to identify speakers for each dialogue segment.
- **Forced Alignment**: Aligns audio transcriptions with the original ebook text (DOCX/PDF) for high-precision dialogue extraction.
- **Pitch Normalization**: Normalizes the pitch of multiple audio files to a common reference.
- **Noise Reduction**: Optional background noise reduction using spectral gating.
- **Live GUI**: A PyQt5-based GUI for manual review, splitting, and reassignment of dialogue speakers.
- **GPU Acceleration**: Leverages NVIDIA GPUs via PyTorch and Whisper for fast transcription.

## Directory Structure

The recommended directory structure is:
```
Z:/AudioBChampion/
├── data/
│   ├── input/       # Place your source audiobook files here
│   ├── output/      # Processed files will be saved here
│   └── pdf/         # Place your ebook file (DOCX/PDF) here
└── src/             # Source code
```

## Setup

### Prerequisites

1.  **Python 3.8+**
2.  **FFmpeg**: Must be installed and available in your system PATH.
3.  **Ollama**: Installed and running (for speaker attribution).
    -   Pull the model: `ollama pull deepseek-r1:8b`
4.  **CUDA**: For GPU acceleration with Whisper (optional but recommended).

### Installation

1.  Clone the repository.
2.  Install the required Python packages:

```bash
pip install -r requirements.txt
```

*Note: You may need to install `torch` with CUDA support separately if the default installation does not detect your GPU.*

## Usage

### GUI Mode

The easiest way to use AudioBChampion is via the main GUI.

```bash
python -m audio_bchampion.gui
```

This interface allows you to:
- Select input audiobook and ebook files.
- Choose output directories.
- Toggle noise reduction.
- Run separation or pitch normalization tasks.

### CLI Batch Mode

For processing entire folders of audio files:

**1. Base Separation (Narrator vs. Dialogue only)**

```bash
python -m audio_bchampion.main separate_folder \
  --input_folder "Z:/AudioBChampion/data/input" \
  --ebook "Z:/AudioBChampion/data/pdf/my_book.docx" \
  --output_folder "Z:/AudioBChampion/data/output"
```

**2. Alternate Separation (Speaker Attribution)**

```bash
python -m audio_bchampion.main separate_folder_alt \
  --input_folder "Z:/AudioBChampion/data/input" \
  --ebook "Z:/AudioBChampion/data/pdf/my_book.docx" \
  --output_folder "Z:/AudioBChampion/data/output"
```

### Manual Review (Live GUI)

If you use `batch_alternate.py` or the alternate mode, the tool generates a processing state file. You can resume or review the assignments using the Live GUI:

```bash
python -m audio_bchampion.dialogue_live_gui \
  --state "path/to/audio_processing_state.json" \
  --ebook "path/to/ebook.docx"
```

## Development

- **Documentation**: All public functions and classes are documented using Google Style Python Docstrings.
- **Style**: The codebase follows standard Python conventions.

## License

[License Information]
