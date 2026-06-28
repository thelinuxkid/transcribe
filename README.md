# transcribe

Multi-speaker audio transcription with speaker diarization, optimized for Apple Silicon.

Combines [Whisper MLX](https://github.com/ml-explore/mlx-examples) for fast Metal-accelerated speech-to-text with [pyannote-audio](https://github.com/pyannote/pyannote-audio) for speaker identification. Outputs transcripts in Markdown, plain text, JSON, and SRT formats.

## Features

- **Fast transcription** on Apple Silicon via MLX (Metal GPU)
- **Speaker diarization** with pyannote — identifies and labels who said what
- **4 output formats** — Markdown, plain text, JSON, and SRT subtitles
- **Batch processing** — transcribe entire directories recursively
- **Smart skip logic** — won't re-transcribe existing files unless `--force` is passed
- **Diarize separately** — add speaker labels to an existing transcript without re-transcribing

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Miniforge](https://github.com/conda-forge/miniforge) (or any conda distribution)
- A [HuggingFace](https://huggingface.co) account and token (free, for diarization models)

## Setup

```bash
git clone <repo-url>
cd transcribe
bash setup.sh
```

`setup.sh` creates a conda environment called `transcribe` with Python 3.11, installs all dependencies (whispermlx, pyannote-audio, torch, ffmpeg), and installs the package in editable mode.

### HuggingFace token (required for diarization)

1. Create a token at https://huggingface.co/settings/tokens (Access: Read)
2. Accept the model terms (one-time, in browser):
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Save your token:
   ```bash
   conda activate transcribe
   huggingface-cli login
   ```
   Or set `export HF_TOKEN=hf_...` or pass `--hf-token hf_...` to commands.

## Usage

```bash
conda activate transcribe
```

### Transcribe audio

```bash
# Basic transcription (Spanish, large-v3-turbo model)
transcribe meeting.m4a

# With speaker diarization
transcribe meeting.m4a --diarize --speakers 3

# English audio with a specific model
transcribe interview.mp3 --language en --model large-v3

# Process a folder recursively
transcribe ./recordings/ --diarize --speakers 4

# Re-transcribe existing files
transcribe meeting.m4a --force
```

### Add speaker labels to an existing transcript

```bash
# Diarize using an existing transcript JSON
diarize meeting.m4a transcripts/meeting.json --speakers 3

# Auto-detect speaker count
diarize meeting.m4a transcripts/meeting.json --min-speakers 2 --max-speakers 6
```

## Output

Transcripts are written to `./transcripts/` by default (configurable with `--output`).

| File | Description |
|------|-------------|
| `name.md` | Markdown with speaker labels and timestamps |
| `name.txt` | Plain text with speaker labels |
| `name.json` | Machine-readable segment data |
| `name.srt` | SRT subtitles for video players |

When diarization is enabled, an additional set of files is written with a `.diarized` suffix (e.g. `name.diarized.md`).

## CLI reference

### `transcribe`

```
transcribe <audio> [options]

  audio                   Path to audio file or directory

  --model MODEL           Whisper model (default: large-v3-turbo)
  --language LANG         Language code (default: es)
  --diarize               Enable speaker diarization
  --speakers N            Exact number of speakers
  --min-speakers N        Min speakers for auto-detection (default: 2)
  --max-speakers N        Max speakers for auto-detection (default: 8)
  --output DIR            Output directory (default: ./transcripts)
  --batch-size N          Batch size for transcription (default: 8)
  --force                 Re-transcribe even if outputs exist
  --hf-token TOKEN        HuggingFace token
```

### `diarize`

```
diarize <audio> <transcript> [options]

  audio                   Original audio file
  transcript              Existing transcript JSON

  --speakers N            Exact number of speakers
  --min-speakers N        Min speakers (default: 2)
  --max-speakers N        Max speakers (default: 8)
  --hf-token TOKEN        HuggingFace token
```

## Supported audio formats

m4a, mp3, wav, mp4, flac, ogg, aac, wma, opus

## Project structure

```
src/transcribe/
  cli_transcribe.py   Main transcription CLI (transcribe → align → diarize)
  cli_diarize.py      Standalone diarization CLI for existing transcripts
  diarization.py      pyannote speaker diarization pipeline
  output.py           Writes transcripts in 4 formats (md, txt, json, srt)
  auth.py             HuggingFace token resolution
  formatting.py       Timestamp formatting utilities
```
