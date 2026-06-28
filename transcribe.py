#!/usr/bin/env python3
"""
transcribe.py — Spanish multi-speaker meeting transcription
Uses whispermlx (WhisperX + MLX backend) on Apple Silicon

Usage:
    python transcribe.py meeting.m4a
    python transcribe.py meeting.m4a --speakers 4
    python transcribe.py meeting.m4a --model large-v3 --output ./transcripts

Requirements:
    pip install whispermlx soundfile
    huggingface-cli login  (for diarization)
    Accept pyannote model terms at:
      https://huggingface.co/pyannote/speaker-diarization-community-1
      https://huggingface.co/pyannote/segmentation-3.0
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

# MLX transcription always uses "cpu" string — whispermlx routes to Metal internally.
# Diarization (pyannote) uses MPS when available on Apple Silicon.
DIARIZE_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".mp4", ".flac", ".ogg", ".aac", ".wma", ".opus"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe Spanish meeting audio with speaker diarization"
    )
    parser.add_argument("audio", help="Path to audio file or directory to transcribe recursively")
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model (default: large-v3-turbo). Options: large-v3, large-v3-turbo, medium",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Language code (default: es for Spanish)",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Exact number of speakers. Omit to auto-detect (less accurate).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=2,
        help="Minimum speakers for auto-detection (default: 2)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Maximum speakers for auto-detection (default: 8)",
    )
    parser.add_argument(
        "--output",
        default="./transcripts",
        help="Output directory (default: ./transcripts)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token. Falls back to HF_TOKEN env var or cached login.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (adds speaker labels, slower)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if output files already exist",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for transcription (default: 8, reduce if OOM)",
    )
    return parser.parse_args()


def get_hf_token(args_token):
    """Resolve HuggingFace token from args, env, or cached login."""
    token = args_token or os.environ.get("HF_TOKEN")
    if token:
        return token
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.exists():
        return cache.read_text().strip()
    return None


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_srt_timestamp(s: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h, rem = divmod(s, 3600)
    m, rem = divmod(rem, 60)
    sec, ms = divmod(rem, 1)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int(ms * 1000):03d}"


def write_outputs(segments, output_dir: Path, stem: str, diarized: bool = False):
    """Write transcript in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".diarized" if diarized else ""

    # Markdown
    md_path = output_dir / f"{stem}{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {stem}\n\n")
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg["text"].strip()
            start = format_timestamp(seg["start"])
            if speaker != current_speaker:
                f.write(f"\n**{speaker}** `{start}`\n\n")
                current_speaker = speaker
            f.write(f"{text} ")
        f.write("\n")
    print(f"  → {md_path}")

    # Plain text
    txt_path = output_dir / f"{stem}{suffix}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg["text"].strip()
            start = format_timestamp(seg["start"])
            if speaker != current_speaker:
                f.write(f"\n[{speaker} - {start}]\n")
                current_speaker = speaker
            f.write(f"{text} ")
        f.write("\n")
    print(f"  → {txt_path}")

    # JSON
    json_out = output_dir / f"{stem}{suffix}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"  → {json_out}")

    # SRT
    srt_path = output_dir / f"{stem}{suffix}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = seg["start"]
            end = seg["end"]
            speaker = seg.get("speaker", "")
            text = seg["text"].strip()
            label = f"[{speaker}] " if speaker else ""
            f.write(f"{i}\n{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}\n{label}{text}\n\n")
    print(f"  → {srt_path}")


def run_diarization(audio_path: Path, segments, args):
    """Run diarization on segments and return diarized result, or None on failure."""
    import whispermlx

    print(f"\n── Diarizing speakers (pyannote on {DIARIZE_DEVICE})...")
    t0 = time.time()

    hf_token = get_hf_token(args.hf_token)
    if not hf_token:
        print("""
   ERROR: No HuggingFace token found.

   Options:
     1. Run: huggingface-cli login
     2. Set env var: export HF_TOKEN=hf_...
     3. Pass flag: --hf-token hf_...

   Also accept pyannote model terms at:
     https://huggingface.co/pyannote/speaker-diarization-community-1
     https://huggingface.co/pyannote/segmentation-3.0
""")
        return None

    try:
        from whispermlx.diarize import DiarizationPipeline

        diarize_kwargs = {}
        if args.speakers:
            diarize_kwargs["num_speakers"] = args.speakers
        else:
            diarize_kwargs["min_speakers"] = args.min_speakers
            diarize_kwargs["max_speakers"] = args.max_speakers

        diarize_model = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-community-1",
            token=hf_token,
            device=DIARIZE_DEVICE,
        )
        diarize_segments = diarize_model(str(audio_path), **diarize_kwargs)
        result = whispermlx.assign_word_speakers(diarize_segments, {"segments": segments})

        print(f"   Done in {time.time() - t0:.1f}s")

        speakers_found = set(
            seg.get("speaker", "") for seg in result["segments"]
        )
        speakers_found.discard("")
        print(f"   Speakers detected: {len(speakers_found)} → {', '.join(sorted(speakers_found))}")

        return result["segments"]

    except Exception as e:
        import traceback
        print(f"\n   ERROR: Diarization failed: {type(e).__name__}: {e}")
        print("   Accept model terms at:")
        print("     https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("     https://huggingface.co/pyannote/segmentation-3.0")
        print("   Then re-run: huggingface-cli login")
        traceback.print_exc()
        return None


def transcribe_file(audio_path: Path, args, output_root: Path):
    """Transcribe a single audio file through the 3-step pipeline."""
    import whispermlx

    output_dir = output_root / audio_path.parent
    stem = audio_path.stem

    # ── Skip logic ──────────────────────────────────────────────────────────
    json_path = output_dir / f"{stem}.json"
    diarized_json_path = output_dir / f"{stem}.diarized.json"

    if not args.force:
        json_exists = json_path.exists()
        diarized_exists = diarized_json_path.exists()

        if json_exists and (not args.diarize or diarized_exists):
            print(f"  Skipping (already transcribed): {audio_path}")
            return

        if json_exists and args.diarize and not diarized_exists:
            print(f"\n  Diarize-only: {audio_path} (transcription exists, adding speaker labels)")
            with open(json_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
            diarized_segments = run_diarization(audio_path, segments, args)
            if diarized_segments:
                print(f"\n── Writing diarized output to {output_dir}/")
                write_outputs(diarized_segments, output_dir, stem, diarized=True)
            return

    # ── Full transcription ──────────────────────────────────────────────────
    total_start = time.time()

    print(f"\nFile    : {audio_path}")
    print(f"Model   : {args.model}")
    print(f"Language: {args.language}")
    print(f"Device  : transcription=MLX(Metal), diarization={DIARIZE_DEVICE}")
    if args.speakers:
        print(f"Speakers: {args.speakers} (exact)")
    else:
        print(f"Speakers: auto-detect ({args.min_speakers}–{args.max_speakers})")
    print()

    # ── Step 1: Transcribe (MLX → Metal) ────────────────────────────────────
    print("── Step 1/3: Transcribing (MLX)...")
    t0 = time.time()

    model = whispermlx.load_model(
        args.model,
        device="cpu",
        language=args.language,
        compute_type="float16",
    )
    result = model.transcribe(
        str(audio_path),
        batch_size=args.batch_size,
        language=args.language,
    )

    t1 = time.time()
    print(f"   Done in {t1 - t0:.1f}s  ({len(result['segments'])} segments)")

    del model
    gc.collect()

    # ── Step 2: Word-level alignment ─────────────────────────────────────────
    print(f"\n── Step 2/3: Aligning word timestamps (language: {args.language})...")
    t0 = time.time()

    try:
        model_a, metadata = whispermlx.load_align_model(
            language_code=args.language,
            device="cpu",
        )
        result = whispermlx.align(
            result["segments"],
            model_a,
            metadata,
            str(audio_path),
            device="cpu",
            return_char_alignments=False,
        )
        del model_a
        gc.collect()
        print(f"   Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"   WARNING: Alignment failed ({e}). Continuing without word timestamps.")

    # ── Write plain output ───────────────────────────────────────────────────
    print(f"\n── Writing output to {output_dir}/")
    write_outputs(result["segments"], output_dir, stem)

    # ── Step 3: Diarization (pyannote → MPS) ────────────────────────────────
    if args.diarize:
        diarized_segments = run_diarization(audio_path, result["segments"], args)
        if diarized_segments:
            print(f"\n── Writing diarized output to {output_dir}/")
            write_outputs(diarized_segments, output_dir, stem, diarized=True)
    else:
        print("\n── Skipping diarization (use --diarize to enable)")

    total = time.time() - total_start
    print(f"\n✓ Finished in {total:.1f}s ({total/60:.1f} min)")
    print()


def main():
    args = parse_args()
    input_path = Path(args.audio)

    if not input_path.exists():
        print(f"ERROR: Path not found: {input_path}")
        sys.exit(1)

    try:
        import whispermlx  # noqa: F401
    except ImportError:
        print("ERROR: whispermlx not installed. Run: pip install whispermlx")
        sys.exit(1)

    output_root = Path(args.output)

    if input_path.is_file():
        transcribe_file(input_path, args, output_root)
    elif input_path.is_dir():
        audio_files = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not audio_files:
            print(f"ERROR: No audio files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(audio_files)} audio file(s) in {input_path}\n")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n{'='*60}")
            print(f"  [{i}/{len(audio_files)}] {audio_file}")
            print(f"{'='*60}")
            transcribe_file(audio_file, args, output_root)
    else:
        print(f"ERROR: {input_path} is not a file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
