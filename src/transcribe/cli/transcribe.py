#!/usr/bin/env python3
"""
Transcribe audio files with whispermlx (Whisper + MLX on Apple Silicon).

Usage:
    transcribe meeting.m4a
    transcribe meeting.m4a --speakers 4
    transcribe meeting.m4a --model large-v3 --output ./transcripts
"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

from ..auth import DIARIZATION_AUTH_ERROR, HF_TOKEN_ERROR, get_hf_token
from ..config import load_config
from ..diarization import DIARIZE_DEVICE, run_diarization
from ..output import write_outputs

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".mp4", ".flac", ".ogg", ".aac", ".wma", ".opus"}


def parse_args():
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization"
    )
    parser.add_argument("audio", help="Path to audio file or directory to transcribe recursively")
    parser.add_argument(
        "--model",
        default=cfg["model"],
        help=f"Whisper model (default: {cfg['model']})",
    )
    parser.add_argument(
        "--language",
        default=cfg["language"],
        help=f"Language code (default: {cfg['language']})",
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
        default=cfg["min_speakers"],
        help=f"Minimum speakers for auto-detection (default: {cfg['min_speakers']})",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=cfg["max_speakers"],
        help=f"Maximum speakers for auto-detection (default: {cfg['max_speakers']})",
    )
    parser.add_argument(
        "--output",
        default=cfg["output"],
        help=f"Output directory (default: {cfg['output']})",
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
        default=cfg["batch_size"],
        help=f"Batch size for transcription (default: {cfg['batch_size']}, reduce if OOM)",
    )
    return parser.parse_args()


def _run_diarization_safe(audio_path, segments, args):
    hf_token = get_hf_token(args.hf_token)
    if not hf_token:
        print(HF_TOKEN_ERROR)
        return None

    try:
        diarized_segments, _ = run_diarization(
            audio_path, segments, hf_token,
            args.speakers, args.min_speakers, args.max_speakers,
        )
        return diarized_segments
    except Exception as e:
        print(f"\n   ERROR: Diarization failed: {type(e).__name__}: {e}")
        print(f"   {DIARIZATION_AUTH_ERROR}")
        traceback.print_exc()
        return None


def transcribe_file(audio_path: Path, args, output_root: Path):
    import whispermlx

    output_dir = output_root / audio_path.parent
    stem = audio_path.stem

    # Skip logic
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
            diarized_segments = _run_diarization_safe(audio_path, segments, args)
            if diarized_segments:
                print(f"\n── Writing diarized output to {output_dir}/")
                write_outputs(diarized_segments, output_dir, stem, diarized=True)
            return

    # Full transcription
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

    # Step 1: Transcribe (MLX → Metal)
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

    # Step 2: Word-level alignment
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

    # Write plain output
    print(f"\n── Writing output to {output_dir}/")
    write_outputs(result["segments"], output_dir, stem)

    # Step 3: Diarization (pyannote → MPS)
    if args.diarize:
        diarized_segments = _run_diarization_safe(audio_path, result["segments"], args)
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
