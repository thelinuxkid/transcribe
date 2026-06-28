#!/usr/bin/env python3
"""
Add speaker labels to an existing transcript JSON.

Usage:
    diarize Mundiastur.m4a transcripts/Mundiastur.json --speakers 4
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from .auth import DIARIZATION_AUTH_ERROR, HF_TOKEN_ERROR, get_hf_token
from .config import load_config
from .diarization import DIARIZE_DEVICE, run_diarization
from .output import write_outputs


def parse_args():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Add speaker diarization to existing transcript")
    parser.add_argument("audio", help="Original audio file (.m4a, .mp3, .wav)")
    parser.add_argument("transcript", help="Existing transcript JSON from whispermlx")
    parser.add_argument("--speakers", type=int, default=None, help="Exact number of speakers (improves accuracy)")
    parser.add_argument("--min-speakers", type=int, default=cfg["min_speakers"],
                        help=f"Min speakers for auto-detection (default: {cfg['min_speakers']})")
    parser.add_argument("--max-speakers", type=int, default=cfg["max_speakers"],
                        help=f"Max speakers for auto-detection (default: {cfg['max_speakers']})")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (falls back to cached login)")
    return parser.parse_args()


def main():
    import json
    import traceback

    args = parse_args()

    audio_path = Path(args.audio)
    json_path = Path(args.transcript)

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)
    if not json_path.exists():
        print(f"ERROR: Transcript JSON not found: {json_path}")
        sys.exit(1)

    hf_token = get_hf_token(args.hf_token)
    if not hf_token:
        print(HF_TOKEN_ERROR)
        sys.exit(1)

    print(f"\nAudio     : {audio_path}")
    print(f"Transcript: {json_path}")
    print(f"Device    : {DIARIZE_DEVICE}")
    if args.speakers:
        print(f"Speakers  : {args.speakers} (exact)")
    else:
        print(f"Speakers  : auto-detect ({args.min_speakers}–{args.max_speakers})")
    print()

    # Load existing transcript
    print("── Loading transcript JSON...")
    with open(json_path, encoding="utf-8") as f:
        transcript = json.load(f)
    if isinstance(transcript, list):
        segments = transcript
    else:
        segments = transcript["segments"]
    print(f"   {len(segments)} segments loaded")

    t0 = time.time()

    try:
        diarized_segments, speakers_found = run_diarization(
            audio_path, segments, hf_token,
            args.speakers, args.min_speakers, args.max_speakers,
        )
    except Exception as e:
        print(f"\nERROR during diarization: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\n{DIARIZATION_AUTH_ERROR}")
        sys.exit(1)

    # Speaker distribution
    dist = Counter(seg.get("speaker", "UNKNOWN") for seg in diarized_segments)
    print("   Speaker segment counts:")
    for speaker, count in sorted(dist.items()):
        print(f"     {speaker}: {count} segments")

    output_dir = json_path.parent
    stem = json_path.stem
    print(f"\n── Writing output to {output_dir}/")
    write_outputs(diarized_segments, output_dir, stem, diarized=True)

    print(f"\n✓ Done in {(time.time() - t0)/60:.1f} min total")
    print()


if __name__ == "__main__":
    main()
