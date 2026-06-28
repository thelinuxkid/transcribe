#!/usr/bin/env python3
"""
diarize.py — Add speaker labels to an existing transcript JSON
Run this if whispermlx produced a transcript without speaker labels.

Uses whispermlx's DiarizationPipeline directly (same as transcribe.py),
which handles audio loading and device placement internally.

Usage:
    bash diarize.sh Mundiastur.m4a transcripts/Mundiastur.json --speakers 4

Outputs updated files alongside the originals:
    transcripts/Mundiastur.diarized.md
    transcripts/Mundiastur.diarized.txt
    transcripts/Mundiastur.diarized.json
    transcripts/Mundiastur.diarized.srt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

DIARIZE_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Add speaker diarization to existing transcript")
    parser.add_argument("audio", help="Original audio file (.m4a, .mp3, .wav)")
    parser.add_argument("transcript", help="Existing transcript JSON from whispermlx")
    parser.add_argument("--speakers", type=int, default=None, help="Exact number of speakers (improves accuracy)")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (falls back to cached login)")
    return parser.parse_args()


def get_hf_token(args_token):
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


def write_outputs(segments, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_dir / f"{stem}.diarized.md"
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
    txt_path = output_dir / f"{stem}.diarized.txt"
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
    json_path = output_dir / f"{stem}.diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"  → {json_path}")

    # SRT
    srt_path = output_dir / f"{stem}.diarized.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start, end = seg["start"], seg["end"]
            speaker = seg.get("speaker", "")
            text = seg["text"].strip()
            label = f"[{speaker}] " if speaker else ""
            f.write(f"{i}\n{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}\n{label}{text}\n\n")
    print(f"  → {srt_path}")


def main():
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
    # whispermlx's assign_word_speakers expects {"segments": [...]}
    # If the JSON is a bare list (our format), wrap it
    if isinstance(transcript, list):
        result = {"segments": transcript}
    else:
        result = transcript
    print(f"   {len(result['segments'])} segments loaded")

    # Run diarization using whispermlx's DiarizationPipeline
    # (same pipeline as transcribe.py — handles audio loading and .to(device) internally)
    print(f"\n── Running speaker diarization (pyannote on {DIARIZE_DEVICE})...")
    print("   This takes ~10-20 min for a 2-hour file on CPU, less on MPS.")
    t0 = time.time()

    try:
        from whispermlx.diarize import DiarizationPipeline
        import whispermlx

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

        print("   Diarizing (this is the slow part)...")
        diarize_df = diarize_model(str(audio_path), **diarize_kwargs)

        elapsed = time.time() - t0
        print(f"   Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        speakers_found = sorted(diarize_df["speaker"].unique())
        print(f"   Speakers detected: {len(speakers_found)} → {', '.join(speakers_found)}")

    except Exception as e:
        import traceback
        print(f"\nERROR during diarization: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nIf you see a token/auth error:")
        print("  1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("  2. Accept terms at https://huggingface.co/pyannote/segmentation-3.0")
        print("  3. Run: huggingface-cli login")
        sys.exit(1)

    # Merge speaker labels using whispermlx's own assign_word_speakers
    print("\n── Merging speaker labels into transcript...")
    result = whispermlx.assign_word_speakers(diarize_df, result)

    from collections import Counter
    dist = Counter(seg.get("speaker", "UNKNOWN") for seg in result["segments"])
    print("   Speaker segment counts:")
    for speaker, count in sorted(dist.items()):
        print(f"     {speaker}: {count} segments")

    output_dir = json_path.parent
    stem = json_path.stem
    print(f"\n── Writing output to {output_dir}/")
    write_outputs(result["segments"], output_dir, stem)

    print(f"\n✓ Done in {(time.time() - t0)/60:.1f} min total")
    print()


if __name__ == "__main__":
    main()
