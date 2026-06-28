import json
from pathlib import Path

from .formatting import format_timestamp, to_srt_timestamp


def write_outputs(segments, output_dir: Path, stem: str, diarized: bool = False):
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
