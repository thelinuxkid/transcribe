import time

import torch

DIARIZE_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


def build_diarize_kwargs(speakers=None, min_speakers=2, max_speakers=8):
    if speakers:
        return {"num_speakers": speakers}
    return {"min_speakers": min_speakers, "max_speakers": max_speakers}


def run_diarization(audio_path, segments, hf_token, speakers=None, min_speakers=2, max_speakers=8):
    import whispermlx
    from whispermlx.diarize import DiarizationPipeline

    print(f"\n── Diarizing speakers (pyannote on {DIARIZE_DEVICE})...")
    t0 = time.time()

    kwargs = build_diarize_kwargs(speakers, min_speakers, max_speakers)

    pipeline = DiarizationPipeline(
        model_name=PYANNOTE_MODEL,
        token=hf_token,
        device=DIARIZE_DEVICE,
    )
    diarize_segments = pipeline(str(audio_path), **kwargs)
    result = whispermlx.assign_word_speakers(diarize_segments, {"segments": segments})

    print(f"   Done in {time.time() - t0:.1f}s")

    speakers_found = set(seg.get("speaker", "") for seg in result["segments"])
    speakers_found.discard("")
    speakers_found = sorted(speakers_found)
    print(f"   Speakers detected: {len(speakers_found)} → {', '.join(speakers_found)}")

    return result["segments"], speakers_found
