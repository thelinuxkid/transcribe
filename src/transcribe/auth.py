import os
from pathlib import Path

HF_TOKEN_ERROR = """
ERROR: No HuggingFace token found.

Options:
  1. Run: huggingface-cli login
  2. Set env var: export HF_TOKEN=hf_...
  3. Pass flag: --hf-token hf_...

Also accept pyannote model terms at:
  https://huggingface.co/pyannote/speaker-diarization-community-1
  https://huggingface.co/pyannote/segmentation-3.0
"""

DIARIZATION_AUTH_ERROR = """\
If you see a token/auth error:
  1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-community-1
  2. Accept terms at https://huggingface.co/pyannote/segmentation-3.0
  3. Run: huggingface-cli login"""


def get_hf_token(args_token):
    token = args_token or os.environ.get("HF_TOKEN")
    if token:
        return token
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.exists():
        return cache.read_text().strip()
    return None
