import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path.home() / ".transcriberc"

DEFAULTS = {
    "language": "en",
    "model": "large-v3-turbo",
    "output": "./transcripts",
    "min_speakers": 2,
    "max_speakers": 8,
    "batch_size": 8,
}


def load_config():
    config = dict(DEFAULTS)
    if not CONFIG_PATH.exists():
        return config
    try:
        with open(CONFIG_PATH, "rb") as f:
            user = tomllib.load(f)
        config.update(user)
    except Exception as e:
        print(f"WARNING: Failed to parse {CONFIG_PATH}: {e}", file=sys.stderr)
    return config
