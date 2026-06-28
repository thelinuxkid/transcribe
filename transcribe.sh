#!/bin/bash
# Wrapper that suppresses the torchcodec UserWarning emitted by pyannote at import time.
# PYTHONWARNINGS is processed before any Python code runs, so it catches import-time warnings
# that warnings.filterwarnings() inside a script is too late to catch.
export PYTHONWARNINGS="ignore::UserWarning:pyannote.audio.core.io"
exec python "$(dirname "$0")/transcribe.py" "$@"
