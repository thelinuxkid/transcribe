#!/bin/bash
export PYTHONWARNINGS="ignore::UserWarning:pyannote.audio.core.io"
exec python "$(dirname "$0")/diarize.py" "$@"
