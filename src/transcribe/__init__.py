import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

__version__ = "0.1.0"
