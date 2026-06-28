def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_srt_timestamp(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, rem = divmod(rem, 60)
    sec, ms = divmod(rem, 1)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int(ms * 1000):03d}"
