"""Dataset image validation using Pillow's canonical verify+reopen+load idiom.

Security:
  - PIL.Image.MAX_IMAGE_PIXELS capped at 50_000_000 (~50 MP) to defend
    against decompression bombs. Pillow's default 89 MP is too generous
    for the 8GB-VRAM target hardware.
  - DecompressionBombError is caught by verify() and surfaced as
    "corrupt: ...", never propagated to the caller.

Contract:
  validate_image(p) -> (ok, reason_if_bad, (w, h) if readable)

  Reason prefixes are stable strings the caller uses to choose an
  ErrorCode:
    "format"            -> unsupported extension
    "corrupt: <exc>"    -> PIL verify() failed (header invalid, bomb,
                           etc.)
    "truncated: <exc>"  -> load() failed (partial decode)
    "low_res: WxH"      -> below MIN_RES
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Hard cap — must be set before any Image.open() in this process.
Image.MAX_IMAGE_PIXELS = 50_000_000

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_RES = 512


def validate_image(p: Path) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Return (ok, reason_if_bad, (w, h) if readable).

    The reason prefix mapping is the stable contract consumed by
    auto_ft.commands.prepare for ErrorCode selection.
    """
    if p.suffix.lower() not in ALLOWED_EXTS:
        return (False, "format", None)
    # verify() detects most corruption (header-level) without fully decoding.
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as e:
        return (False, f"corrupt: {e}", None)
    # verify() consumes the file pointer; must reopen+load() to catch truncation.
    try:
        with Image.open(p) as img:
            img.load()
            w, h = img.size
    except Exception as e:
        return (False, f"truncated: {e}", None)
    if w < MIN_RES or h < MIN_RES:
        return (False, f"low_res: {w}x{h}", (w, h))
    return (True, None, (w, h))
