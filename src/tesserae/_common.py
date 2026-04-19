"""Shared types and constants."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PathLike = str | Path
Compression = Literal["ZSTD", "LZW", "DEFLATE", "LZMA", "PACKBITS", "JPEG", "WEBP"]

NODATA = -9999.0
