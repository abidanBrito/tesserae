"""Tesserae - Stitch geospatial rasters into seamless mosaics."""

from importlib.metadata import version

from tesserae.io import write_raster
from tesserae.processing import feather_blend, make_feather_blend_fn, stitch

__version__ = version("tesserae")

__all__ = [
    "feather_blend",
    "make_feather_blend_fn",
    "stitch",
    "write_raster",
]
