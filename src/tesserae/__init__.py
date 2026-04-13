"""Tesserae - Stitch geospatial rasters into seamless mosaics."""

from importlib.metadata import version

from tesserae.io import write_raster

__version__ = version("tesserae")

__all__ = [
    "write_raster",
]
