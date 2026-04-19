"""Tesserae - Stitch geospatial rasters into seamless mosaics."""

from importlib.metadata import version

from tesserae.io import write_raster
from tesserae.processing import stitch

__version__ = version("tesserae")

__all__ = ["stitch", "write_raster"]
