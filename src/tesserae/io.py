"""Raster I/O utilities for reading metadata and writing GeoTIFF files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling

PathLike = str | Path
_NODATA = -9999.0


def write_raster(
    data: np.ndarray,
    transform: rio.Affine,
    epsg: int,
    path: PathLike,
    *,
    nodata: float = _NODATA,
    compression: str | None = None,
    dtype: np.typing.DTypeLike | None = None,
    cog: bool = False,
    tiled: bool = False,
    block_size: int = 512,
    bigtiff: str = "IF_SAFER",
    band_names: Sequence[str] | None = None,
    overview_levels: Sequence[int] | None = None,
    overview_resampling: Resampling = Resampling.average,
    nan_to_nodata: bool = False,
) -> Path:
    """
    Write a NumPy array to a GeoTIFF (or Cloud Optimized GeoTIFF).

    :param data: raster pixel data. Accepted shapes:
        - ``(H, W)`` — single band (promoted to ``(1, H, W)`` internally).
        - ``(B, H, W)`` — multi-band, bands-first (rasterio convention).
    :param transform: affine geotransform mapping pixel coordinates to CRS coordinates.
    :param epsg: EPSG code for the coordinate reference system.
    :param path: output file path. Parent directories are created automatically.
    :param nodata: sentinel value for missing or invalid pixels.
    :param compression: GeoTIFF compression codec (e.g. ``"ZSTD"``, ``"LZW"``, ``"DEFLATE"``).
        ``None`` disables compression.
    :param dtype: output data type. Defaults to *data.dtype*.
    :param cog: if ``True``, write a Cloud Optimized GeoTIFF using GDAL's ``COG`` driver.
        The COG driver handles tiling, overviews, and internal layout automatically. Mutually
        exclusive with *tiled* and manual *overview_levels* (the COG driver builds overviews
        itself when requested via ``overviews="AUTO"``).
    :param tiled: enable internal tiling for standard GTiff output. Ignored when *cog* is ``True``.
    :param block_size: tile dimensions in pixels. Used as ``blockxsize``/``blockysize`` for GTiff,
        or ``blocksize`` for COG.
    :param bigtiff: bigTIFF creation option (``"IF_SAFER"``, ``"YES"``, ``"NO"``). Only applies to
        standard GTiff output.
    :param band_names: optional sequence of human-readable band descriptions, written as band
        metadata. Only supported for standard GTiff output (the COG driver does not allow post-write
        metadata updates).
    :param overview_levels: pyramid levels to build (e.g. ``[2, 4, 8, 16]``). For standard
        GTiff, overviews are built after writing. For COG, passing any value here enables
        ``overviews="AUTO"`` and the driver builds them internally.
    :param overview_resampling: resampling method for overview generation.
    :param nan_to_nodata: if ``True``, replace ``NaN`` values with *nodata* before writing.
    :return: the resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    if nan_to_nodata:
        data = np.where(np.isnan(data), nodata, data)

    out_dtype = dtype if dtype is not None else data.dtype

    if cog:
        _write_cog(
            data=data,
            transform=transform,
            epsg=epsg,
            path=path,
            nodata=nodata,
            compression=compression,
            dtype=out_dtype,
            block_size=block_size,
            overview_levels=overview_levels,
            overview_resampling=overview_resampling,
        )
    else:
        _write_geotiff(
            data=data,
            transform=transform,
            epsg=epsg,
            path=path,
            nodata=nodata,
            compression=compression,
            dtype=out_dtype,
            tiled=tiled,
            block_size=block_size,
            bigtiff=bigtiff,
            band_names=band_names,
            overview_levels=overview_levels,
            overview_resampling=overview_resampling,
        )

    return path


def _write_geotiff(
    data: np.ndarray,
    transform: rio.Affine,
    epsg: int,
    path: Path,
    *,
    nodata: float,
    compression: str | None,
    dtype: np.typing.DTypeLike,
    tiled: bool,
    block_size: int,
    bigtiff: str,
    band_names: Sequence[str] | None,
    overview_levels: Sequence[int] | None,
    overview_resampling: Resampling,
) -> None:
    """Write a standard GeoTIFF with optional tiling and overviews."""
    band_count, height, width = data.shape

    profile: dict = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": band_count,
        "crs": CRS.from_epsg(epsg),
        "dtype": dtype,
        "transform": transform,
        "nodata": nodata,
        "BIGTIFF": bigtiff,
    }

    if compression:
        profile["compress"] = compression

        if np.issubdtype(dtype, np.floating):
            profile["predictor"] = 2

        if compression.upper() == "ZSTD":
            profile["zstd_level"] = 3

    if tiled:
        profile["tiled"] = True
        profile["blockxsize"] = block_size
        profile["blockysize"] = block_size

    with rio.open(path, "w", **profile) as dst:
        for i in range(band_count):
            dst.write(data[i].astype(dtype), i + 1)

            if band_names and i < len(band_names):
                dst.set_band_description(i + 1, band_names[i])

        if overview_levels:
            dst.build_overviews(list(overview_levels), overview_resampling)
            dst.update_tags(ns="rio_overview", resampling=overview_resampling.name)


def _write_cog(
    data: np.ndarray,
    transform: rio.Affine,
    epsg: int,
    path: Path,
    *,
    nodata: float,
    compression: str | None,
    dtype: np.typing.DTypeLike,
    block_size: int,
    overview_levels: Sequence[int] | None,
    overview_resampling: Resampling,
) -> None:
    """Write a Cloud Optimized GeoTIFF using GDAL's COG driver."""
    band_count, height, width = data.shape

    profile: dict = {
        "driver": "COG",
        "height": height,
        "width": width,
        "count": band_count,
        "crs": CRS.from_epsg(epsg),
        "dtype": dtype,
        "transform": transform,
        "nodata": nodata,
        "blocksize": block_size,
    }

    if compression:
        profile["compress"] = compression

        if np.issubdtype(dtype, np.floating):
            profile["predictor"] = 2

        if compression.upper() == "ZSTD":
            profile["zstd_level"] = 3

    if overview_levels:
        profile["overviews"] = "AUTO"
        profile["overview_resampling"] = overview_resampling.name

    with rio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def _extract_epsg(path: PathLike) -> int:
    """
    Read the EPSG code from a raster file.

    :param path: path to a georeferenced raster file.
    :return: the integer EPSG code.
    :raises ValueError: if the file has no CRS or the CRS cannot be mapped to an EPSG code.
    """
    with rio.open(path) as src:
        epsg = src.crs.to_epsg()
        if epsg is None:
            raise ValueError(f"Cannot determine EPSG for {path}")

        return epsg
