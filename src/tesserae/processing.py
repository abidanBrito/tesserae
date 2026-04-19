"""Stateless raster processing functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge

from .io import _NODATA, _extract_epsg, write_raster

BlendFn = Callable[..., np.ndarray] | None


def stitch(
    paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    pixel_size: float,
    nodata: float | None = None,
    resampling: Resampling = Resampling.nearest,
    blend_fn: BlendFn = None,
    pad_width: int = 0,
    compression: str | None = None,
) -> tuple[np.ndarray, rio.Affine]:
    """
    Stitch multiple rasters into a single mosaic.

    Note that all inputs must share the same CRS.

    :param paths: input raster file paths.
    :param output_path: where to write the output GeoTIFF.
    :param pixel_size: target resolution in CRS units.
    :param nodata: optional NoData sentinel value. If ``None`` it gets the first tiles'
        NoData value.
    :param resampling: resampling method applied during warping.
    :param blend_fn: optional blending callback following the ``rasterio.merge``
        *method* signature. ``None`` falls back to ``"first"``.
    :param pad_width: pixels of reflective padding added to each tile before merging.
        Useful when *blend_fn* needs overlap context.
    :param compression: GeoTIFF compression codec for the output.
    :return: ``(mosaic_data, geotransform)`` tuple. *mosaic_data* has shape ``(B, H, W)``.
    """
    output_path = Path(output_path)

    sources: list[rio.DatasetReader] = []
    memfiles: list[rio.MemoryFile] = []

    try:
        for path in paths:
            with rio.open(str(path)) as src:
                data = src.read()
                transform = src.transform

                if pad_width > 0:
                    padded_transform = transform
                    padded_bands = []

                    for band in data:
                        padded_band, padded_transform = rio.pad(
                            band, transform, pad_width=pad_width, mode="reflect"
                        )
                        padded_bands.append(padded_band)

                    data = np.array(padded_bands)
                    transform = padded_transform

                profile = src.profile.copy()
                profile.update(
                    height=data.shape[1],
                    width=data.shape[2],
                    transform=transform,
                )

                memfile = rio.MemoryFile()
                with memfile.open(**profile) as dst:
                    dst.write(data)

                memfiles.append(memfile)
                sources.append(memfile.open())

        output_nodata = nodata if nodata is not None else _NODATA
        method: Any = blend_fn if blend_fn is not None else "first"

        result = merge(
            sources,
            res=pixel_size,
            nodata=output_nodata,
            method=method,
            resampling=resampling,
            target_aligned_pixels=True,
        )
        assert result is not None
        mosaic: np.ndarray = result[0]
        geotransform: rio.Affine = result[1]  # pyrefly: ignore
    finally:
        for s in sources:
            s.close()

        for mf in memfiles:
            mf.close()

    if np.all(mosaic == output_nodata):
        raise ValueError("Resulting mosaic contains no valid data.")

    write_raster(
        data=mosaic,
        transform=geotransform,
        epsg=_extract_epsg(paths[0]),
        path=output_path,
        nodata=output_nodata,
        compression=compression,
    )

    return mosaic, geotransform


def feather_blend(
    merged_data: np.ndarray,
    new_data: np.ndarray,
    merged_mask: np.ndarray,
    new_mask: np.ndarray,
    *,
    nodata: float,
    feather_width: int = 5,
    **_kwargs,
) -> None:
    """
    Distance-weighted feather blending for :func:`stitch`.

    Follows the ``rasterio.merge`` *method* callback signature so it can
    be passed directly as a *blend_fn*.

    Requires ``scipy``.

    :param merged_data/new_data: pixel arrays. Note that *merged_data* is modified in-place.
    :param merged_mask/new_mask: boolean masks (``True`` = nodata).
    :param nodata: sentinel value for missing pixels.
    :param feather_width: controls the blending ramp width (pixels).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as e:
        raise ImportError(
            "feather_blend requires scipy. Install with: pip install tesserae[blend]"
        ) from e

    merged_valid = ~merged_mask
    new_valid = ~new_mask
    overlap = merged_valid & new_valid
    new_only = ~merged_valid & new_valid

    # Take new non-overlapping data as-is
    np.copyto(merged_data, new_data, where=new_only)

    if not np.any(overlap):
        merged_data[~merged_valid & ~new_valid] = nodata
        return

    # Detect overlap-region boundary pixels (4-connected)
    boundaries = np.zeros_like(overlap, dtype=bool)
    boundaries[1:, :] |= ~overlap[1:, :] & overlap[:-1, :]
    boundaries[:-1, :] |= ~overlap[:-1, :] & overlap[1:, :]
    boundaries[:, 1:] |= ~overlap[:, 1:] & overlap[:, :-1]
    boundaries[:, :-1] |= ~overlap[:, :-1] & overlap[:, 1:]

    # Distance from boundary -> Blend weights
    dist: np.ndarray = np.asarray(distance_transform_edt(overlap & ~boundaries))
    weights = np.clip(dist / max(float(feather_width), 1.0), 0.0, 1.0)

    # Weighted average in the overlap zone
    m = np.where(merged_data == nodata, np.nan, merged_data)
    n = np.where(new_data == nodata, np.nan, new_data)
    blended = m * weights + n * (1.0 - weights)

    merged_data[overlap] = np.where(np.isnan(blended[overlap]), nodata, blended[overlap])
    merged_data[~merged_valid & ~new_valid] = nodata


def make_feather_blend_fn(nodata: float, feather_width: int = 5) -> Callable:
    """
    Return a closure suitable for :func:`stitch`'s *blend_fn* parameter.

    :param nodata: NoData sentinel value baked into the returned callback.
    :param feather_width: blending ramp width (pixels) baked into the returned callback.
    :return: a callable matching the ``rasterio.merge`` *method* callback signature.
    """

    def _blend(merged_data, new_data, merged_mask, new_mask, **kwargs):
        return feather_blend(
            merged_data,
            new_data,
            merged_mask,
            new_mask,
            nodata=nodata,
            feather_width=feather_width,
            **kwargs,
        )

    return _blend
