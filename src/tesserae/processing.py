"""Stateless raster processing functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge

from .io import _extract_epsg, write_raster

BlendFn = Callable[..., np.ndarray] | None


def stitch(
    paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    pixel_size: float,
    nodata: float,
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
    :param nodata: NoData sentinel value.
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
        for p in paths:
            with rio.open(str(p)) as src:
                data = src.read()
                transform = src.transform
                tile_nodata = nodata if nodata is not None else src.nodata

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
                    nodata=tile_nodata,
                )

                memfile = rio.MemoryFile()
                with memfile.open(**profile) as dst:
                    dst.write(data)

                memfiles.append(memfile)
                sources.append(memfile.open())

        method: Any = blend_fn if blend_fn is not None else "first"
        result = merge(
            sources,
            res=pixel_size,
            nodata=nodata,
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

    if np.all(mosaic == nodata):
        raise ValueError("Resulting mosaic contains no valid data.")

    write_raster(
        data=mosaic,
        transform=geotransform,
        epsg=_extract_epsg(paths[0]),
        path=output_path,
        nodata=nodata,
        compression=compression,
    )

    return mosaic, geotransform
