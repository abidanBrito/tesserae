"""Stateless postprocessing raster functions."""

from __future__ import annotations

from pathlib import Path

import rasterio as rio
from rasterio.mask import mask

from ._common import NODATA, Compression, PathLike


def clip(
    raster_path: PathLike,
    roi_path: PathLike,
    *,
    nodata: float = NODATA,
    output_path: PathLike | None = None,
    crop_bounds: bool = True,
    compression: Compression | None = None,
) -> Path:
    """
    Clip a raster to a vector region of interest.

    Requires ``geopandas``.

    :param raster_path: input GeoTIFF file to clip.
    :param roi_path: vector file (Shapefile, GPKG, GeoJSON, etc.) defining the region of interest.
    :param nodata: NoData sentinel value for pixels outside the ROI.
    :param output_path: where to write the clipped rasters. Defaults to overwriting *raster_path*.
    :param crop_bounds: if ``True``, crop the output extent to the ROI bounds. If ``False``, the
        output retains the original extent with pixels outside the ROI set to *nodata*.
    :param compression: GeoTIFF compression codec. ``None`` disables compression.
    :return: the resolved output path.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "clip requires geopandas. Install with: pip install tesserae[clip]"
        ) from e

    raster_path = Path(raster_path)
    output_path = Path(output_path) if output_path else raster_path

    roi = gpd.read_file(roi_path)
    if roi.empty or "geometry" not in roi.columns:
        raise ValueError(f"ROI file contains no valid geometries: {roi_path}")

    with rio.open(raster_path) as src:
        if roi.crs != src.crs:
            roi = roi.to_crs(src.crs)

        clipped, clipped_transform = mask(
            src,
            roi.geometry.tolist(),
            crop=crop_bounds,
            all_touched=False,
            nodata=nodata,
        )
        clipped_profile = src.profile.copy()

    clipped_profile.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=clipped_transform,
        nodata=nodata,
    )

    if compression is not None:
        clipped_profile["compress"] = compression

    with rio.open(output_path, "w", **clipped_profile) as dst:
        dst.write(clipped)

    return output_path
