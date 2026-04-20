"""Stateless postprocessing raster functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform
from rasterio.warp import reproject as rio_reproject

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


def reproject(
    raster_path: PathLike,
    epsg: int,
    *,
    nodata: float,
    output_path: PathLike | None = None,
    resampling: Resampling = Resampling.bilinear,
    compression: Compression | None = None,
) -> Path:
    """
    Reproject a raster to *target_epsg*, skipping if already matching.

    :param raster_path: input GeoTIFF file to reproject.
    :param epsg: target EPSG code for the output CRS.
    :param nodata: NoData sentinel value for both source and destination.
    :param output_path: where to write the reprojected raster. Defaults to overwriting
        *raster_path*.
    :param resampling: resampling method applied during warping.
    :compression: GeoTIFF compression codec. ``None`` disables compression.
    :return: the resolved output path (possibly unchanged if the source CRS already matches).
    """
    raster_path = Path(raster_path)
    output_path = Path(output_path) if output_path else raster_path

    with rio.open(raster_path) as src:
        if src.crs.to_epsg() == epsg:
            return output_path

        dst_crs = CRS.from_epsg(epsg)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        assert width is not None and height is not None

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=nodata,
        )

        if compression is not None:
            profile["compress"] = compression

        dest = np.empty((src.count, height, width), dtype=src.dtypes[0])

        rio_reproject(
            source=src.read(),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=nodata,
            resampling=resampling,
        )

    # NOTE(abi): we write in a separate block so that the source handle is
    #            closed, just in case we overwrite the input GeoTIFF file.
    with rio.open(output_path, "w", **profile) as dst:
        dst.write(dest)

    return output_path


def fill_nodata(
    input_path: PathLike,
    output_path: PathLike | None = None,
    *,
    max_search_distance: int = 100,
    smoothing_iterations: int = 0,
    compression: Compression | None = None,
) -> Path:
    """
    Fill NoData holes via GDAL's built-in interpolation.

    Uses ``rasterio.fill.fillnodata`` (backed by GDAL's ``GDALFillNodata``),
    which performs an iterative morphological expansion of valid pixels into
    nodata regions.

    :param input_path: input GeoTIFF file to process.
    :param output_path: destination GeoTIFF file. Defaults to overwriting *input_path*.
    :param max_search_distance: maximum number of pixels to search in all directions
        for valid values to interpolate from.
    :param smoothing_iterations: number of 3x3 smoothing passes to run on the filled
        pixels after interpolation.
    :param compression: GeoTIFF compression codec. ``None`` disables compression.
    :return: the resolved output path of the filled raster.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path

    output_path = Path(output_path)

    with rio.open(input_path) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        filled_bands = []

        for i in range(1, src.count + 1):
            band = src.read(i)
            band = band.astype("float32")

            if nodata is not None:
                mask = (band != nodata).astype("uint8")
            else:
                mask = (~np.isnan(band)).astype("uint8")

            filled = fillnodata(
                band,
                mask=mask,
                max_search_distance=max_search_distance,
                smoothing_iterations=smoothing_iterations,
            )
            filled_bands.append(filled)

    if compression is not None:
        profile["compress"] = compression

    with rio.open(output_path, "w", **profile) as dst:
        for i, band in enumerate(filled_bands, start=1):
            dst.write(band, i)

    return output_path
