"""Screen the world's mountain ranges for analogues of the European Alps.

Each GMBA v2 major mountain range is described on four criteria (climate,
elevation, forest type, fire regime) and ranked by Gower distance to the
European Alps. See README.md for data sources and the rationale.

Usage:
    python screen_ranges.py [--data DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio import features, windows
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

ALPS_ID = 10001  # GMBA_V2_ID of "European Alps"
MIN_AREA_KM2 = 10_000
FIRE_YEARS = range(2003, 2024)  # full years in the Global Fire Atlas extension

# Köppen-Geiger classes (Beck et al. 2023 legend), grouped by climate regime. Method v2 keeps
# the Köppen distinctions that matter for fuel and fire season: precipitation seasonality
# (s/w/f) and, within cold climates, warm-summer continental (Dfa/Dfb) versus subarctic (Dfc/Dfd).
KG_GROUPS = {
    "kg_A": range(1, 4),                   # tropical
    "kg_B": range(4, 8),                   # arid
    "kg_Cs": range(8, 11),                 # temperate, dry summer (Mediterranean)
    "kg_Cw": range(11, 14),                # temperate, dry winter (monsoonal)
    "kg_Cf": range(14, 17),                # temperate, no dry season
    "kg_Ds": range(17, 21),                # cold, dry summer
    "kg_Dw": range(21, 25),                # cold, dry winter (monsoonal)
    "kg_Df_warm": range(25, 27),           # cold, no dry season, hot or warm summer (Dfa, Dfb)
    "kg_Df_cold": range(27, 29),           # subarctic (Dfc, Dfd)
    "kg_E": range(29, 31),                 # polar / alpine tundra
}
# Heavy-tailed fire metrics are compared on a log10 scale (method v2).
LOG_METRICS = ["fire_density", "fire_size_median", "fire_size_p90"]
# Copernicus LC100 v3 discrete classes.
FOREST = {111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126}
NEEDLE = {111, 113, 121, 123}
BROAD = {112, 114, 122, 124}
MIXED = {115, 125}
BURNABLE = FOREST | {20, 30, 90, 100}       # forest, shrub, herbaceous, wetland, moss

CRITERIA = {
    "climate": list(KG_GROUPS),
    "elevation": ["elev_forest_median", "elev_relief"],
    "forest": ["forest_share", "needle_share", "broad_share", "mixed_share"],
    "fire": ["fire_density", "fire_size_median", "fire_size_p90", "spring_share", "season_concentration"],
}


def read_on_grid(src: rasterio.io.DatasetReader, ref_window_bounds, shape, resampling) -> np.ndarray:
    """Read src over the given bounds, resampled to `shape` (rows, cols)."""
    win = windows.from_bounds(*ref_window_bounds, transform=src.transform)
    return src.read(1, window=win, out_shape=shape, resampling=resampling, boundless=True,
                    fill_value=src.nodata if src.nodata is not None else 0)


def range_metrics(geom, kg_src, dem_src, lc_src) -> dict:
    """Climate, elevation and forest metrics for one range polygon on the 1 km grid."""
    win = windows.from_bounds(*geom.bounds, transform=kg_src.transform).round_offsets().round_lengths()
    transform = kg_src.window_transform(win)
    shape = (int(win.height), int(win.width))
    bounds = windows.bounds(win, kg_src.transform)
    inside = features.geometry_mask([geom], out_shape=shape, transform=transform, invert=True)

    kg = kg_src.read(1, window=win, boundless=True, fill_value=0)[inside]
    dem = dem_src.read(1, window=win, boundless=True, fill_value=-32768).astype(float)[inside]
    lc = read_on_grid(lc_src, bounds, shape, Resampling.mode)[inside]
    dem[dem == -32768] = np.nan

    forest = np.isin(lc, list(FOREST))
    burnable = np.isin(lc, list(BURNABLE))
    climate_mask = forest if forest.mean() > 0.05 else burnable
    kg_sel = kg[climate_mask & (kg > 0)]
    out = {f"{k}": float(np.isin(kg_sel, list(v)).mean()) if kg_sel.size else np.nan for k, v in KG_GROUPS.items()}
    out.update({
        "n_cells": int(inside.sum()),
        "forest_share": float(forest.mean()),
        "burnable_share": float(burnable.mean()),
        "needle_share": float(np.isin(lc[forest], list(NEEDLE)).mean()) if forest.any() else np.nan,
        "broad_share": float(np.isin(lc[forest], list(BROAD)).mean()) if forest.any() else np.nan,
        "mixed_share": float(np.isin(lc[forest], list(MIXED)).mean()) if forest.any() else np.nan,
        "elev_forest_median": float(np.nanmedian(dem[forest])) if forest.any() else np.nan,
        "elev_relief": float(np.nanpercentile(dem, 95) - np.nanpercentile(dem, 5)) if np.isfinite(dem).any() else np.nan,
    })
    return out


def load_fires(fire_dir: Path) -> pd.DataFrame:
    """Read Global Fire Atlas ignitions (lat, lon, size km2, start date) for full years."""
    frames = []
    for year in FIRE_YEARS:
        path = fire_dir / f"GFA_v20250411_ignitions_{year}.shp"
        df = pyogrio.read_dataframe(path, columns=["lat", "lon", "size", "start_DOY"], read_geometry=False)
        df["year"] = year
        frames.append(df)
        logger.info("fires %s: %d", year, len(df))
    return pd.concat(frames, ignore_index=True)


def fire_metrics(fires: pd.DataFrame, ranges: gpd.GeoDataFrame, kg_src) -> pd.DataFrame:
    """Assign fires to ranges via a 1 km rasterised range map, then summarise per range."""
    shape = (kg_src.height, kg_src.width)
    ids = features.rasterize(((g, i) for g, i in zip(ranges.geometry, ranges["GMBA_V2_ID"])),
                             out_shape=shape, transform=kg_src.transform, fill=0, dtype="int32")
    rows, cols = rasterio.transform.rowcol(kg_src.transform, fires["lon"].to_numpy(), fires["lat"].to_numpy())
    rows, cols = np.clip(rows, 0, shape[0] - 1), np.clip(cols, 0, shape[1] - 1)
    fires = fires.assign(range_id=ids[rows, cols])
    fires = fires[fires["range_id"] > 0]
    n_years = len(FIRE_YEARS)
    out = []
    for rid, f in fires.groupby("range_id"):
        lat = ranges.loc[ranges["GMBA_V2_ID"] == rid].geometry.iloc[0].centroid.y
        doy = f["start_DOY"].to_numpy() if lat >= 0 else (f["start_DOY"].to_numpy() + 182) % 365
        angle = 2 * np.pi * doy / 365.0
        month = np.clip((doy / 30.44).astype(int), 0, 11)  # 0 = Jan (northern-hemisphere equivalent)
        out.append({
            "GMBA_V2_ID": rid,
            "n_fires": len(f),
            "fires_per_year": len(f) / n_years,
            "fire_size_median": float(f["size"].median()),
            "fire_size_p90": float(f["size"].quantile(0.9)),
            "spring_share": float(np.isin(month, [1, 2, 3, 4]).mean()),          # Feb-May
            "season_concentration": float(np.hypot(np.cos(angle).mean(), np.sin(angle).mean())),
            "peak_month": int(np.bincount(month, minlength=12).argmax()) + 1,
        })
    return pd.DataFrame(out)


def gower_to_reference(df: pd.DataFrame, ref_id: int, criteria: dict) -> pd.DataFrame:
    """Gower distance of every range to the reference, equal weight per criterion."""
    dist = pd.DataFrame(index=df.index)
    df = df.copy()
    for c in LOG_METRICS:
        df[c] = np.log10(df[c].clip(lower=1e-3))
    ref = df.loc[df["GMBA_V2_ID"] == ref_id].iloc[0]
    for crit, cols in criteria.items():
        parts = []
        for c in cols:
            span = df[c].max() - df[c].min()
            parts.append((df[c] - ref[c]).abs() / span if span > 0 else 0 * df[c])
        dist[crit] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    dist["gower"] = dist[list(criteria)].mean(axis=1, skipna=False)
    for crit in criteria:  # leave-one-criterion-out sensitivity
        others = [c for c in criteria if c != crit]
        dist[f"rank_without_{crit}"] = dist[others].mean(axis=1).rank()
    dist["rank"] = dist["gower"].rank()
    return dist


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("/mnt/CEPH_PROJECTS/Firescape/Data/09_Global_Screening"))
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    ranges = gpd.read_file(args.data / "gmba/GMBA_Inventory_v2.0_standard_300/GMBA_Inventory_v2.0_standard_300.shp")
    ranges = ranges[ranges["Area"] >= MIN_AREA_KM2]
    # Polygons crossing the antimeridian (Arctic Ocean, Chukotka, Aleutians) are arctic or
    # maritime and cannot pass the climate filters; excluded to avoid 360-degree windows.
    ranges = ranges[(ranges.bounds.maxx - ranges.bounds.minx) <= 180].reset_index(drop=True)
    logger.info("ranges >= %d km2, not crossing the antimeridian: %d", MIN_AREA_KM2, len(ranges))

    with rasterio.open(args.data / "koppen/1991_2020/koppen_geiger_0p00833333.tif") as kg_src, \
         rasterio.open(args.data / "dem/wc2.1_30s_elev.tif") as dem_src, \
         rasterio.open(args.data / "landcover/LC100_2019_Discrete-Classification-map.tif") as lc_src:
        rows = []
        for i, r in ranges.iterrows():
            m = range_metrics(r.geometry, kg_src, dem_src, lc_src)
            rows.append({"GMBA_V2_ID": r["GMBA_V2_ID"], **m})
            logger.info("%3d/%d %s", i + 1, len(ranges), r["MapName"])
        metrics = pd.DataFrame(rows)
        fires = fire_metrics(load_fires(args.data / "fire_atlas"), ranges, kg_src)

    df = ranges.drop(columns="geometry")[["GMBA_V2_ID", "MapName", "Path", "Countries", "Area", "Elev_Low", "Elev_High"]]
    df = df.merge(metrics, on="GMBA_V2_ID").merge(fires, on="GMBA_V2_ID", how="left")
    df["n_fires"] = df["n_fires"].fillna(0)
    burnable_km2 = df["Area"] * df["burnable_share"]
    df["fire_density"] = df["fires_per_year"].fillna(0) / burnable_km2.clip(lower=1) * 1000

    df = df.join(gower_to_reference(df, ALPS_ID, CRITERIA))
    df["passes_filters"] = (
        (df[["kg_Cs", "kg_Cw", "kg_Cf", "kg_Ds", "kg_Dw", "kg_Df_warm", "kg_Df_cold"]].sum(axis=1) >= 0.5)  # temperate or cold
        & (df["forest_share"] >= 0.15)                      # a forested montane belt
        & (df["elev_relief"] >= 1000)                       # mountain relief
    )
    df.sort_values("gower").to_csv(args.out / "ranges_screened.csv", index=False)
    short = df[df["passes_filters"] & (df["GMBA_V2_ID"] != ALPS_ID)].nsmallest(15, "gower")
    short.to_csv(args.out / "shortlist.csv", index=False)
    ranges.merge(df[["GMBA_V2_ID", "gower", "rank", "passes_filters"]], on="GMBA_V2_ID").to_file(
        args.out / "ranges_screened.gpkg", driver="GPKG")
    logger.info("Alps rank %s; shortlist:\n%s", df.loc[df.GMBA_V2_ID == ALPS_ID, "rank"].iloc[0],
                short[["MapName", "gower", "climate", "elevation", "forest", "fire"]].to_string())


if __name__ == "__main__":
    main()
