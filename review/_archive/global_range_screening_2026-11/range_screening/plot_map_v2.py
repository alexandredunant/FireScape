"""Map of the screen result: similarity of every mountain range to the European Alps.

World map (Robinson) plus a European inset (LAEA Europe). Ranges are coloured by Gower
distance on a single-hue sequential ramp (dark = most similar); ranges failing the hard
filters are grey and hatched. The 20 closest ranges passing the filters are numbered by
rank and listed beside the map. No retained shortlist is marked.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
COUNTRIES = Path("/mnt/CEPH_PROJECTS/Firescape/Data/09_Global_Screening/naturalearth/ne_110m_admin_0_countries.shp")

# Reference sequential blue ramp (dataviz palette), reversed so that dark = low distance.
BLUE = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
        "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
CMAP = LinearSegmentedColormap.from_list("similarity", BLUE)
NORM = Normalize(vmin=0.0, vmax=0.30)
SURFACE, INK, MUTED, FAIL_FILL, LAND, BORDER = "#fcfcfb", "#2b2b2a", "#6b6b68", "#d9d9d5", "#f1f1ee", "#b9b9b4"
N_LABEL = 20
ALPS_ID = 10001


def load() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    g = gpd.read_file(OUT / "ranges_screened.gpkg")
    g["passes_filters"] = g["passes_filters"].astype(bool)
    top = (g[g["passes_filters"] & (g["GMBA_V2_ID"] != ALPS_ID)]
           .nsmallest(N_LABEL, "gower").reset_index(drop=True))
    top["label"] = range(1, len(top) + 1)
    countries = gpd.read_file(COUNTRIES)
    countries = countries[countries["CONTINENT"] != "Antarctica"]
    return g, countries, top


def draw(ax, g, countries, top, crs, xlim=None, ylim=None, fontsize=6.5) -> None:
    gp, cp = g.to_crs(crs), countries.to_crs(crs)
    cp.plot(ax=ax, color=LAND, edgecolor=BORDER, linewidth=0.3, zorder=1)
    fail = gp[~gp["passes_filters"]]
    ok = gp[gp["passes_filters"]]
    fail.plot(ax=ax, color=FAIL_FILL, edgecolor="#a8a8a3", hatch="////", linewidth=0.2, zorder=2)
    ok.plot(ax=ax, column="gower", cmap=CMAP, norm=NORM, edgecolor=SURFACE, linewidth=0.25, zorder=3)
    alps = gp[gp["GMBA_V2_ID"] == ALPS_ID]
    alps.plot(ax=ax, facecolor="none", edgecolor="#c2410c", linewidth=1.1, zorder=5)
    tp = top.to_crs(crs)
    for _, r in tp.iterrows():
        p = r.geometry.representative_point()
        if xlim and not (xlim[0] <= p.x <= xlim[1] and ylim[0] <= p.y <= ylim[1]):
            continue
        ax.text(p.x, p.y, str(r["label"]), fontsize=fontsize, color=INK, ha="center", va="center",
                fontweight="bold", zorder=6, path_effects=[pe.withStroke(linewidth=2.2, foreground=SURFACE)])
    if xlim:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_axis_off()


def main() -> None:
    g, countries, top = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig = plt.figure(figsize=(13, 8.2), facecolor=SURFACE)
    ax_w = fig.add_axes([0.01, 0.30, 0.74, 0.66], facecolor=SURFACE)
    draw(ax_w, g, countries, top, "ESRI:54030")
    ax_w.set_title("Similarity of the world's mountain ranges to the European Alps", fontsize=11,
                   color=INK, loc="left", fontweight="bold")

    # European inset (LAEA Europe)
    ax_e = fig.add_axes([0.01, 0.0, 0.42, 0.36], facecolor=SURFACE)
    draw(ax_e, g, countries, top, "EPSG:3035", xlim=(2.4e6, 7.6e6), ylim=(1.35e6, 4.9e6), fontsize=7.5)
    ax_e.set_title("Europe, Scandinavia and the Caucasus", fontsize=9, color=INK, loc="left")
    for s in ax_e.spines.values():
        s.set_visible(True); s.set_edgecolor(BORDER)

    # Colour bar and legend
    cax = fig.add_axes([0.47, 0.25, 0.25, 0.018])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Gower distance to the Alps (0 = identical; darker = more similar)", color=INK, fontsize=8)
    cb.outline.set_edgecolor(BORDER); cax.tick_params(colors=MUTED, labelsize=7)
    handles = [Patch(facecolor="none", edgecolor="#c2410c", linewidth=1.1, label="European Alps (reference)"),
               Patch(facecolor=FAIL_FILL, edgecolor="#a8a8a3", hatch="////",
                     label="Fails a filter (climate, forest cover or relief)"),
               Patch(facecolor=LAND, edgecolor=BORDER, label="No major mountain range")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.465, 0.06), frameon=False,
               fontsize=8, labelcolor=INK)

    # Ranked list of the 20 closest ranges passing the filters
    ax_t = fig.add_axes([0.76, 0.06, 0.23, 0.88]); ax_t.set_axis_off()
    ax_t.text(0, 1.0, f"{N_LABEL} closest ranges passing the filters", fontsize=9, fontweight="bold",
              color=INK, va="top", transform=ax_t.transAxes)
    ax_t.text(0, 0.968, "number on map  ·  range  ·  distance", fontsize=7, color=MUTED, va="top",
              transform=ax_t.transAxes)
    for i, r in top.iterrows():
        y = 0.93 - i * 0.043
        ax_t.text(0.0, y, f"{r['label']:>2}", fontsize=8, fontweight="bold", color=INK, transform=ax_t.transAxes, va="top")
        ax_t.text(0.09, y, r["MapName"], fontsize=8, color=INK, transform=ax_t.transAxes, va="top")
        ax_t.text(0.98, y, f"{r['gower']:.2f}", fontsize=8, color=MUTED, transform=ax_t.transAxes, va="top", ha="right")
    ax_t.text(0, 0.0, "GMBA v2 ranges >= 10,000 km² (n = 235).\nScreen and data: review/range_screening/README.md",
              fontsize=6.5, color=MUTED, transform=ax_t.transAxes, va="bottom")

    fig.savefig(OUT / "similarity_map_v2.png", dpi=220, facecolor=SURFACE)
    top[["label", "MapName", "Countries", "gower", "rank"]].to_csv(OUT / "similarity_map_v2_top20.csv", index=False)


if __name__ == "__main__":
    main()
