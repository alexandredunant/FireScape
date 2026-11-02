"""World map of similarity to the European Alps, with the retained comparable ranges."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    g = gpd.read_file(OUT / "ranges_screened.gpkg")
    keep = pd.read_csv(OUT / "shortlist_final.csv")
    sel = g[g["GMBA_V2_ID"].isin(keep["gmba_id"])]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    g.plot(ax=ax, column="gower", cmap="viridis_r", vmin=0, vmax=0.35, linewidth=0.1, edgecolor="white",
           legend=True, legend_kwds={"label": "Gower distance to the European Alps", "shrink": 0.6})
    sel.boundary.plot(ax=ax, color="red", linewidth=0.8)
    for _, r in sel.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r["MapName"].replace(" Mountains", ""), (p.x, p.y), fontsize=6, color="red",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-170, 180); ax.set_ylim(-58, 80); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Mountain ranges (GMBA v2, >= 10,000 km2) by similarity to the European Alps; "
                 "retained comparable ranges outlined", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "similarity_map.png", dpi=200)


if __name__ == "__main__":
    main()
