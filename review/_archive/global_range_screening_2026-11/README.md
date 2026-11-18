# Archived: Global Mountain Range Screening (October–November 2026)

Archived on 18 November 2026 following Eurac Research co-author feedback on the review manuscript's scope.

## Why this material was retired from the main review
- The exploratory framing of *review_paper_v6* and the initial *Reviews of Geophysics* draft proposed screening 235 major mountain ranges from the GMBA Mountain Inventory v2 (using Gower distance across Köppen–Geiger climate, elevation, Copernicus forest cover, and Global Fire Atlas metrics) to select global "comparable ranges".
- Following co-author review, the team decided to keep the review strictly anchored in **the European Alps** (matching Eurac Research's interdisciplinary mandate and the actual corpus of the review) and to centre the analytical argument on **imported fire tools and doctrines versus Alpine reality**, using non-Alpine mountain studies solely as bounded process benchmarks where Alpine measurements do not yet exist.

## Archived contents (preserved intact for potential standalone reuse)
- `range_screening/`: Python scripts (`screen_ranges.py`, `plot_map.py`, `plot_map_v2.py`), methodology `README.md`, and `outputs/` (including `shortlist_final.csv` and `similarity_map_v2.png`). Note: the finding that global satellite burned-area products ($\ge$ 21 ha in the Global Fire Atlas) systematically miss small-fire temperate mountain regimes could stand alone as a separate short technical or data note.
- `range_selection_diagram.tex` / `.pdf` / `.png`: Standalone TikZ workflow diagram of the nine-step GMBA screening pipeline.
- `search_strategy.md`: Draft Scopus / Web of Science query blocks prepared for the retired global comparable-ranges expansion.
