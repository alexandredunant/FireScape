# Screen for mountain ranges comparable to the European Alps

This folder selects the comparable mountain ranges (tier 2) of the review *What makes mountain wildfire different?* using a reproducible screen. It replaces selection by convenience. Every major mountain range is described on four criteria and ranked by its distance to the European Alps.

## Criteria and data

| Criterion | Metric per range | Dataset |
|---|---|---|
| Unit of analysis | 235 GMBA major ranges of at least 10,000 km², not crossing the antimeridian | GMBA Mountain Inventory v2.0, standard extent, 300-range selection (Snethlage et al., 2022, *Sci. Data* 9:149; data doi:10.48601/earthenv-t9k2-1407) |
| Climate | Share of the forested part of the range in ten Köppen–Geiger groups: A; B; Cs; Cw; Cf; Ds; Dw; Df with warm summer (Dfa, Dfb); subarctic Df (Dfc, Dfd); E. The whole burnable area is used where forest is below 5% | Köppen–Geiger 1 km, 1991–2020 (Beck et al., 2023, *Sci. Data* 10:724; figshare 21789074) |
| Elevation | Median elevation of forest; relief (95th minus 5th percentile of elevation) | WorldClim 2.1 elevation, 30 arc-seconds (Fick & Hijmans, 2017) |
| Forest type | Forest share of the range; needleleaf, broadleaf and mixed shares of the forest | Copernicus Global Land Service LC100 v3.0.1, 2019, discrete classification (Buchhorn et al., 2020; Zenodo 3939050) |
| Fire regime | Fires per 1,000 km² of burnable land per year; median and 90th-percentile fire size; share of fires starting February–May; concentration of the fire season (mean resultant length of start dates). All are computed on hemisphere-adjusted dates | Global Fire Atlas, extended version v20250411, ignitions 2003–2023 (Andela et al., 2019, *ESSD* 11:529; Zenodo 14065246) |

**Processing grid.** All rasters are read per range on the 1 km Köppen–Geiger / WorldClim grid. The two share an identical 43,200 × 21,600 grid. The 100 m land cover is resampled to that grid with a majority rule. Fires are assigned to ranges by rasterising the range polygons on the same grid; the GMBA 300 selection has no overlapping polygons.

**Unused download.** RESOLVE Ecoregions 2017 (Dinerstein et al., 2017) was downloaded but is not used in the score. Forest type comes from the land cover, which resolves leaf type and phenology directly.

## Similarity and filters

- **Distance:** Gower distance to the European Alps (GMBA_V2_ID 10001). Fire density and fire sizes are log10-transformed because they are heavy-tailed. Each metric is then scaled by its range across all mountain ranges. Metrics are averaged within each criterion, and the four criteria carry equal weight.
- **Sensitivity:** ranks are recomputed leaving out each criterion in turn (`rank_without_*` columns).
- **Hard filters, applied for the shortlist:**
  - at least half of the forested part is temperate or cold (Köppen Cs, Cf/Cw or D)
  - forest covers at least 15% of the range
  - relief of at least 1,000 m
- **Shortlist:** the 15 closest ranges that pass the filters. They are then pruned by available fire literature before entering the review.

## Caveat: what the fire-regime criterion can see

The Global Fire Atlas records fires of at least 21 ha, one MODIS pixel. Around 80% of Alpine fires are smaller than one hectare (Müller et al., 2020), so in the atlas the Alps appear as a regime with few detected fires. The criterion therefore compares ranges as global satellite products see them, not as national inventories record them. We report this as a result in its own right: it is how global products see small-fire mountain regimes.

## Method versions

- **v1 (first pass, archived in `outputs/v1_firstpass/`):**
  - six Köppen groups, with all cold (D) climates merged
  - linear scaling of the fire metrics

  Two problems followed:
  - 74% of the Alpine forest belt fell in the single D group, so boreal Siberian and Canadian ranges looked climatically identical to the Alps.
  - The fire metrics were compressed by the extreme savanna ranges, so nearly all temperate ranges looked equally close on fire regime.
- **v2 (current):** keeps the Köppen distinctions that matter for fuel and fire season, and log-scales the heavy-tailed fire metrics.
  - Precipitation seasonality: s, w, f.
  - Warm-summer continental (Dfa, Dfb) versus subarctic (Dfc, Dfd) climates.

  Both changes were made after inspecting the v1 ranking, on methodological grounds, and before the shortlist was chosen. No further changes were made to move particular ranges up or down. Ranks under v1 are kept for comparison.

## Results (v2, 21 October 2026)

- **Sanity checks:**
  - The Alps rank first against themselves (distance 0).
  - Arid and Saharan ranges rank last: Saharan Atlas 186th, West Sahara Mountains 231st.
  - The Himalaya rank 167th.
- **Close analogues passing all filters:**
  - Pontic Mountains 2nd, Northern Altai 3rd, Eastern Sayan 5th, Caucasus 6th
  - Dinaric Alps 7th, Rila-Rhodope 9th, Carpathians 11th
  - Southern Rocky Mountains 19th, Pyrenees 20th
- **Ranges often used as comparators that rank lower:**
  - Sierra Nevada 54th, Cascade Range 58th, Southern Alps of New Zealand 103rd
  - Southern Great Dividing Range (south-east Australian Alps) 180th
  - The Apennines fall inside the GMBA aggregate "South European Highlands", 141st.
- **Comparable ranges retained** (decision of 21 October 2026, recorded in `outputs/shortlist_final.csv`):

  | Group | Ranges (rank) |
  |---|---|
  | European core | Carpathians (11), Dinaric Alps (7), Pyrenees (20), Rila-Rhodope (9), Balkan Mountains (37) |
  | Caucasus | Caucasus (6) |
  | Rocky Mountains | Southern (19), Western (27), Canadian (33) |
  | Altai–Sayan | Northern Altai (3), Eastern Sayan (5), Western Sayan (16) |

  The Pontic Mountains (2nd) were not retained because their fire literature is very small (5 OpenAlex hits).
- **Other ranges:** ranges not retained can still contribute evidence on a specific process under the review's transfer rule (mechanisms transfer, coefficients do not). Examples are post-fire debris-flow thresholds from Colorado, or fire–snow coupling in the Sierra Nevada. They are not presented as regime analogues of the Alps.
- **Literature counts:** `outputs/shortlist_literature.csv` holds OpenAlex counts of wildfire publications per range. They are indicative only, because GMBA names often differ from usage ("Dinaric Alps" vs "Dinarides", "Scandes" vs "Scandinavian mountains"), and generic names ("Coast", "Columbia") inflate counts. The evidence step uses curated synonyms in `../rog_proposal/search_strategy.md`.

## Figure

`outputs/similarity_map.png` maps all ranges by Gower distance, with the retained ranges outlined (`plot_map.py`). The labels overlap in Europe; a publication version needs a European inset.

## Reproduce

Raw data are in `Data/09_Global_Screening/`, which is not tracked by git; download links are in the table above.

```bash
conda activate hazard_agent
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python screen_ranges.py
```

Outputs, in `outputs/`:
- `ranges_screened.csv`: all ranges, with metrics, criterion distances, Gower distance and ranks.
- `shortlist.csv`: the 15 closest ranges passing the filters.
- `ranges_screened.gpkg`: range polygons with distance and rank, for mapping.
- `shortlist_final.csv`: the retained comparable ranges, with criterion distances and key metrics.
- `similarity_map.png`, from `plot_map.py`.
- `run_v2.log`: the log of the current run.

## Licences

- **CC-BY 4.0:** the Köppen–Geiger maps, Copernicus LC100 and the Global Fire Atlas, as stated on their Zenodo or figshare records.
- **GMBA Inventory v2:** offered for unrestricted download by GMBA. Cite Snethlage et al. (2022).
- **WorldClim:** free for research use under the WorldClim terms.

Cite the sources above when using the outputs.
