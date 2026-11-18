# Changes from v5 to v6 (19 October 2026)

`review_paper_v6.tex` is `review_paper_v5.tex` with the changes below. It compiles against `FireScape_master_v7.bib`: 79 pages, 150 references, no undefined citations, no LaTeX errors and no bibtex warnings.

Every new wording was checked against the source PDF before it went in. Where no source supports a claim, the sentence now presents it as our own inference ("we expect…"). The style-sheet rules were followed: no narrative citations, imported mechanisms marked as imported, and single cases counted.

## Housekeeping

- The bibliography is now `FireScape_master_v7.bib`. See `reference_audit_report.md` for every metadata correction.
- Three keys were renamed:
  - `moreno_1_nodate` → `moreno_spacetime_2025`
  - `cagliero_effect_nodate` → `cagliero_effect_2022`
  - `kyriakopoulos_impact_2023` → `bozicek_impact_2023`

## The 23 citations not supported by their source

Line numbers are those of v5.

| v5 line | Source | What changed |
|---|---|---|
| 156 (Table 1) | Cagliero 2022, Šeben 2024 | The Dinarides/Tatras row now says what the two sources cover: long-term fire and land-use history on karst (Dinarides), and windthrow and salvage logging (Tatras). "Fire-prone karst" and "cross-border coordination" are removed. |
| 158 (Table 1) | Rogeau 2018 | The row now credits Rogeau with terrain control of fire refugia in the Canadian Rockies. The post-fire debris-flow science stays with Rengers 2024 (Colorado). |
| 233 | Valese 2011 | Valese is kept for winter fires. The foehn and thermal-belt point is now cited to Müller 2020 and Schunk 2013, where it actually appears. |
| 296 (Table 2) | Forrest 2024 | "Reports explicitly limited skill in mountainous regions" is replaced by what the paper reports: 32–36% deviance explained; temperate and boreal fire occurrence not reproduced; trained on satellite burnt area that misses many small fires. |
| 333 | Forrest 2024 | "Report their own failure in mountains" is replaced by "trained on satellite burnt-area products that miss many small fires". |
| 339 | Diakakis 2023 | The model is now described as the one used operationally in the western United States and tested by Diakakis outside its region of origin. Diakakis no longer appears as its developer, and the missing dNBR term is added. **Add Staley et al. (2017) as the model's original source once its PDF is in `papers/`.** |
| 395 | Forrest 2024 | The deviance figures are kept, and the mountain claim is removed. The text now notes that the model is not evaluated separately in mountains, which is itself a gap. |
| 405 | Forrest 2024 | As at line 395: "does not report its skill for mountain terrain separately". |
| 405 | Ruffault 2018, Beloiu 2026 | The spring-versus-summer claim is removed. The sentence now gives Ruffault's two drought types for summer fires in Mediterranean France (marked as not yet tested in the Alps), and Beloiu's finding that fires above 1,400 m burn under drier conditions. |
| 501 | Asteriou 2025 | The litter-layer and delayed-stem-fall claims are moved to the sources that support them: Melzner 2019 (falling burnt trees carrying rock, one Austrian site) and Maringer 2016 (damaged beech stems collapse within about two decades). Asteriou keeps the simulated loss of the barrier effect. |
| 549 | Frei 2023 | The "participatory modelling of vegetation belts" is in no cited source, so it is removed. Frei is kept for treeline advance in one Swiss valley. **If you know the participatory-modelling study, add it.** |
| 578, 624 | Müller 2020 | "The burn window is narrow and coincides with the spring peak" is now stated as our inference, built from what Müller does say: prescribed burning is little used or prohibited, and human-caused fires peak in March–April. |
| 618 | Reilly 2022 | Rewritten: in the southern Appalachians, exclusion reduced flammability (mesophication), and the 2016 fires followed an acute autumn drought. The source contradicted the old fuel-accumulation claim. |
| 651, 671 | Abrham 2024, Pede 2022 | Abrham is removed from the seasonal-shift claim. It is now cited only for tourism raising ignition risk in Mediterranean destinations. Pede's projection is now limited to one Piedmont zone up to 2050, and the text notes that fire danger is projected to rise mainly in spring. |
| 655 | Moftakhari 2019 | "Greater than the sum of its exposures" is replaced by "projected nearly to double" (38% → 68–73%). |
| 809 | Forrest 2024 | As above: not calibrated on small fires, and not evaluated in mountains separately. |
| 821 | Malanchini 2025 | Malanchini is now cited for what it shows: the two simulators agree on propagation but disagree on intensity. The "will fail in Alpine terrain" prediction is removed, and Hayajneh keeps the experimental finding that separable wind and slope terms degrade on steep slopes. |
| 842 | Beloiu 2026 | Beloiu is moved to the 72 m/decade sentence, which now carries its qualifiers (fires > 30 ha, since 2000, led by the Balkans, Iberia and Turkey, weak in the Alps). The land-abandonment fuel-bed claim is cited to Conedera 2024 and Gerberding 2025. |
| 844 | La Porta 2025 | Threshold recovery is now attributed to Colorado only (Rengers 2024). The claim that window duration matters more is stated as our expectation, and the text adds that no Alpine record yet tracks the recovery. |
| 878 | Mantero 2020 | The search rationale is now stated as our own choice. Mantero is cited only as an English-only database review that acknowledges it may miss literature. |

## The 207 partly supported citations (second pass, 19 October 2026)

All 207 were revised in 164 edits; where several citations shared a sentence, one edit covers them. Every edit is listed with its before and after text and its source evidence in `v6_partial_fixes.md`.

The most consequential changes:
- **Upslope migration:** the 72 m/decade shift is no longer attributed to the Alps (Beloiu 2026).
- **Protective value:** "protective value far exceeds timber value" is withdrawn, because no cited source computes a timber value.
- **Fire danger:** current Alpine danger is described as low, not moderate.
- **Water repellency:** Negri 2026 found that the wildfire *lowered* the naturally high water repellency of pine soils.
- **Smoke and glacier melt:** smoke reduced melt and only soot increased it (one Canadian glacier).
- **Debris-flow thresholds:** Rengers' 33.7 mm/h was a fixed rule, not a result of vegetation recovery.
- **Faunal loss:** "never measured" became "measured only in part" (Rey 2019).
- **Index name:** Cardil's index is renamed the Initial Attack Assessment index.
- **Scope:** single-site and non-Alpine evidence is now marked throughout.

The abstract's 72 m/decade sentence now refers to the elevation of large fires across European mountains.

After these edits, v6 compiles to 81 pages with 150 references, no undefined citations, no LaTeX errors and no bibtex warnings.

## Not yet done

- **PDFs to replace with the version of record:** `senf_mapping_2021`, `riviere_mapping_2023`, `santi_wildfire_2022`.
- **Sources to add once their PDFs are available:** Staley et al. 2017 (line 339), and the Swiss participatory vegetation-belt study (line 549).
- **Reframing for *Reviews of Geophysics*:** title, abstract, Introduction, the new global-regime section, and the AGU template. This waits for the decision on the proposal.
