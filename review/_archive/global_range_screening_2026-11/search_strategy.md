# Search strategy for the comparable mountain ranges

Revised 21 October 2026. To be run after the *Reviews of Geophysics* invitation; the deadline is about three months.

## Scope

**Comparable ranges** are those retained by the screen in `../range_screening/` (README and `outputs/shortlist_final.csv`). Each was selected for its similarity to the Alps in climate, elevation, forest type and fire regime.

| Group | Ranges (GMBA rank of similarity to the Alps) |
|---|---|
| E. European ranges | Carpathians (11), Dinaric Alps (7), Pyrenees (20), Rila–Rhodope Massif (9), Balkan Mountains (37) |
| C. Caucasus | Greater and Lesser Caucasus (6) |
| R. Rocky Mountains | Southern (19), Western (27) and Canadian (33) Rockies |
| S. Altai–Sayan | Northern Altai (3), Eastern Sayan (5), Western Sayan (16) |

**Inclusion rule:** the same as Section 2 of the manuscript. A source is admitted when it documents a process the review treats in one of its six domains:
- hazard
- climate and weather
- water
- forest
- risk
- governance

Mechanisms transfer between ranges; coefficients do not, and Alpine parameters come from Alpine data only.

**Other ranges:** evidence from ranges outside the shortlist may still be cited for a specific process (for example, post-fire debris-flow thresholds in Colorado), marked as such. These ranges are not searched systematically.

**Period:** 2010–2026; earlier landmark studies enter through citation chaining.

**Databases:**
- Scopus and Web of Science for the keyword search.
- OpenAlex and Crossref for chaining, metadata verification and DOI resolution.
- For group S, also eLIBRARY.ru (Russian) and CNKI (Chinese Altai). Non-English sources are cited only where a co-author or a reliable translation can confirm the content.

## Query blocks

Each search is **[RANGE] AND [FIRE] AND [DOMAIN]**, in title, abstract and keywords. The syntax is Scopus; adapt it for Web of Science.

**[FIRE]**

```
(wildfire* OR "forest fire*" OR "wildland fire*" OR "vegetation fire*" OR "fire regime*" OR "burn* area" OR "burn severity" OR "post-fire" OR postfire)
```

**[RANGE].** The terms are curated because GMBA names often differ from usage, which is why the screen's automatic literature counts are only indicative.

Group E:
```
(Carpathian* OR Karpaty OR Carpați OR Tatra* OR Dinaric OR Dinarides OR Velebit OR Pyrenees OR Pirineos OR Pyrénées OR Pirineus OR Rhodope* OR Rodopi OR Rila OR Pirin OR "Balkan Mountains" OR "Stara Planina")
```

Group C:
```
(Caucasus OR Caucasian OR Kavkaz OR "Greater Caucasus" OR "Lesser Caucasus" OR Georgia* OR Armenia* OR Azerbaijan* OR "North Caucasus")
AND (mountain* OR montane OR forest*)
```

Group R:
```
("Rocky Mountains" OR Rockies OR "Front Range" OR "Canadian Rockies" OR "San Juan Mountains" OR "Sangre de Cristo" OR Wasatch OR Uinta OR Colorado OR Wyoming OR Montana OR Idaho OR Alberta OR "British Columbia")
AND (mountain* OR montane OR subalpine OR alpine OR slope* OR terrain)
```

Group S:
```
(Altai OR Altay OR Sayan OR "Altai-Sayan" OR Tuva OR Khakassia OR "Krasnoyarsk" OR "Kuznetsk Alatau")
AND (mountain* OR montane OR forest* OR taiga)
```

**[DOMAIN]**, one query per domain:

| Domain | Terms |
|---|---|
| D1 Hazard: ignition, spread, danger, data | `(ignition* OR "fire occurrence" OR "rate of spread" OR "slope effect*" OR "fire danger" OR "fire weather" OR FWI OR susceptibility OR "active fire*" OR MODIS OR VIIRS OR "small fire*")` |
| D2 Climate and weather | `(foehn OR föhn OR chinook OR katabatic OR "valley wind*" OR "thermal belt" OR inversion OR lightning OR convection OR "elevation-dependent" OR upslope OR "fire elevation")` |
| D3 Water and the geomorphic cascade | `("debris flow*" OR "rainfall threshold*" OR rockfall OR landslide* OR erosion OR "water repellency" OR "sediment yield" OR runoff OR snowpack OR "snow melt" OR glacier* OR "black carbon" OR albedo)` |
| D4 Forest | `("protection forest*" OR "protective forest*" OR avalanche OR treeline OR "tree line" OR regeneration OR "fuel treatment*" OR "prescribed burn*" OR "land abandonment" OR "fuel continuity")` |
| D5 Risk | `(exposure OR vulnerability OR "wildland-urban interface" OR WUI OR infrastructure OR "risk assessment" OR tourism OR "power line*")` |
| D6 Governance | `(suppression OR "initial attack" OR "fire management" OR governance OR policy OR "cross-border" OR economic* OR cost*)` |

## Procedure

1. **Screening:** titles and abstracts are screened against the inclusion rule, with each decision and its reason recorded in `supplementary/FireScape_Alps_screening_register.xlsx`. New tier: "B2 Comparable range", with a column for the group (E, C, R or S).
2. **Seeds:** the most relevant syntheses per group. Run one round of backward and forward chaining from them through OpenAlex.
3. **Full text:** read before citing. Open-access PDFs go into `papers/` under the `FirstAuthor_Year_First_five_title_words.pdf` convention. Paywalled ones go on a want-list for the authors.
4. **Bookkeeping:** each newly cited source gets a structured note in `supplementary/FireScape_evidence_notes.md`, a Crossref-verified bibliography entry, and a row in `papers/FireScape_review_reference_list.csv`. Claims are checked against the PDF using the procedure in `audit/`.
5. **Writing:** evidence from the comparable ranges enters within each domain section, next to the Alpine evidence, and in the closing mountain versus non-mountain comparison of that section. There are no separate regional sections.

## Already in the collection

Several comparable ranges are already cited in v6:
- Carpathians (Manczinger 2025)
- Pyrenees (Oliveras 2025)
- Dinarides and Tatras (Cagliero 2022, Šeben 2024)
- Rocky Mountains (Rengers 2024, Rogeau 2018, Aubry-Wake 2022)

The bibliography also holds uncited Rocky Mountain entries (Kulakowski, Coop 2009, Sadegh 2021, Harvey 2016; Cannon, Staley and Kean on post-fire debris flows). Screen these first; Staley 2017 is also needed at v5 line 339.

The Caucasus and Altai–Sayan are not yet represented in the corpus, so they will need the most new searching.
