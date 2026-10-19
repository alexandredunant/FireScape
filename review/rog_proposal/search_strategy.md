# Search strategy for the international extension

Draft, 19 October 2026. To be run after the *Reviews of Geophysics* invitation; the deadline is about three months.

## Scope

**Comparative systems,** chosen by global fire-regime type (see `proposal.md`):

| Group | Systems | Expected regime type |
|---|---|---|
| A | Himalaya / Hindu Kush Himalaya (Nepal, Bhutan, Indian Himalaya), Hengduan margin of the Tibetan Plateau | Intermediate–cool–small; human-set, dry-season fire |
| B | North American Cordillera (Rocky Mountains incl. Colorado Front Range and Canadian Rockies, Sierra Nevada, Cascades) | Rare–intense–large; wild |
| C | South-east Australian Alps (Victorian Alps, Snowy Mountains / Kosciuszko); Japan, Taiwan and New Zealand only if post-fire hazard evidence exists | Rare–intense–large (Australian Alps) |

**Excluded:** tropical montane systems (East African mountains, Andean páramo and puna), for the reason stated in the proposal.

**Inclusion rule:** the same as Section 2 of the manuscript. A source from a comparative system is admitted when it documents a process shared with the Alpine case: ignition, spread on slopes, terrain meteorology, post-fire geomorphic response, snow and water, protective function, or exposure. Mechanisms transfer between systems; coefficients do not, and Alpine parameters come from Alpine data only.

**Period:** 2010–2026. Earlier landmark studies are admitted through citation chaining.

**Databases:**
- Scopus and Web of Science for the keyword search.
- OpenAlex and Crossref (free APIs) for backward and forward chaining, metadata verification and DOI resolution.
- National and agency reports from these sources:
  - ICIMOD for the Hindu Kush Himalaya
  - USGS and the US Forest Service for the Cordillera
  - Australian state agencies for the Australian Alps

## Query blocks

Each search is **[SYSTEM] AND [FIRE] AND [DOMAIN]**, run once for each of the six domains. Field: title, abstract and keywords. The syntax is Scopus; adapt it for Web of Science.

**[FIRE]**

```
(wildfire* OR "forest fire*" OR "wildland fire*" OR bushfire* OR "vegetation fire*" OR "fire regime*" OR "burn* area" OR "burn severity" OR "post-fire" OR postfire)
```

**[SYSTEM]**

Group A:
```
(Himalaya* OR "Hindu Kush" OR HKH OR Nepal* OR Bhutan* OR Uttarakhand OR "Himachal Pradesh" OR "Jammu and Kashmir" OR Sikkim OR Hengduan OR "Tibetan Plateau" OR Sichuan OR Yunnan OR "Yajiang")
```

Group B:
```
("Rocky Mountains" OR "Front Range" OR "Canadian Rockies" OR "Sierra Nevada" OR Cascades OR "Pacific Northwest" OR Colorado OR Montana OR Idaho OR Wyoming OR "British Columbia" OR Alberta)
AND (mountain* OR montane OR subalpine OR alpine OR slope* OR terrain)
```

Group C:
```
("Australian Alps" OR "Victorian Alps" OR "Snowy Mountains" OR Kosciuszko OR "alpine ash" OR "Eucalyptus delegatensis" OR (Japan* AND mountain*) OR (Taiwan* AND mountain*) OR ("New Zealand" AND (mountain* OR "high country" OR tussock)))
```

**[DOMAIN]**, one query per domain:

| Domain | Terms |
|---|---|
| D1 Hazard: ignition, spread, danger, data | `(ignition* OR "fire occurrence" OR "rate of spread" OR "slope effect*" OR "fire danger" OR "fire weather" OR FWI OR susceptibility OR "active fire*" OR MODIS OR VIIRS OR "small fire*" OR "burned area product*")` |
| D2 Climate and meteorology | `(foehn OR föhn OR chinook OR "katabatic" OR "valley wind*" OR "mountain-valley" OR "thermal belt" OR inversion OR lightning OR convection OR "elevation-dependent" OR "upslope" OR "fire elevation")` |
| D3 Water and the geomorphic cascade | `("debris flow*" OR "rainfall threshold*" OR "rockfall" OR "landslide*" OR erosion OR "water repellency" OR "sediment yield" OR runoff OR snowpack OR "snow melt" OR glacier* OR "black carbon" OR albedo OR "water supply")` |
| D4 Forest and protective function | `("protection forest*" OR "protective forest*" OR "avalanche" OR treeline OR "tree line" OR regeneration OR "fuel treatment*" OR "prescribed burn*" OR "land abandonment" OR "fuel continuity")` |
| D5 Exposure and vulnerability | `(exposure OR vulnerability OR "wildland-urban interface" OR WUI OR infrastructure OR "risk assessment" OR tourism OR "power line*")` |
| D6 Governance and suppression | `(suppression OR "initial attack" OR "aerial firefighting" OR "fire management" OR governance OR "community forest*" OR "cross-border" OR policy)` |

**Example,** Group A × D3:

```
TITLE-ABS-KEY((Himalaya* OR "Hindu Kush" OR Nepal* OR Bhutan* OR Uttarakhand OR Hengduan OR "Tibetan Plateau" OR Sichuan OR Yunnan)
  AND (wildfire* OR "forest fire*" OR "post-fire" OR postfire OR "burn severity")
  AND ("debris flow*" OR "rainfall threshold*" OR rockfall OR landslide* OR erosion OR snowpack OR glacier* OR "black carbon" OR albedo))
  AND PUBYEAR > 2009
```

## Screening and chaining

1. **Screening:** titles and abstracts are screened against the inclusion rule, with each decision and its reason recorded in `supplementary/FireScape_Alps_screening_register.xlsx`. New tier: "B2 Mountain fire (non-European)", with a column for the regime group (A, B or C).
2. **Seeds:** the three to five most relevant reviews or syntheses per group. Run one round of backward and forward chaining from them through OpenAlex.
3. **Full text:** read before citing. Open-access PDFs go into `papers/` under the `FirstAuthor_Year_First_five_title_words.pdf` convention. Paywalled ones go on a want-list for the authors. Nothing is cited without its PDF being read.
4. **Bookkeeping:** each newly cited source gets a structured note in `supplementary/FireScape_evidence_notes.md`, a Crossref-verified entry in the bibliography, and a row in `papers/FireScape_review_reference_list.csv`. Claims are checked against the PDF using the same procedure as `audit/`.

## Screen first: unused entries already in `FireScape_master_v7.bib`

A scan of the bibliography on 18 October 2026 listed the following uncited entries as candidates. They are not yet verified, so each must pass the inclusion rule and be checked against its PDF.

- **Group B, Cordillera:**
  - Kulakowski 2002/2003/2012, Coop 2009, Sadegh 2021 (upslope advance), Benefield 2022, Harvey 2016, Turner 1999/2019
  - Post-fire debris flows: Cannon 2001/2003/2008/2010, Staley 2013/2017/2018, Kean 2011/2019, Rengers 2016/2021, Wall 2020
  - Snow: Maxwell 2019, Stevens 2017
  - Canadian Rockies: Jordan 2009, Martin 2011
  - Staley 2017 is also needed for the debris-flow model now described at v5 line 339.
- **Group C, Australian Alps:** Williams 2008, Zylstra 2006, Williamson 2014, Green 2006, Camac 2013, Walsh 2004, McDougall 2007, plus the entries on post-fire hillslope response and wind characteristics.
- **Group A, Hengduan / Tibet:** Yang 2022 (Hengduan), and the Yajiang fire studies ("SpatiotemporalDistribution2025", "PostWildfire2025b").

The corpus is thinnest for Group A: the screening register includes only one Himalaya record. Expect that group to need the most new searching.
