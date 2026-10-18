# Evidence notes — expansion batch 1 (open-access, read 2026-08-20)

## gelabert2025_nhess — Assessing human-caused wildfire ignition likelihood across Europe
Gelabert, Jiménez-Ruano, Ochoa, Alcasena, Sjöström, Marrs, Ribeiro, Palaiologou, Bentué Martínez, Chuvieco, Vega-García, Rodrigues. NHESS 25:4713, 2025. doi:10.5194/nhess-25-4713-2025. OA.
- Five pilot sites incl. Central Europe (DE/CZ/PL); fires >1 ha, 1996–2022; Random Forest, 1000 pseudo-absence iterations.
- Site AUC 0.70–0.89 (Central Europe 0.89, Attica 0.70); pooled multi-site model AUC 0.81 but regional range 0.61–0.85, dropping 0.10 in Sweden and Attica.
- Yearly anomaly in dead fine fuel moisture content = most influential variable at ALL sites. Human factors ranked: distance to WUI > road proximity > population density > wildland fraction.
- Limits: overfitting where sample small; Euclidean distance fields prevent extrapolation beyond bounding box; uses STATIC human factors, recommends daily (commuting) and seasonal (tourism/recreation) dynamics.
- USE: transferability of ignition models; WUI as first-order human predictor; argues exposure is dynamic.

## ozaki2022_fire — Impact of Vertical Atmospheric Structure on an Atypical Fire in a Mountain Valley
Ozaki, Harris, Love, Aryal, Fox-Hughes, Williamson. Fire 5(4):104, 2022. doi:10.3390/fire5040104. OA.
- Riveaux Road Fire, southern Tasmania, ~637 km2, lightning-ignited Jan 2019, terrain to 1375 m.
- "Conducive wind structures" (wind+temperature layering producing enhanced surface winds): present in 6.51% of profiles in least-rugged terrain vs 40.15% in most-rugged terrain.
- Simulations using upper-air data scored 80% usefulness in gentle terrain, declining with terrain complexity; surface-only downscaled models showed the inverse.
- USE: hard number for terrain-dependence of fire weather; surface-only danger systems inadequate in complex terrain.

## hayajneh2026_fire — Wind and Slope Effects on Wildland Fire Spread: A Review
Hayajneh, Alzghoul, Naser. Fire 9(3):100, 2026. doi:10.3390/fire9030100. OA.
- Scoping review, >150 studies, 4 modelling traditions.
- Slope: radiative preheating dominates <=20 deg; mixed regime ~20 deg; convection-led >=30 deg. Bench study: ROS x2.5 from 20 to 30 deg. Eruption/flame attachment near 25–30 deg with confined geometry (aspect ratio >=0.2). Downslope non-monotonic: decreases to -20 deg, re-intensifies at -30 deg.
- Wind: ~1.5 m/s threshold for radiation->convection crossover. Pyrotron eucalypt litter ROS 33.6 -> 187.4 m/h; pine 155.7 -> 575.4 m/h. At 3.0 m/s and 0.8 kg/m2, ROS ~order of magnitude above still air. Discontinuous fuel regimes: no ignition <=1.0 m/s; direct contact 1.5–2.5 m/s; spotting >=3.0 m/s.
- Model failures: separable wind/slope correction factors miss nonlinear interaction; Rothermel-based operational models underestimate spread on steep slopes with wide fronts, cannot represent canyon blow-up, miss downslope re-intensification.
- USE: the quantitative core of the topography/fire-behaviour section; direct evidence that operational spread models are biased in exactly Alpine geometry.

## degiuli2024_diversity — Arrested Succession on Fire-Affected Slopes in the Krummholz Zone
De Giuli, Winkler, Deola, Henschel, Sass, Wolff, Jentsch. Diversity 16(7):366, 2024. doi:10.3390/d16070366. OA.
- Arnspitze massif (2196 m), Wetterstein, Northern Limestone Alps; 41 plots (26 burned/15 unburned) at 1800–1950 m; orthophotos 1974–2021.
- Burned plots richer: 35.1 +/- 8.8 vs 25.8 +/- 11.5 species/plot (p=0.05); forbs 64% vs 20% relative cover.
- Near-complete absence of tree seedlings; NO meaningful Pinus mugo re-establishment over 80 years post-fire (1946–2023).
- USE: decisive counterexample to the "3–4 year window" of post-fire susceptibility; at subalpine elevations the loss of protective cover can be effectively permanent.

## malandra2022_fire — Burn Severity Drivers in Italian Large Wildfires
Malandra, Vitali, Morresi, Garbarino, Foster, Stephens, Urbinati. Fire 5(6):180, 2022. doi:10.3390/fire5060180. OA.
- 113 large wildfires (>=100 ha), 2007 and 2017, ~102,565 ha, central and southern Italy; random forest.
- Land cover best predictor: conifer RdNBR 357 > shrubland 293 > mixed 288. Elevation second: higher severity at 0–650 m. Population density >3000 inhab/km2 associated with higher severity. Positive relation with distance to road and railway (remote = higher uncontrolled severity). Slope weak positive, 0–16 deg. TWI and heat load least important.
- USE: severity is fuel-type-led, not slope-led, at this scale; access/remoteness enters severity empirically — supports the suppression-constraint argument.

## hillebrand2023_ffgc — Contrasting impacts of climate change on protection forests of the Italian Alps
Hillebrand, Marzini, Crespi, Hiltner, Mina. Front. For. Glob. Change, 2023. doi:10.3389/ffgc.2023.1240235. OA.
- South Tyrol; ForClim on 11 sites along elevation gradient; CHELSA V2.1; SSP1-2.6 and SSP3-7.0 to 2100.
- Under severe scenario montane forests show drastic decline in avalanche protection index (to 0.03 by 2100); rockfall protection declines at lower elevation; subalpine forests resilient or improving. Dry inner Alpine valleys worst: drought-induced mortality with biomass reductions up to 97%.
- Limits: no herbaceous competition (regeneration possibly overestimated); monthly climate resolution may overestimate drought; insect and wind disturbance excluded.
- USE: elevation-contingent loss of protective function; the montane/subalpine contrast.

## moos2021_ffgc — Climate Change Impacts the Protective Effect of Forests: Switzerland
Moos, Guisan, Randin, Lischke. Front. For. Glob. Change 4:682923, 2021. doi:10.3389/ffgc.2021.682923. OA.
- Protection forest above Martigny, Valais, 500–1000 m; TreeMig + 3D rockfall simulation + quantitative risk analysis.
- Risk 19,100 CHF/yr with forest vs 95,500 CHF/yr without (5x). Rising to 59,250 (RCP4.5) and 70,100 CHF/yr (RCP8.5) within 150 yr. Composition shifts Fagus -> Quercus pubescens. Small blocks (0.1–0.5 m3) reaching railway once/decade now -> once/3 yr by 2150.
- USE: the monetised protective-function argument; the 5x factor is the single best number for "the forest is the mitigation measure".

## asteriou2025_geohazards — Assessing Post-Fire Rockfall Hazards: Evros, Greece
Asteriou, Sotiriadis, Petala, Kazelis. GeoHazards 6(3):54, 2025. doi:10.3390/geohazards6030054. OA.
- Avantas, Evros; August 2023 fire >90,000 ha; FiRo-RiS framework; RockyFor3D + RocFall2.
- Small blocks (0.01 m3): most stop on slope pre-fire, travel much further post-fire. Effect diminishes for larger blocks (0.125 m3) where kinetic energy overwhelms vegetation resistance.
- No timeline given for hazard reduction.
- USE: block-size dependence of the vegetation-loss effect — a mechanism detail the Alpine literature asserts but does not quantify.

## zacharakis2023_land — Environmental Forest Fire Danger Rating Systems and Indices around the Globe: A Review
Zacharakis & Tsihrintzis. Land 12(1):194, 2023. doi:10.3390/land12010194. OA.
- 63 systems/indices across 16 countries. Where FWI is used outside Canada, danger classes are redefined by local calibration.
- Few systems handle terrain: NFDRS uses three slope classes; LASI uses pressure levels by elevation.
- USE: independent corroboration that index transfer requires recalibration, and that terrain is largely absent from operational danger systems worldwide.

## venalainen2014_nhess — Temporal variations and change in forest fire danger in Europe 1960–2012
Venäläinen, Korhonen, Hyvärinen, Koutsias, Xystrakis, Urbieta, Moreno. NHESS 14:1477, 2014. doi:10.5194/nhess-14-1477-2014. OA.
- No obvious trend 1960–1999 (ERA-40); mean FWI increasing for southern and eastern Europe 1980–2012 (ERA-Interim), significant at 99%.
- Cross-correlation burned area vs fire-season weather indices ~0.6 in Greece, Spain, Finland.
- USE: baseline for "danger trend is detectable but weather explains only part of burned area".

## frei2023_forests — Evidence for 40 Years of Treeline Shift in a Central Alpine Valley
Frei, Barbeito, Erdle, Leibold, Bebi. Forests 14(2):412, 2023. doi:10.3390/f14020412. OA.
- Dischma valley, Davos, 1600–2800 m; surveys 1972/73 vs 2012.
- Treeline advanced on average 10 m/decade, max local 42 m/decade. 21% of new trees are advance, 79% infilling below the previous treeline. Larch +47 m on south-facing, pine +36 m on north-facing.
- Drivers: warming above 2330 m; aspect, curvature, snow disappearance timing; dwarf-shrub competition; 1900 land use at lower elevations; current grazing no significant effect; avalanche causes local downward shift.
- USE: quantifies the fuel-moving-upslope claim, and the 79% infilling figure is the fuel-continuity argument in numbers.

## bercak2024_fire — A Combination of Human Activity and Climate Drives Forest Fire Occurrence in Central Europe (Czech Republic)
Berčák, Holuša, Trombik, Resnerová, Hlásny. Fire 7(4):109, 2024. doi:10.3390/fire7040109. OA.
- Czech Republic 2006–2015. Annual scale: human activity dominates, climate not significant, model explains 71.4% of variability. Monthly/summer scale: climate significant, 48.7–53.8% explained.
- Bimodal seasonality: spring peak April (median burned area 0.04 ha), summer peak Jul–Aug (median 0.005 ha). Spring linked to forestry operations after winter; summer to heat, drought and tourism, with threshold response above 18 degC.
- USE: independent, non-Alpine-arc corroboration of the bimodal central European regime — removes the reliance on Valese alone.

## forrest2024_bg — BASE (Burnt Area Simulator for Europe)
Forrest, Hetzer, Billing, Bowring, Kosczor, Oberhagemann, Perkins, Warren, Arrogante-Funes, Thonicke, Hickler. Biogeosciences 21:5539, 2024. doi:10.5194/bg-21-5539-2024. OA.
- GLMs at ~9x9 km monthly. Europe 2001–2020 burnt area ~1.46 Mha/yr: cropland 1.09 Mha/yr (74%), non-cropland vegetation 0.38 Mha/yr (26%).
- Explains 32% (NCV) and 36% (cropland) of deviance; underestimates peak burning.
- Explicitly limited skill in mountainous and temperate regions.
- USE: continental fire models self-report failure in mountains — direct support for the scope-bias argument.

## ruffault2018_nhess — Extreme wildfire events linked to global-change-type droughts in the northern Mediterranean
Ruffault, Curt, Martin-StPaul, Moron, Trigo. NHESS 18:847, 2018. doi:10.5194/nhess-18-847-2018. OA.
- Southern France (PACA), JJAS 1996–2016. Six extreme events in 2003, three in 2016.
- Two drought types: "press drought" (2016, long water deficit, wind-driven fires) and "hot drought" (2003, water deficit plus heat wave, heat-induced fires).
- USE: the drought typology distinction, useful for separating Alpine spring/summer mechanisms.

## pais2023_fire — Optimizing Wildfire Prevention through Integration of Prescribed Burning into Fire-Smart Land-Use Policies
Pais, Aquilué, Honrado, Fernandes, Regos. Fire 6(12):457, 2023. doi:10.3390/fire6120457. OA.
- Prescribed fire alone reduced future wildfire by up to 36% over 30 years; leverage 0.07–1.78. Strategic allocation to high-risk zones: leverage 0.32–0.45. Combined with High Nature Value farmland + fire-smart conversion: leverage 1.78 (1 ha wildfire avoided per ha treated).
- Advocates EU-CAP support for agropastoral activity.
- USE: quantifies the grazing/land-use lever that Rouet-Leduc argues for qualitatively, and gives prescribed fire a benchmark the Alps can be compared against.

## senf2021_natsust — Mapping the forest disturbance regimes of Europe
Senf & Seidl. Nature Sustainability, 2021. doi:10.1038/s41893-020-00609-y. OA.
- Landsat 1986–2016, LandTrendr. 17% of Europe's forest area disturbed over 30 years; 36 million patches, mean patch size 1.09 ha, 0.52 patches/km2/yr. Disturbance frequency increasing, severity decreasing.
- Alps conifer and mixed forests: substantially higher disturbance frequency in Austria than Italy.
- USE: the disturbance baseline against which fire must be sized; mean patch 1.09 ha frames why Alpine fires are small.

## cardil2025_ijwf — Assessing the suppression difficulty of wildland fires for initial attack response
Cardil et al. Int. J. Wildland Fire 34:WF24160, 2025. doi:10.1071/WF24160.
- California 2020–2023, 26,907 fires; validation in Oregon, Colorado, Montana and Spain.
- Terrain Difficulty Index = accessibility + penetrability + construction indices + average slope + road proximity; combined with Fire Behaviour Index into Initial Attack Assessment (1–5).
- IAA-5 vs IAA-1: 90% decrease in odds of initial attack success. Mean burned area 18.5 ha (IAA-1) to 1731 ha (IAA-5). Response time interacts with TDI to reduce containment probability. Overall 97.02% of fires contained at <=4 ha.
- Reported adoption beyond California incl. British Columbia and European agencies.
- USE: the framework for terrain-conditioned suppression exists and is transferable — so the Alpine gap is application, not method. Revises the earlier "no evidence base" claim.

## stritih2023_landecol — Alternative states in the structure of mountain forests across the Alps
Stritih, Seidl, Senf. Landscape Ecology, 2023. doi:10.1007/s10980-023-01597-y.
- Entire Alpine Convention area, ~10.5 Mha forest.
- Two structural states: closed canopy 76% of area (cover >~0.75, height percentile >~10 m), open canopy 24% (high elevation, forest edges, warm dry sites).
- Disturbance initiated transition to open canopy in ~50% of cases; 72% of Alpine forests returned to closed canopy within 35 years; submediterranean forests much slower, staying open for decades.
- USE: quantifies recovery of protective structure at Alpine scale, and the 35-year figure bridges the 3–4 year soil window and the 80-year krummholz result.

## laporta2025_landslides — Post-wildfire debris flow in the NW Italian Alps: the June 2018 Bussoleno event
La Porta, Leonardi, La Ferlita, Pirulli. Landslides 22(11):3625, 2025. doi:10.1007/s10346-025-02605-9.
- Comba delle Foglie basin, 1.37 km2, 490–1790 m, Susa Valley. October 2017 fire burned >95% of catchment; spring 2018 rainfall 40% above normal; debris flow 7 June 2018, ~8 months post-fire.
- ~20,000 m3 mobilised; two residential buildings destroyed, ~150 residents flooded from the alluvial fan.
- Trigger: radar-estimated 12 mm in 30 min = 24 mm/h, 5-year return period — BELOW the >30 mm/h threshold typical for unburned basins of this type.
- Mechanical shear strength of burned soils reduced by a factor of three. RASH3D depth-averaged model; Bingham rheology best (yield stress 1300 Pa, viscosity 80 Pa s). Susceptibility-based triggering used 84,000 m3 of identified unstable soil.
- USE: the single best-quantified Alpine post-fire cascade event; the sub-threshold triggering is the key risk-transfer argument.
# Evidence notes — expansion batch 2 (open-access, read 2026-08-20)

## muller2020_eusalp — Forest Fires in the Alps: State of Knowledge, Future Challenges and Options for an Integrated Fire Management
Müller M.M., Vilà-Vilardell L., Vacik H. (2020), EUSALP Action Group 8 white paper. OA PDF.
- Annual averages 2006–2017 (fires/yr; ha/yr): Italy 1043 / 9984; France 213 / 818; Austria 122 / 64; Switzerland 105 / 515; Slovenia 37 / 201.
- Burnt area per 10,000 ha forest per year: Italy 44.13; France 5.80; Switzerland 5.19; Slovenia 2.77; Austria 0.22. Two orders of magnitude spread across the arc.
- ~80% of Alpine fires burn <1 ha. Median fire size <0.1 ha (AT, DE, CH) to 1 ha (FR, IT).
- ~90% of Alpine fires human-caused (cigarettes, escaped fires, sparks from trains/work, arson, hot ashes, power lines); lightning ~10% regionally but up to 50% in summer in some areas.
- Two seasons: spring peak Mar–Apr, summer peak Jul–Aug, plus winter fires in foehn-influenced regions.
- Direct firefighting + post-fire management cost ~EUR 75 million/yr across the Alpine region.
- Proposed integrated fire management framework ~EUR 10 M/yr: prevention 5–7, suppression 1.5–2.5, post-fire 1–2, knowledge transfer 0.5.
- States firefighting is generally difficult in the Alps due to rugged topography and low accessibility; coniferous forests on southern slopes providing protection against avalanche, rockfall and mudslide are disproportionately at risk; fires create new avalanche-prone slopes, higher rockfall risk, mudslides, erosion and local hydraulic regime change.
- Concludes current efforts "are unable to prevent the occurrence of extreme forest fire events".
- USE: the single best source of hard Alpine baseline statistics and of the cost argument. Replaces several places where Valese or Müller-2020-EcoInf were being reused.

## beloiu2026_natcomm — Climate-driven upward spread of forest fires in European mountain regions
Beloiu et al. Nature Communications, 2026. doi:10.1038/s41467-026-72551-0. OA.
- Eight European mountain regions incl. the Alps; 2000–2025; fires >=800 m and >=30 ha.
- Fires spread upward ~72 m per decade. Median fire elevation 910 m (2000–2014) -> 1004 m (2015–2025). 90th percentile of annual fire elevations 1353 -> 1460 m; top-ten annual fires 1451 -> 1777 m.
- After 2015, fire occurrence increasing 30%/yr below 1400 m and 12%/yr above.
- VPD strongest predictor of burned area; fires above 1400 m linked to extreme atmospheric aridity.
- Limitation the authors state: only fires >30 ha, and between 1980 and 2010 the vast majority of Alpine wildfires were smaller than 10 ha — so emerging high-elevation activity is likely underestimated.
- USE: replaces palaeo-inference as the primary evidence for upslope migration; the >30 ha threshold caveat is itself an argument about Alpine observability.

## mutterer2026_fireecol — Bark traits and thermal resistance to wildfires: six Central European tree species
Mutterer, Sheppard, Kohnle, Seifert. Fire Ecology, 2026. doi:10.1186/s42408-025-00438-4. OA.
- 129 stem discs heated at 212.3 +/- 5.2 degC; time for cambium to reach the lethal 60 degC (tau60).
- Bark thickness is the most robust predictor of insulation. Resistance ranking: Larix decidua, Pseudotsuga menziesii, Pinus sylvestris (higher) > Abies alba, Picea abies, Fagus sylvatica (lower, beech lowest).
- Thermal resistance decreases with bark moisture content and with height on the stem; bark density not significant.
- USE: experimental corroboration of Dupire's trait-based vulnerability ranking, from an independent method. Removes reliance on Dupire alone for the species-vulnerability claim.

## diakakis2023_land — Debris flow likelihood regression model in Mediterranean post-fire environments
Diakakis, Mavroulis, Vassilakis, Chalvatzi. Land 12(3):555, 2023. doi:10.3390/land12030555. OA.
- Schinos-Alepochori, Attica, after the May 2021 fire; 21 basins.
- Adapts the Staley et al. logistic model developed for the western USA: predictors are proportion of high/moderate soil burn severity on gradients >=23 deg, dNBR, and soil erodibility K-factor, multiplied by peak 15-minute rainfall.
- Probabilities 0.05–0.893; correlation with observed debris flows significant at 0.001; ROC AUC 0.913.
- Authors report high transferability, noting it runs on European soil datasets.
- USE: important nuance — physically grounded post-fire debris-flow models DO transfer from the US to Europe, unlike weather-only fire danger indices. Sharpens the transferability argument.

## avouris2025_egusphere — Exposure of Settlements to Wildfires in a Transboundary WUI Region in Central Europe (PREPRINT)
Avouris, Marrs, Beetz, Kudláčková, Poděbradská, Trnka, Forkel. EGUsphere preprint, 2025. doi:10.5194/egusphere-2025-4859.
- Saxon-Bohemian Switzerland, DE/CZ border, ~700 km2; FlamMap with 3 fire durations and 3 wind directions under medium to very high danger; validated on the 2022 fire.
- Higher burn probability and potential flame length at settlements on the Czech side; wildfires disregard political boundaries.
- Caveats from review: standard methods, validation dependent on remote sensing commission/omission errors, weather coverage favours the German side.
- STATUS: preprint — cite as such or replace with the published version.
- USE: the transboundary exposure argument, with a Central European rather than Mediterranean example.

## cardil2025_ijwf — see batch1. Terrain Difficulty Index / Initial Attack Assessment.

## keeping2026_egusphere — Attribution of the record-breaking 2025 European fire season (PREPRINT)
Keeping, Zachariah, Haas, Grillakis, Barnes, Clay, Ekberzade, Jaupaj, Ribeiro, Trigo, Voulgarakis, Otto. EGUsphere preprint, 2026. doi:10.5194/egusphere-2026-1193.
- Record burned area >1,400,000 ha in 2025. Strong trends to drier summers and extreme weekly VPD; summer drought emergent from natural variability in most southern European regions, VPD extremes emergent in all five regions studied.
- Five regions: NW Iberia, N/W Britain, Occitania, eastern Adriatic/Ionian, N/W Türkiye. Explicitly EXCLUDES the Alps and central Europe.
- STATUS: preprint.
- USE: use the exclusion as evidence — the flagship European attribution study of the record season does not cover the Alps.

## baranovskiy2021_forecasting — Deterministic-probabilistic prediction of lightning-caused forest fires in mountain areas
Baranovskiy. Forecasting 3(4):695, 2021. doi:10.3390/forecast3040043. OA.
- Republic of Altay and Buryatia, Russia. Five probability factors incl. an explicit altitude term; deterministic unsteady heat-transfer ignition models for deciduous vs coniferous wood.
- Mountain-specific features: cloud-to-ground discharge frequency increases with elevation; altitudinal forest belts change fuel; precipitation varies strongly across mountain zones.
- No validation accuracy reported (synthetic discharge data).
- USE: mechanism for elevation-dependence of lightning ignition; weak on validation, so use for structure not for numbers.

## Manifest of landing pages for PDF download into papers/
- https://nhess.copernicus.org/articles/25/4713/2025/
- https://www.mdpi.com/2571-6255/5/4/104
- https://www.mdpi.com/2571-6255/9/3/100
- https://www.mdpi.com/1424-2818/16/7/366
- https://www.mdpi.com/2571-6255/5/6/180
- https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2023.1240235/full
- https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2021.682923/full
- https://www.mdpi.com/2624-795X/6/3/54
- https://www.mdpi.com/2073-445X/12/1/194
- https://nhess.copernicus.org/articles/14/1477/2014/
- https://www.mdpi.com/1999-4907/14/2/412
- https://www.mdpi.com/2571-6255/7/4/109
- https://bg.copernicus.org/articles/21/5539/2024/
- https://nhess.copernicus.org/articles/18/847/2018/
- https://www.mdpi.com/2571-6255/6/12/457
- https://www.nature.com/articles/s41893-020-00609-y
- https://link.springer.com/article/10.1007/s10980-023-01597-y
- https://link.springer.com/article/10.1007/s10346-025-02605-9
- https://link.springer.com/article/10.1186/s42408-025-00438-4
- https://www.mdpi.com/2073-445X/12/3/555
- https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4859/
- https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1193/
- https://www.mdpi.com/2571-9394/3/4/43
- https://www.nature.com/articles/s41467-026-72551-0
- https://fireblog.boku.ac.at/wp-content/uploads/2024/09/eusalp_whitepaper_forestfires.pdf
- https://www.nwfirescience.org/sites/default/files/publications/wf24160.pdf
# Download manifest — open-access papers read for the expanded review
Generated 2026-08-20. The sandbox has no general internet, so PDFs could not be
downloaded here. Each entry below is open access; the URL resolves to the article
landing page from which the PDF can be saved into `papers/`.
# Evidence notes — expansion batch 1 (open-access, read 2026-08-20)

## gelabert2025_nhess — Assessing human-caused wildfire ignition likelihood across Europe
Gelabert, Jiménez-Ruano, Ochoa, Alcasena, Sjöström, Marrs, Ribeiro, Palaiologou, Bentué Martínez, Chuvieco, Vega-García, Rodrigues. NHESS 25:4713, 2025. doi:10.5194/nhess-25-4713-2025. OA.
- Five pilot sites incl. Central Europe (DE/CZ/PL); fires >1 ha, 1996–2022; Random Forest, 1000 pseudo-absence iterations.
- Site AUC 0.70–0.89 (Central Europe 0.89, Attica 0.70); pooled multi-site model AUC 0.81 but regional range 0.61–0.85, dropping 0.10 in Sweden and Attica.
- Yearly anomaly in dead fine fuel moisture content = most influential variable at ALL sites. Human factors ranked: distance to WUI > road proximity > population density > wildland fraction.
- Limits: overfitting where sample small; Euclidean distance fields prevent extrapolation beyond bounding box; uses STATIC human factors, recommends daily (commuting) and seasonal (tourism/recreation) dynamics.
- USE: transferability of ignition models; WUI as first-order human predictor; argues exposure is dynamic.

## ozaki2022_fire — Impact of Vertical Atmospheric Structure on an Atypical Fire in a Mountain Valley
Ozaki, Harris, Love, Aryal, Fox-Hughes, Williamson. Fire 5(4):104, 2022. doi:10.3390/fire5040104. OA.
- Riveaux Road Fire, southern Tasmania, ~637 km2, lightning-ignited Jan 2019, terrain to 1375 m.
- "Conducive wind structures" (wind+temperature layering producing enhanced surface winds): present in 6.51% of profiles in least-rugged terrain vs 40.15% in most-rugged terrain.
- Simulations using upper-air data scored 80% usefulness in gentle terrain, declining with terrain complexity; surface-only downscaled models showed the inverse.
- USE: hard number for terrain-dependence of fire weather; surface-only danger systems inadequate in complex terrain.

## hayajneh2026_fire — Wind and Slope Effects on Wildland Fire Spread: A Review
Hayajneh, Alzghoul, Naser. Fire 9(3):100, 2026. doi:10.3390/fire9030100. OA.
- Scoping review, >150 studies, 4 modelling traditions.
- Slope: radiative preheating dominates <=20 deg; mixed regime ~20 deg; convection-led >=30 deg. Bench study: ROS x2.5 from 20 to 30 deg. Eruption/flame attachment near 25–30 deg with confined geometry (aspect ratio >=0.2). Downslope non-monotonic: decreases to -20 deg, re-intensifies at -30 deg.
- Wind: ~1.5 m/s threshold for radiation->convection crossover. Pyrotron eucalypt litter ROS 33.6 -> 187.4 m/h; pine 155.7 -> 575.4 m/h. At 3.0 m/s and 0.8 kg/m2, ROS ~order of magnitude above still air. Discontinuous fuel regimes: no ignition <=1.0 m/s; direct contact 1.5–2.5 m/s; spotting >=3.0 m/s.
- Model failures: separable wind/slope correction factors miss nonlinear interaction; Rothermel-based operational models underestimate spread on steep slopes with wide fronts, cannot represent canyon blow-up, miss downslope re-intensification.
- USE: the quantitative core of the topography/fire-behaviour section; direct evidence that operational spread models are biased in exactly Alpine geometry.

## degiuli2024_diversity — Arrested Succession on Fire-Affected Slopes in the Krummholz Zone
De Giuli, Winkler, Deola, Henschel, Sass, Wolff, Jentsch. Diversity 16(7):366, 2024. doi:10.3390/d16070366. OA.
- Arnspitze massif (2196 m), Wetterstein, Northern Limestone Alps; 41 plots (26 burned/15 unburned) at 1800–1950 m; orthophotos 1974–2021.
- Burned plots richer: 35.1 +/- 8.8 vs 25.8 +/- 11.5 species/plot (p=0.05); forbs 64% vs 20% relative cover.
- Near-complete absence of tree seedlings; NO meaningful Pinus mugo re-establishment over 80 years post-fire (1946–2023).
- USE: decisive counterexample to the "3–4 year window" of post-fire susceptibility; at subalpine elevations the loss of protective cover can be effectively permanent.

## malandra2022_fire — Burn Severity Drivers in Italian Large Wildfires
Malandra, Vitali, Morresi, Garbarino, Foster, Stephens, Urbinati. Fire 5(6):180, 2022. doi:10.3390/fire5060180. OA.
- 113 large wildfires (>=100 ha), 2007 and 2017, ~102,565 ha, central and southern Italy; random forest.
- Land cover best predictor: conifer RdNBR 357 > shrubland 293 > mixed 288. Elevation second: higher severity at 0–650 m. Population density >3000 inhab/km2 associated with higher severity. Positive relation with distance to road and railway (remote = higher uncontrolled severity). Slope weak positive, 0–16 deg. TWI and heat load least important.
- USE: severity is fuel-type-led, not slope-led, at this scale; access/remoteness enters severity empirically — supports the suppression-constraint argument.

## hillebrand2023_ffgc — Contrasting impacts of climate change on protection forests of the Italian Alps
Hillebrand, Marzini, Crespi, Hiltner, Mina. Front. For. Glob. Change, 2023. doi:10.3389/ffgc.2023.1240235. OA.
- South Tyrol; ForClim on 11 sites along elevation gradient; CHELSA V2.1; SSP1-2.6 and SSP3-7.0 to 2100.
- Under severe scenario montane forests show drastic decline in avalanche protection index (to 0.03 by 2100); rockfall protection declines at lower elevation; subalpine forests resilient or improving. Dry inner Alpine valleys worst: drought-induced mortality with biomass reductions up to 97%.
- Limits: no herbaceous competition (regeneration possibly overestimated); monthly climate resolution may overestimate drought; insect and wind disturbance excluded.
- USE: elevation-contingent loss of protective function; the montane/subalpine contrast.

## moos2021_ffgc — Climate Change Impacts the Protective Effect of Forests: Switzerland
Moos, Guisan, Randin, Lischke. Front. For. Glob. Change 4:682923, 2021. doi:10.3389/ffgc.2021.682923. OA.
- Protection forest above Martigny, Valais, 500–1000 m; TreeMig + 3D rockfall simulation + quantitative risk analysis.
- Risk 19,100 CHF/yr with forest vs 95,500 CHF/yr without (5x). Rising to 59,250 (RCP4.5) and 70,100 CHF/yr (RCP8.5) within 150 yr. Composition shifts Fagus -> Quercus pubescens. Small blocks (0.1–0.5 m3) reaching railway once/decade now -> once/3 yr by 2150.
- USE: the monetised protective-function argument; the 5x factor is the single best number for "the forest is the mitigation measure".

## asteriou2025_geohazards — Assessing Post-Fire Rockfall Hazards: Evros, Greece
Asteriou, Sotiriadis, Petala, Kazelis. GeoHazards 6(3):54, 2025. doi:10.3390/geohazards6030054. OA.
- Avantas, Evros; August 2023 fire >90,000 ha; FiRo-RiS framework; RockyFor3D + RocFall2.
- Small blocks (0.01 m3): most stop on slope pre-fire, travel much further post-fire. Effect diminishes for larger blocks (0.125 m3) where kinetic energy overwhelms vegetation resistance.
- No timeline given for hazard reduction.
- USE: block-size dependence of the vegetation-loss effect — a mechanism detail the Alpine literature asserts but does not quantify.

## zacharakis2023_land — Environmental Forest Fire Danger Rating Systems and Indices around the Globe: A Review
Zacharakis & Tsihrintzis. Land 12(1):194, 2023. doi:10.3390/land12010194. OA.
- 63 systems/indices across 16 countries. Where FWI is used outside Canada, danger classes are redefined by local calibration.
- Few systems handle terrain: NFDRS uses three slope classes; LASI uses pressure levels by elevation.
- USE: independent corroboration that index transfer requires recalibration, and that terrain is largely absent from operational danger systems worldwide.

## venalainen2014_nhess — Temporal variations and change in forest fire danger in Europe 1960–2012
Venäläinen, Korhonen, Hyvärinen, Koutsias, Xystrakis, Urbieta, Moreno. NHESS 14:1477, 2014. doi:10.5194/nhess-14-1477-2014. OA.
- No obvious trend 1960–1999 (ERA-40); mean FWI increasing for southern and eastern Europe 1980–2012 (ERA-Interim), significant at 99%.
- Cross-correlation burned area vs fire-season weather indices ~0.6 in Greece, Spain, Finland.
- USE: baseline for "danger trend is detectable but weather explains only part of burned area".

## frei2023_forests — Evidence for 40 Years of Treeline Shift in a Central Alpine Valley
Frei, Barbeito, Erdle, Leibold, Bebi. Forests 14(2):412, 2023. doi:10.3390/f14020412. OA.
- Dischma valley, Davos, 1600–2800 m; surveys 1972/73 vs 2012.
- Treeline advanced on average 10 m/decade, max local 42 m/decade. 21% of new trees are advance, 79% infilling below the previous treeline. Larch +47 m on south-facing, pine +36 m on north-facing.
- Drivers: warming above 2330 m; aspect, curvature, snow disappearance timing; dwarf-shrub competition; 1900 land use at lower elevations; current grazing no significant effect; avalanche causes local downward shift.
- USE: quantifies the fuel-moving-upslope claim, and the 79% infilling figure is the fuel-continuity argument in numbers.

## bercak2024_fire — A Combination of Human Activity and Climate Drives Forest Fire Occurrence in Central Europe (Czech Republic)
Berčák, Holuša, Trombik, Resnerová, Hlásny. Fire 7(4):109, 2024. doi:10.3390/fire7040109. OA.
- Czech Republic 2006–2015. Annual scale: human activity dominates, climate not significant, model explains 71.4% of variability. Monthly/summer scale: climate significant, 48.7–53.8% explained.
- Bimodal seasonality: spring peak April (median burned area 0.04 ha), summer peak Jul–Aug (median 0.005 ha). Spring linked to forestry operations after winter; summer to heat, drought and tourism, with threshold response above 18 degC.
- USE: independent, non-Alpine-arc corroboration of the bimodal central European regime — removes the reliance on Valese alone.

## forrest2024_bg — BASE (Burnt Area Simulator for Europe)
Forrest, Hetzer, Billing, Bowring, Kosczor, Oberhagemann, Perkins, Warren, Arrogante-Funes, Thonicke, Hickler. Biogeosciences 21:5539, 2024. doi:10.5194/bg-21-5539-2024. OA.
- GLMs at ~9x9 km monthly. Europe 2001–2020 burnt area ~1.46 Mha/yr: cropland 1.09 Mha/yr (74%), non-cropland vegetation 0.38 Mha/yr (26%).
- Explains 32% (NCV) and 36% (cropland) of deviance; underestimates peak burning.
- Explicitly limited skill in mountainous and temperate regions.
- USE: continental fire models self-report failure in mountains — direct support for the scope-bias argument.

## ruffault2018_nhess — Extreme wildfire events linked to global-change-type droughts in the northern Mediterranean
Ruffault, Curt, Martin-StPaul, Moron, Trigo. NHESS 18:847, 2018. doi:10.5194/nhess-18-847-2018. OA.
- Southern France (PACA), JJAS 1996–2016. Six extreme events in 2003, three in 2016.
- Two drought types: "press drought" (2016, long water deficit, wind-driven fires) and "hot drought" (2003, water deficit plus heat wave, heat-induced fires).
- USE: the drought typology distinction, useful for separating Alpine spring/summer mechanisms.

## pais2023_fire — Optimizing Wildfire Prevention through Integration of Prescribed Burning into Fire-Smart Land-Use Policies
Pais, Aquilué, Honrado, Fernandes, Regos. Fire 6(12):457, 2023. doi:10.3390/fire6120457. OA.
- Prescribed fire alone reduced future wildfire by up to 36% over 30 years; leverage 0.07–1.78. Strategic allocation to high-risk zones: leverage 0.32–0.45. Combined with High Nature Value farmland + fire-smart conversion: leverage 1.78 (1 ha wildfire avoided per ha treated).
- Advocates EU-CAP support for agropastoral activity.
- USE: quantifies the grazing/land-use lever that Rouet-Leduc argues for qualitatively, and gives prescribed fire a benchmark the Alps can be compared against.

## senf2021_natsust — Mapping the forest disturbance regimes of Europe
Senf & Seidl. Nature Sustainability, 2021. doi:10.1038/s41893-020-00609-y. OA.
- Landsat 1986–2016, LandTrendr. 17% of Europe's forest area disturbed over 30 years; 36 million patches, mean patch size 1.09 ha, 0.52 patches/km2/yr. Disturbance frequency increasing, severity decreasing.
- Alps conifer and mixed forests: substantially higher disturbance frequency in Austria than Italy.
- USE: the disturbance baseline against which fire must be sized; mean patch 1.09 ha frames why Alpine fires are small.

## cardil2025_ijwf — Assessing the suppression difficulty of wildland fires for initial attack response
Cardil et al. Int. J. Wildland Fire 34:WF24160, 2025. doi:10.1071/WF24160.
- California 2020–2023, 26,907 fires; validation in Oregon, Colorado, Montana and Spain.
- Terrain Difficulty Index = accessibility + penetrability + construction indices + average slope + road proximity; combined with Fire Behaviour Index into Initial Attack Assessment (1–5).
- IAA-5 vs IAA-1: 90% decrease in odds of initial attack success. Mean burned area 18.5 ha (IAA-1) to 1731 ha (IAA-5). Response time interacts with TDI to reduce containment probability. Overall 97.02% of fires contained at <=4 ha.
- Reported adoption beyond California incl. British Columbia and European agencies.
- USE: the framework for terrain-conditioned suppression exists and is transferable — so the Alpine gap is application, not method. Revises the earlier "no evidence base" claim.

## stritih2023_landecol — Alternative states in the structure of mountain forests across the Alps
Stritih, Seidl, Senf. Landscape Ecology, 2023. doi:10.1007/s10980-023-01597-y.
- Entire Alpine Convention area, ~10.5 Mha forest.
- Two structural states: closed canopy 76% of area (cover >~0.75, height percentile >~10 m), open canopy 24% (high elevation, forest edges, warm dry sites).
- Disturbance initiated transition to open canopy in ~50% of cases; 72% of Alpine forests returned to closed canopy within 35 years; submediterranean forests much slower, staying open for decades.
- USE: quantifies recovery of protective structure at Alpine scale, and the 35-year figure bridges the 3–4 year soil window and the 80-year krummholz result.

## laporta2025_landslides — Post-wildfire debris flow in the NW Italian Alps: the June 2018 Bussoleno event
La Porta, Leonardi, La Ferlita, Pirulli. Landslides 22(11):3625, 2025. doi:10.1007/s10346-025-02605-9.
- Comba delle Foglie basin, 1.37 km2, 490–1790 m, Susa Valley. October 2017 fire burned >95% of catchment; spring 2018 rainfall 40% above normal; debris flow 7 June 2018, ~8 months post-fire.
- ~20,000 m3 mobilised; two residential buildings destroyed, ~150 residents flooded from the alluvial fan.
- Trigger: radar-estimated 12 mm in 30 min = 24 mm/h, 5-year return period — BELOW the >30 mm/h threshold typical for unburned basins of this type.
- Mechanical shear strength of burned soils reduced by a factor of three. RASH3D depth-averaged model; Bingham rheology best (yield stress 1300 Pa, viscosity 80 Pa s). Susceptibility-based triggering used 84,000 m3 of identified unstable soil.
- USE: the single best-quantified Alpine post-fire cascade event; the sub-threshold triggering is the key risk-transfer argument.
# Evidence notes — expansion batch 2 (open-access, read 2026-08-20)

## muller2020_eusalp — Forest Fires in the Alps: State of Knowledge, Future Challenges and Options for an Integrated Fire Management
Müller M.M., Vilà-Vilardell L., Vacik H. (2020), EUSALP Action Group 8 white paper. OA PDF.
- Annual averages 2006–2017 (fires/yr; ha/yr): Italy 1043 / 9984; France 213 / 818; Austria 122 / 64; Switzerland 105 / 515; Slovenia 37 / 201.
- Burnt area per 10,000 ha forest per year: Italy 44.13; France 5.80; Switzerland 5.19; Slovenia 2.77; Austria 0.22. Two orders of magnitude spread across the arc.
- ~80% of Alpine fires burn <1 ha. Median fire size <0.1 ha (AT, DE, CH) to 1 ha (FR, IT).
- ~90% of Alpine fires human-caused (cigarettes, escaped fires, sparks from trains/work, arson, hot ashes, power lines); lightning ~10% regionally but up to 50% in summer in some areas.
- Two seasons: spring peak Mar–Apr, summer peak Jul–Aug, plus winter fires in foehn-influenced regions.
- Direct firefighting + post-fire management cost ~EUR 75 million/yr across the Alpine region.
- Proposed integrated fire management framework ~EUR 10 M/yr: prevention 5–7, suppression 1.5–2.5, post-fire 1–2, knowledge transfer 0.5.
- States firefighting is generally difficult in the Alps due to rugged topography and low accessibility; coniferous forests on southern slopes providing protection against avalanche, rockfall and mudslide are disproportionately at risk; fires create new avalanche-prone slopes, higher rockfall risk, mudslides, erosion and local hydraulic regime change.
- Concludes current efforts "are unable to prevent the occurrence of extreme forest fire events".
- USE: the single best source of hard Alpine baseline statistics and of the cost argument. Replaces several places where Valese or Müller-2020-EcoInf were being reused.

## beloiu2026_natcomm — Climate-driven upward spread of forest fires in European mountain regions
Beloiu et al. Nature Communications, 2026. doi:10.1038/s41467-026-72551-0. OA.
- Eight European mountain regions incl. the Alps; 2000–2025; fires >=800 m and >=30 ha.
- Fires spread upward ~72 m per decade. Median fire elevation 910 m (2000–2014) -> 1004 m (2015–2025). 90th percentile of annual fire elevations 1353 -> 1460 m; top-ten annual fires 1451 -> 1777 m.
- After 2015, fire occurrence increasing 30%/yr below 1400 m and 12%/yr above.
- VPD strongest predictor of burned area; fires above 1400 m linked to extreme atmospheric aridity.
- Limitation the authors state: only fires >30 ha, and between 1980 and 2010 the vast majority of Alpine wildfires were smaller than 10 ha — so emerging high-elevation activity is likely underestimated.
- USE: replaces palaeo-inference as the primary evidence for upslope migration; the >30 ha threshold caveat is itself an argument about Alpine observability.

## mutterer2026_fireecol — Bark traits and thermal resistance to wildfires: six Central European tree species
Mutterer, Sheppard, Kohnle, Seifert. Fire Ecology, 2026. doi:10.1186/s42408-025-00438-4. OA.
- 129 stem discs heated at 212.3 +/- 5.2 degC; time for cambium to reach the lethal 60 degC (tau60).
- Bark thickness is the most robust predictor of insulation. Resistance ranking: Larix decidua, Pseudotsuga menziesii, Pinus sylvestris (higher) > Abies alba, Picea abies, Fagus sylvatica (lower, beech lowest).
- Thermal resistance decreases with bark moisture content and with height on the stem; bark density not significant.
- USE: experimental corroboration of Dupire's trait-based vulnerability ranking, from an independent method. Removes reliance on Dupire alone for the species-vulnerability claim.

## diakakis2023_land — Debris flow likelihood regression model in Mediterranean post-fire environments
Diakakis, Mavroulis, Vassilakis, Chalvatzi. Land 12(3):555, 2023. doi:10.3390/land12030555. OA.
- Schinos-Alepochori, Attica, after the May 2021 fire; 21 basins.
- Adapts the Staley et al. logistic model developed for the western USA: predictors are proportion of high/moderate soil burn severity on gradients >=23 deg, dNBR, and soil erodibility K-factor, multiplied by peak 15-minute rainfall.
- Probabilities 0.05–0.893; correlation with observed debris flows significant at 0.001; ROC AUC 0.913.
- Authors report high transferability, noting it runs on European soil datasets.
- USE: important nuance — physically grounded post-fire debris-flow models DO transfer from the US to Europe, unlike weather-only fire danger indices. Sharpens the transferability argument.

## avouris2025_egusphere — Exposure of Settlements to Wildfires in a Transboundary WUI Region in Central Europe (PREPRINT)
Avouris, Marrs, Beetz, Kudláčková, Poděbradská, Trnka, Forkel. EGUsphere preprint, 2025. doi:10.5194/egusphere-2025-4859.
- Saxon-Bohemian Switzerland, DE/CZ border, ~700 km2; FlamMap with 3 fire durations and 3 wind directions under medium to very high danger; validated on the 2022 fire.
- Higher burn probability and potential flame length at settlements on the Czech side; wildfires disregard political boundaries.
- Caveats from review: standard methods, validation dependent on remote sensing commission/omission errors, weather coverage favours the German side.
- STATUS: preprint — cite as such or replace with the published version.
- USE: the transboundary exposure argument, with a Central European rather than Mediterranean example.

## cardil2025_ijwf — see batch1. Terrain Difficulty Index / Initial Attack Assessment.

## keeping2026_egusphere — Attribution of the record-breaking 2025 European fire season (PREPRINT)
Keeping, Zachariah, Haas, Grillakis, Barnes, Clay, Ekberzade, Jaupaj, Ribeiro, Trigo, Voulgarakis, Otto. EGUsphere preprint, 2026. doi:10.5194/egusphere-2026-1193.
- Record burned area >1,400,000 ha in 2025. Strong trends to drier summers and extreme weekly VPD; summer drought emergent from natural variability in most southern European regions, VPD extremes emergent in all five regions studied.
- Five regions: NW Iberia, N/W Britain, Occitania, eastern Adriatic/Ionian, N/W Türkiye. Explicitly EXCLUDES the Alps and central Europe.
- STATUS: preprint.
- USE: use the exclusion as evidence — the flagship European attribution study of the record season does not cover the Alps.

## baranovskiy2021_forecasting — Deterministic-probabilistic prediction of lightning-caused forest fires in mountain areas
Baranovskiy. Forecasting 3(4):695, 2021. doi:10.3390/forecast3040043. OA.
- Republic of Altay and Buryatia, Russia. Five probability factors incl. an explicit altitude term; deterministic unsteady heat-transfer ignition models for deciduous vs coniferous wood.
- Mountain-specific features: cloud-to-ground discharge frequency increases with elevation; altitudinal forest belts change fuel; precipitation varies strongly across mountain zones.
- No validation accuracy reported (synthetic discharge data).
- USE: mechanism for elevation-dependence of lightning ignition; weak on validation, so use for structure not for numbers.

## Manifest of landing pages for PDF download into papers/
- https://nhess.copernicus.org/articles/25/4713/2025/
- https://www.mdpi.com/2571-6255/5/4/104
- https://www.mdpi.com/2571-6255/9/3/100
- https://www.mdpi.com/1424-2818/16/7/366
- https://www.mdpi.com/2571-6255/5/6/180
- https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2023.1240235/full
- https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2021.682923/full
- https://www.mdpi.com/2624-795X/6/3/54
- https://www.mdpi.com/2073-445X/12/1/194
- https://nhess.copernicus.org/articles/14/1477/2014/
- https://www.mdpi.com/1999-4907/14/2/412
- https://www.mdpi.com/2571-6255/7/4/109
- https://bg.copernicus.org/articles/21/5539/2024/
- https://nhess.copernicus.org/articles/18/847/2018/
- https://www.mdpi.com/2571-6255/6/12/457
- https://www.nature.com/articles/s41893-020-00609-y
- https://link.springer.com/article/10.1007/s10980-023-01597-y
- https://link.springer.com/article/10.1007/s10346-025-02605-9
- https://link.springer.com/article/10.1186/s42408-025-00438-4
- https://www.mdpi.com/2073-445X/12/3/555
- https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4859/
- https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1193/
- https://www.mdpi.com/2571-9394/3/4/43
- https://www.nature.com/articles/s41467-026-72551-0
- https://fireblog.boku.ac.at/wp-content/uploads/2024/09/eusalp_whitepaper_forestfires.pdf
- https://www.nwfirescience.org/sites/default/files/publications/wf24160.pdf
# Evidence notes — expansion batch 3: forestry, drought/water, suppression, economics (2026-08-20)

## accastello2019_forests — ASFORESEE: harmonized model for economic evaluation of forest protection against rockfall
Accastello, Bianchi, Blanc, Brun. Forests 10(7):578, 2019. doi:10.3390/f10070578. OA.
- Alpine Space FORest Ecosystem Services Economic Evaluation. Replacement-cost approach: values protection by the cost of the artificial structures that would be needed instead (rockfall nets designed per ETAG027), net of the forest management expenditure required to maintain protective capacity. Uses a Rockfall Protection Index for forest effectiveness.
- Case study, 10 ha Scots pine stand, Italian Alps: total stand value EUR 304,380; EUR 30,440 per ha; EUR 950 per ha per year over 25 years. Forest sufficient to meet local safety requirement without artificial measures.
- USE: the per-hectare monetisation of protective function. Complements Moos (risk avoided) with replacement cost.

## jandl2018_forests — Strategies for climate-smart forest management in Austria
Jandl, Ledermann, Kindermann, Freudenschuss, Gschwantner, Weiss. Forests 9(10):592, 2018. doi:10.3390/f9100592. OA.
- Caldis growth model on 11,000 NFI plots + Yasso07 soil carbon, 2010–2100, RCP8.5 (+3.5 degC).
- Stem biomass change by 2100: reference -9%, bioenergy -17%, domestic wood products -29%, import-assisted -25%, reduced harvest +34%.
- Conclusion: long-lived wood products are the preferred climate-smart implementation, balancing sequestration, economics and disturbance risk.
- USE: the carbon-vs-risk trade-off in Alpine forestry, quantified; shows reduced-harvest maximises carbon but builds standing biomass.

## seben2024_forests — Impact of salvage logging on deadwood decomposition and forest regeneration, Tatra NP
Seben, Pajtik, Konopka. Forests 15(11):1936, 2024. doi:10.3390/f15111936. OA.
- High Tatra, windstorm Alzbeta Nov 2004, assessed 18 years later.
- Deadwood retained: 0.35 m3/are salvaged vs 2.94 m3/are unsalvaged (8x). Decay rate higher where deadwood contacts soil (3.83 vs 3.10).
- Regeneration diversity higher on salvaged sites (5.40 vs 3.85 species/plot); unsalvaged dominated by Norway spruce (66.9% of trees).
- Authors caution against the prevailing view that salvage logging is exclusively negative; argue for locally tailored rather than categorical decisions.
- USE: complicates the standard anti-salvage position; relevant to post-fire salvage debate in protection forests.

## costa2021_forests — Biological legacies and rockfall: the protective effect of a windthrown forest
Costa, Marchi, Bettella, Bolzon, Berger, Lingua. Forests 12(9):1141, 2021. doi:10.3390/f12091141. OA.
- Monte Pore, Belluno, Eastern Italian Alps (Dolomites). 3000 rockfall simulations: no forest / pre-windstorm forest / post-windstorm with deadwood.
- Barrier effect index improved after the windstorm; the windthrown stand was MORE protective than the standing forest. Lying logs and stumps act as physical barriers rather than energy absorbers.
- Lying logs may provide barrier effect against snow and rock for up to 30 years post-disturbance, declining as logs decay and contact soil.
- Recommend avoiding salvage logging to maintain multifunctional protection during recovery.
- USE: KEY CONTRAST. Windthrow leaves protective biological legacies; fire consumes or weakens them. This is a strong, previously unmade argument for why fire is a distinct disturbance in protection forests.

## gamba2025_screp — Prescribed burning has negligible effects on the plant-soil system in Pinus sylvestris forests of the European Alps
Gamba et al. Scientific Reports 15:12076, 2025. doi:10.1038/s41598-025-97239-1. OA.
- SW European Alps, 45.0569 N 6.7853 E, 1600 m, south-facing, 7.0 degC and 645 mm/yr. 18 circular plots (12 treated, 6 control) in 1 ha, spring 2022; measured pre-burn, immediately post, 6 and 12 months.
- Fuel load reduced ~51%. No significant change in branch hydraulic conductivity or stem water potential; no heat-induced wood anatomy deformation; minimal change in soil organic carbon and water repellency, despite severe drought.
- Concludes prescribed burning is an ecologically sustainable fuel-reduction option for Alpine Scots pine.
- USE: CORRECTS AN ERROR IN v2/v3, which asserted that no Alpine prescribed-burning trial exists. It does. The claim must be rewritten.

## brodie2024_fireecol — Forest thinning and prescribed burning treatments reduce wildfire severity and buffer severe fire weather
Brodie, Knapp, Brooks, Drury, Ritchie. Fire Ecology 20, 2024. doi:10.1186/s42408-023-00241-z. OA.
- Goosenest Adaptive Management Area, Klamath NF, NE California, 1460–1770 m. Randomised replicated experiment that subsequently burned in a wildfire.
- Thin + prescribed fire vs control: tree mortality probability 0.60 lower (95% HDCI 0.51–0.69); bole char height 7.38 m lower; crown volume consumed 86% lower; crown volume affected 68% lower.
- Treatment benefit for mortality maximised at intermediate burning-index values and declined under the most severe fire weather, while crown-fire metrics stayed responsive at high burning index.
- USE: the effectiveness-vs-fire-weather interaction. Fuel treatment buys most where danger is moderate, which matters for an Alps whose danger distribution is shifting.

## goncalves2025_fire — Fostering post-fire research towards a more balanced wildfire science agenda
Goncalves, Portela, Regos, Sil, Marcos, Alonso, Honrado. Fire 8(2):51, 2025. doi:10.3390/fire8020051. OA.
- Bibliometric: post-fire stage ~3976 papers vs 5802 for prevention ("before") and 7041 for active fire. Post-fire research is 46% less prevalent than pre-fire.
- Recommends long-term funding for recovery, standardised burn-severity and socioecological impact protocols, transdisciplinary work across fire ecology, hydrology, soil and social science.
- USE: quantifies the structural imbalance in the discipline that this review's cascade argument runs against.

## lecinadiaz2021_ecosystems — Assessing the risk of losing forest ecosystem services due to wildfires
Lecina-Diaz, Martinez-Vilalta, Alvarez, Vayreda, Retana. Ecosystems, 2021. doi:10.1007/s10021-021-00611-1.
- Catalonia. Risk = exposed value x hazard x susceptibility x lack of adaptive capacity, for carbon storage, bird biodiversity, water regulation and erosion prevention; median vs 90th percentile FWI.
- Under average conditions hazard magnitude dominates risk; under extreme conditions exposed values (carbon, erosion control) dominate. Wet forests disproportionately vulnerable despite currently low risk. Non-Mediterranean conifer forests with limited adaptive capacity are most vulnerable under extreme fire weather.
- USE: directly describes the Alpine situation from outside it; and the average/extreme switch in what drives risk is analytically useful.

## michetti2019_ere — Forest fires across Italian regions and implications for climate change: a panel data analysis
Michetti & Pinar. Environmental and Resource Economics 72:207, 2019. doi:10.1007/s10640-018-0279-z.
- 19 Italian regions, monthly 2000–2011, four geographic clusters.
- +10 mm monthly precipitation reduces fire frequency by 8.9% same month, 7.8% next, 3.4% at two months; effects persist longer in the north. +1 degC raises frequency 15% in the south, 0.9–5.7% elsewhere.
- +1 percentage point tertiary education reduces fire events 11% (north), 7% (central-north), 2.2% (central-south). In southern Italy one additional extortion per 100,000 inhabitants raises fire frequency 4.6% and burnt area 7%.
- Projection to 2016–2035 under A1B: north +11–12% events and +23–27% burnt area; central-south +25% events and +75% burnt area.
- Policy conclusion: investing upstream in prevention is more efficient than suppression.
- USE: the economics section's backbone; northern-cluster numbers are the closest available to the Italian Alps, and the education and crime coefficients give the socio-economic dimension real content.

## leone2026_hess — The 2022-2023 snow drought in the Italian Alps doubled glacier contribution to summer streamflow
Leone, Avanzi, Morra di Cella, Gabellani, Cremonese, Isabellon, Pogliotti, Scotti, Monti, Ferraris, Colombo. HESS 30:4649, 2026. doi:10.5194/hess-30-4649-2026. OA.
- Dora Baltea (Aosta Valley, 3304 km2) and Adda (Lombardy, 2344 km2), 2011–2023.
- Peak glacier contribution to streamflow: Tavagnasco 75% (2022) and 65% (2023) vs median 31.5%; Fuentes 83% (2022) and 61% (2023) vs median 22.6%. Mean annual glacier melt ~148% above the 2011–2021 average.
- Streamflow nonetheless remained below long-term average: glaciers cannot fully compensate precipitation deficit.
- USE: Alpine water availability quantified, and the buffer-of-last-resort role of glaciers. Connects to the fire-soot-on-ice pathway (Aubry-Wake) as a compounding risk.

## basso2021_fenvs — Potential post-fire impacts on a water supply reservoir
Basso, Mateus, Ramos, Vieira. Front. Environ. Sci. 9:684703, 2021. doi:10.3389/fenvs.2021.684703. OA.
- Zezere watershed to Castelo de Bode reservoir, central Portugal, supplying ~2 million people in Lisbon; scenario burning ~30% of the contributing area.
- Median total suspended solids up 4.7x; nitrate maximum up 4x (0.30 to 1.49 mg N-NO3/L); phosphate maximum up 4x. Dilution at the dam wall removed ~50% of nitrate and ~90% of phosphate relative to inlet.
- Reservoir storage attenuated the signal enough to avoid breaching drinking-water standards, but the authors warn dilution capacity falls during drought.
- USE: the downstream water-supply consequence, with the important nuance that reservoir size buffers it — and that drought removes the buffer. Comparative (Portugal), not Alpine.

## giuggiola2010_iforest — Drought-induced mortality of Scots pine at its southern limits
Giuggiola, Kuster, Saha. iForest 3:95, 2010. doi:10.3832/ifor0542-003. OA.
- Mediterranean and dry inner-Alpine valleys. Mechanism: carbon starvation from reduced uptake under drought, raising susceptibility to pests and pathogens.
- Management: thinning and shrub removal reduce competition for water and improve carbon acquisition.
- No quantified mortality rates or elevation thresholds.
- USE: links drought stress to the silvicultural lever; the thinning recommendation is a fuel treatment with a second justification.

## gomezgallego2022_pce — Interaction of drought- and pathogen-induced mortality in Norway spruce and Scots pine
Gomez-Gallego et al. Plant, Cell & Environment 45, 2022. doi:10.1111/pce.14360. OA.
- Sapling experiment, Sweden. Spruce: drought plus Heterobasidion infection accelerates death (p=0.059); 90% of accumulated mortality after dormancy; surviving inoculated saplings held 31% reaction zone vs 3% in dead ones; needle starch predicted mortality. Pine showed no interaction; 75% of inoculated deaths within three months.
- Mechanism: competing carbon demands between compartmentalising the pathogen and maintaining turgor, plus reduced functional sapwood.
- USE: mechanism for the drought-pathogen-fuel chain. Experimental and non-Alpine; use for process, not magnitude.

## misseyanni2025_forests — The impact of forest fires on ecosystem services: the case of Greece
Misseyanni, Christopoulou, Kougkoulos, Vassilakis, Arianoutsou. Forests 16(3):533, 2025. doi:10.3390/f16030533. OA.
- Systematic review, 56 publications 1997–2024, CICES v5.1, short (1–2 yr), medium (3–20 yr) and long (>20 yr) term.
- Provisioning services 80% negative entries; cultural services all negative; erosion control all short-term entries negative; hydrologic cycle regulation all entries negative; climate regulation predominantly negative. Positive short-term effects for plant biodiversity and pollination.
- Most pronounced effects in the first two years post-fire.
- USE: the service-by-service and time-resolved picture, and independent corroboration that hydrological regulation is uniformly degraded.

## Manifest additions (landing pages for PDF download)
- https://www.mdpi.com/1999-4907/10/7/578
- https://www.mdpi.com/1999-4907/9/10/592
- https://www.mdpi.com/1999-4907/15/11/1936
- https://www.mdpi.com/1999-4907/12/9/1141
- https://www.nature.com/articles/s41598-025-97239-1
- https://link.springer.com/article/10.1186/s42408-023-00241-z
- https://www.mdpi.com/2571-6255/8/2/51
- https://link.springer.com/article/10.1007/s10021-021-00611-1
- https://link.springer.com/article/10.1007/s10640-018-0279-z
- https://hess.copernicus.org/articles/30/4649/2026/
- https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2021.684703/full
- https://iforest.sisef.org/abstract/?id=ifor0542-003
- https://onlinelibrary.wiley.com/doi/10.1111/pce.14360
- https://www.mdpi.com/1999-4907/16/3/533
# Evidence notes — batch 4: Jandl MANFRED + FIRE-RES outputs (2026-08-20)

## jandl_management_2013 — Management strategies to adapt Alpine Space forests to climate change risks: an introduction to the MANFRED project
Jandl, Cerbu, Hanewinkel, Berger, Gerosa, Schueler. In: Management Strategies to Adapt Alpine Space Forests to Climate Change Risks, InTech, 2013. doi:10.5772/56267. Open access (CC-BY). SUPPLIED BY AUTHOR AS PDF.
- Interreg Alpine Space project MANFRED: 15 partners, 6 countries, 4 transnational case study regions; partners include BFW, FVA, WSL, Cemagref/IRSTEA, Slovenian Forest Institute, Regione Lombardia, Valle d'Aosta, Bavarian LWF, BAFU.
- Expected warming at least 3-4 degC over the next century; snow cover duration expected to decrease, water discharge regime altered, rockfall frequency may increase, species distributions change, pests and pathogens shift.
- KEY: "Due to the history of active fire suppression the fire risk does not receive the full attention in the Alps. Nevertheless, future climate conditions may well call for an improved alerting system for the prevention of devastating forest fires." A 2013 statement of the institutional blind spot from a major Alpine forestry consortium.
- Protective function described as "by far the most important ecosystem service of mountain forests"; eco-engineering invoked when it is jeopardised; climate change makes forests more vulnerable to rockfall, avalanche and landslide.
- Storm damage is not purely climate-driven: "the practised form of forest management such as the maturing of increasingly dense forests has created types of forest stands that are quite vulnerable to storm damages". Management history is itself a disturbance driver.
- Genetic adaptive capacity: provenance trials evaluated for Picea abies; potential productivity greatly increased by choosing appropriate provenance; reduces risk over the stand lifetime.
- Growth model coupled to a soil carbon model to extend evaluation from timber to sequestration.
- Adaptive management concepts are deliberately conservative given climate uncertainty: increase stand stability by distributing risk among tree species and age cohorts.
- BINDING CONSTRAINT: "the suggested concepts only can be successfully implemented when damages of forests due to high population densities of ungulates are kept at an acceptable level. Presently, damages caused by deer are inhibiting the establishment of species-rich forest ecosystems."
- Also produced an online database of documented extreme events in the Alpine Space, and a transnational web-based pest/pathogen alerting system.
- USE: anchors the whole forest-management section in an Alpine transnational programme; supplies the 2013 baseline against which the current fire-attention deficit can be measured; supplies the ungulate constraint on adaptive silviculture; supplies the management-history-as-disturbance-driver point.

## stoof_temperate_2024 — In temperate Europe, fire is already here: the case of The Netherlands
Stoof, Kok, Cardil Forradellas, van Marle. Ambio 53(4):604-623, 2024. doi:10.1007/s13280-023-01960-y. FIRE-RES. OA.
- 2017-2022: 3,667 landscape fires, mean 611/yr (range 212-949), ~405 ha/yr, mean fire size 1.5 ha, 88% under 1 ha. Spring (Apr-Jun) and summer drought peaks; 69% daytime; overwhelmingly human-caused (only 10 lightning fires). Heathland and peatland most affected, not forest.
- CRITICAL: EFFIS satellite data captured only 0.2% of Dutch wildfires and 45% of burned area.
- Costs: suppression and restoration ~EUR 2.7 M/yr; highway disruption ~EUR 297,000/yr.
- Policy: a legal framework is needed to guarantee continuity of recordkeeping as the foundation of integrated fire management; Denmark, Luxembourg and Germany lack systematic wildfire statistics; fire services are specialised in urban firefighting, not landscape fire; reliable statistics are needed as a climate baseline and for EU Forest Strategy and UNFCCC reporting.
- USE: the strongest available parallel to the Alpine case, and quantified. Small fires, spring peak, human ignition, absent statistics, urban-specialised services. The 0.2%/45% detection figures are decisive for the observational-threshold argument in the modelling section.

## rodrigues_vpdbased_2024 — VPD-based models of dead fine fuel moisture provide best estimates in a global dataset
Rodrigues, Resco de Dios, Sil, Cunill Camprubi, Fernandes. Agricultural and Forest Meteorology 346:109868, 2024. doi:10.1016/j.agrformet.2023.109868. FIRE-RES.
- BONFIRE database: 1,603 records of 1-h dead fuel moisture from 153 experiments in 21 countries, 1970-2020, all biomes except tundra and tropical rainforest.
- Full moisture range 4-35%: semi-mechanistic VPD model MAE 4.06%, R2 0.40; best alternatives SVMveg MAE 3.89% R2 0.44 and LMMRH MAE 3.85% R2 0.43 - marginal gains.
- Below 10% moisture, all approaches deteriorate: VPD model MAE 4.23%, R2 0.11; MK5 lowest MAE 2.36% but R2 0.11.
- Authors caution against machine-learning complexity: gains in fit are small relative to computational cost. They argue prediction at the dry end should be the research priority because spread responds exponentially to dryness, with 1% moisture changes producing substantial behavioural effects.
- USE: strong independent support for the parsimony argument, and it quantifies where the models fail - precisely in the moisture range that produces fire.

## plana_framing_2024 — Framing coherence across EU policies towards integrated wildfire risk management and nature-based solutions
Plana, Serra, Smeenk, Regos, Berchtold, Huertas, Fuentes, Trasobares, Vinders, Colaco, Bonet. Fire 7(11):415, 2024. doi:10.3390/fire7110415. FIRE-RES. OA.
- 40 EU sectoral policies in 7 clusters analysed.
- Integrated Fire Management appears 7 times across the whole corpus, with a unique mention in only one policy text. "Extreme wildfire event" appears 4 times in total. The beneficial role of fire is missing or not explicitly mentioned. Nature-based solutions appear minimally in forestry and disaster-risk policy despite prominence in climate and biodiversity frameworks.
- Recommends formal ecosystem-service classification of wildfire protection and its inclusion in natural capital accounting; fire ecology in biodiversity and restoration policy; fire-smart certification for bioeconomy and agriculture; whole-of-government coordination; mainstreaming IFM terminology.
- USE: quantifies EU-level policy fragmentation, which is the supranational counterpart to the Alpine institutional fragmentation already documented.

## lecinadiaz_incorporating_2023 — Incorporating fire-smartness into agricultural policies reduces suppression costs and ecosystem services damages from wildfires
Lecina-Diaz, Chas-Amil, Aquilue, Sil, Brotons, Regos, Touza. Journal of Environmental Management 337:117707, 2023. doi:10.1016/j.jenvman.2023.117707. FIRE-RES.
- Transboundary Biosphere Reserve Geres-Xures (Spain-Portugal), 2,760 km2. REMAINS process-based landscape fire model; cost-plus-net-value-change analysis, 2010-2050, four land-use scenarios.
- Suppression cost present value: High Nature Value farmland -3,922 kEUR (saving); HNVf + fire-smart -3,959 kEUR (saving); fire-smart alone +2,308 kEUR; business as usual +1,149 kEUR.
- Total net cost (suppression plus ecosystem service damage): HNVf + fire-smart 7,132 kEUR; business as usual 25,710 kEUR.
- USE: the quantified prevention-versus-suppression economics the review needed. Note the structure of the result: fire-smart forestry ALONE increases suppression cost; it is the agricultural-landscape component that delivers the saving. That is a nuance worth stating.

## kalapodis_towards_2025 — Towards resilient critical infrastructure in the face of extreme wildfire events
Kalapodis, Sakkas, Kazantzidou-Firtinidou, Alcasena, Cardarilli, Eftychidis and others. Infrastructures 10(9):246, 2025. doi:10.3390/infrastructures10090246. FIRE-RES. OA.
- Critical infrastructure exposed: energy transmission and distribution, water supply, telecommunications, transport (road, rail, bridges), health care, financial services.
- EU operates under the CER Directive; the US under PPD-21 moving to NSM-22 and NIRMP. Southern European countries account for nearly 80% of total EU burned area.
- Diagnoses insufficient integration of scientific knowledge into policy formulation, and governance fragmentation; recommends standardised risk assessment and cross-border, multi-stakeholder coordination.
- USE: places the Alpine transport-corridor exposure argument in an EU critical-infrastructure policy frame.

## cardil_climate_2023 — Climate teleconnections modulate global burned area
Cardil, Rodrigues, Tapia, Barbero, Ramirez, Stoof, Silva, Mohan, de-Miguel. Nature Communications 14, 2023. doi:10.1038/s41467-023-36052-8. FIRE-RES. OA.
- 13 teleconnections analysed. Collectively they modulate 52.9% of global burned area; Tropical North Atlantic alone accounts for ~25.7%, Southern Annular Mode ~12.3%.
- Europe: teleconnection-fire relationships are WEAKER than on other continents. Positive NAO associated with reduced burned area in temperate regions; positive East Atlantic pattern with increased burned area in central and western Europe; Mediterranean fire linked to PDO, IOD and ENSO.
- USE: bounds what seasonal forecasting can deliver in Europe, and supports the argument that European - and by extension Alpine - fire is driven more by local and land-surface conditions than by remote climate modes.

## krsnik_spatial_2024 — Spatial multi-criteria analysis for prioritising forest management zones to prevent large forest fires in Catalonia
Krsnik, Busquets Olive, Pique Nicolau, Larranaga, Teres, Garcia-Gonzalo, Gonzalez Olabarria. Environmental Challenges 15:100959, 2024. doi:10.1016/j.envc.2024.100959. FIRE-RES.
- Catalonia, ~32,000 km2, 42% forest, 238,096 forest polygons. GIS plus Multi-Attribute Utility Theory with participatory expert weighting.
- Five criteria: fuel hazard; fire behaviour (FlamMap intensity, spread, flame height); exposure (urban, recreational, protected areas, infrastructure); ACCESSIBILITY (terrain aspect, road density, distance to water points and strategic management areas); continuity (intrinsic risk, agroforested share, historical fire return period).
- Expert weighting prioritised 20.69% of the territory; equal weighting prioritised 7.98%.
- USE: a transferable prioritisation architecture that already contains an explicit terrain-accessibility criterion, which is what Alpine prioritisation needs; and the weighting sensitivity is a caution about participatory methods.

## gorrizmifsud_current_2026 — Economic incentives boosting coordinated fuel reduction for wildfire risk mitigation in Catalonia
Gorriz-Mifsud & Rovellada Ballesteros. Fire 9(2):70, 2026. doi:10.3390/fire9020070. FIRE-RES. OA.
- Two mechanisms: wildfire prevention subsidies 2022-2025 (EUR 5,000-200,000 per proposal, funded by a Catalan vehicle CO2 tax, prioritising collective applications in strategic areas) and PROMACC forest climate credits (voluntary carbon market, minimum 50 ha landscape plans).
- Subsidies reached only 3.87% of the 151,196 ha designated as strategic wildfire areas. Average minimum distance between subsidised plots 287 m, so spatial continuity was not achieved despite being a design requirement. Climate credits: 293 of 542 approved credits sold, mobilising EUR 1,088,071.
- Barriers: parcels lack adjacency; weaker social capital disadvantages some areas; upfront costs before reimbursement exclude capital-poor owners; administrative and certification complexity.
- USE: the fragmented-ownership coordination problem, quantified. Directly transferable to Alpine forest ownership structures.

## miranda_widespread_2023 — Widespread synchronous decline of Mediterranean-type forest driven by accelerated aridity
Miranda, Syphard, Berdugo and others. Nature Plants 9(11):1810-1817, 2023. doi:10.1038/s41477-023-01541-7. FIRE-RES.
- Central Chile, compared across all five Mediterranean-type ecosystems. Over 90% of Chilean forest browned within fewer than 100 days; conditions unprecedented in the recent history of all Mediterranean-type ecosystems; 2000-2021 NDVI and climate analysis.
- Long-term recovery uncertain given continuing decline in regional water balance.
- USE: comparative evidence that drought-driven forest decline can be abrupt and synchronous at regional scale rather than gradual. Imported, non-European; use for mechanism and for the abruptness, not for magnitude.

## couto_numerical_2024 — Numerical investigation of the Pedrogao Grande pyrocumulonimbus using a fire-to-atmosphere coupled model
Couto, Filippi, Baggio, Campos, Salgado. Atmospheric Research 299:107223, 2024. doi:10.1016/j.atmosres.2024.107223. FIRE-RES.
- Pedrogao Grande, Portugal, 17 June 2017; 28,914 ha and more than 60 fatalities; concurrent Gois fire 17,521 ha. Meso-NH coupled to ForeFire, nested domains at 2 km, 400 m and 80 m.
- Updraughts above 15 m/s; convective column to ~10 km; pyroCb persisted about an hour; plume top rose from below 4 km to over 9 km. Wind direction changes triggered fire front acceleration and merger; a localised outflow with downward velocities around 15 m/s at 3 km developed ahead of the fire front; convective collapse when mid-level southeasterly winds opposed vertical motion.
- USE: demonstrates that coupled fire-atmosphere modelling at 80 m resolves the plume-driven feedbacks that terrain-channelled Alpine fires would also generate. Supports the call for terrain-aware, coupled modelling rather than surface-wind-only danger rating.

## Manifest additions
- https://link.springer.com/article/10.1007/s13280-023-01960-y
- https://www.mdpi.com/2571-6255/7/11/415
- https://fire-res.eu/wp-content/uploads/2023/11/Incorporating-fire-smartness.pdf
- https://fire-res.eu/wp-content/uploads/2024/02/VPD-based-models-of-dead-fine-fuel-moisture-provide-best-estimates-in-a.pdf
- https://www.mdpi.com/2412-3811/10/9/246
- https://www.nature.com/articles/s41467-023-36052-8
- https://fire-res.eu/wp-content/uploads/2024/06/Spatial-multi-criteria-analysis-for-prioritising-forest-management-zones-to.pdf
- https://www.mdpi.com/2571-6255/9/2/70
- https://www.nature.com/articles/s41477-023-01541-7
- https://fire-res.eu/wp-content/uploads/2024/01/Numerical-investigation-of-the-Pedrogao-Grande-pyrocumulonimbus-using-a-fire-to-atmosphere-coupled-model.pdf
- Jandl et al. 2013: http://dx.doi.org/10.5772/56267  (PDF already supplied by the author)

## schunk_forest_2013 — Forest fire danger rating in complex topography: Bavarian Alps, autumn 2011
Schunk, Wastl, Leuchner, Schuster, Menzel. NHESS 13(9):2157-2167, 2013. doi:10.5194/nhess-13-2157-2013. OA.
- Two met stations at different elevations, Bavarian Alps, autumn 2011 drought under persistent high pressure. Compared standard index calculation against fire danger determined from expert observation, fire occurrence and fuel moisture measurement.
- KEY RESULT: pronounced fire danger and multiple incidents occurred at MID-SLOPE elevations, in the thermal belt, while valleys registered only moderate danger. Valley-station data therefore misrepresents the danger field.
- Mechanism: under temperature inversion, differences in the daily cycles of meteorological parameters drive fire danger, and once-daily station observation cannot capture them.
- Recommends additional high-elevation stations, or high-resolution meteorological models feeding indices that accept at least hourly input.
- USE: FILLS THE GAP the review previously flagged as unquantified. Direct Alpine evidence that interpolation from valley floors fails, with a named mechanism (inversion, thermal belt) and a stated remedy (sub-daily input, elevation-stratified observation).

## vacchiano_modeling_2018 — Modeling anthropogenic and natural fire ignitions in an inner-alpine valley
Vacchiano, Foderi, Berretti, Marchi, Motta. NHESS 18(3):935-950, 2018. doi:10.5194/nhess-18-935-2018. OA.
- Aosta Valley, 3,262 km2, continental inner-Alpine, mean annual temperature 10.9 degC, precipitation 494 mm.
- Three separately modelled classes: summer ignitions May-Sept (n=246), winter forest fires Oct-Apr (n=316), winter grassland/fallow fires Oct-Apr (n=556). MaxEnt with topographic, climatic and anthropogenic predictors; PCA for collinearity.
- AUC 0.90 (summer), 0.95 (winter forest), 0.94 (winter agropastoral).
- 30% of fires in agricultural areas, 24% in forest; negligence in agro-silvicultural activity caused 64% of identified ignitions; lightning 9% overall, rising from 6% to 10% across the period.
- Predictors: summer driven by maximum temperature and precipitation of the wettest month; winter forest fires by precipitation and elevation; distance from infrastructure matters in winter but not summer; grazing animal density is inversely related to winter fire probability and directly related to summer.
- USE: the strongest single demonstration that the Alpine bimodal regime requires separately fitted models, with the predictor sets differing between modes. Also gives a within-Alps lightning trend and quantifies the grazing-fire relationship in both directions.

## arndt_modeling_2013 — Modeling human-caused forest fire ignition for assessing forest fire danger in Austria
Arndt, Vacik, Koch, Arpaci, Gossow. iForest 6:315-325, 2013. doi:10.3832/ifor0936-006. OA.
- Austria, 17 years of fire records 1993-2009. Logistic regression on 59 socio-economic variables covering tourism, infrastructure, agriculture and forestry.
- Retained predictors: railroad density, forest road density, hiking trail density, together with agricultural and forestry development. Final model 60.5% explanatory power against fire occurrence.
- Output: maps of vulnerable forest areas and municipality-level fire danger classes for prevention targeting.
- USE: the empirical basis for the claim that Alpine ignition clusters on the access network; supplies the specific infrastructure predictors and a national-scale validation.

## --- Author-supplied PDFs, batch 2 (2026-08-20) ---

## ascoli_firebreak_2018 — Firebreak and fuelbreak
Ascoli, Russo, Giannino, Siettos, Moreira. In: Encyclopedia of Wildfires and WUI Fires, Springer, 2018. doi:10.1007/978-3-319-51727-8_70-1. SUPPLIED BY AUTHOR.
- DEFINITIONS that the review previously lacked: firebreak = strip cleared to mineral soil; fuelbreak = retains herbaceous and shrub cover, load and vertical continuity reduced; shaded fuelbreak = retains tree canopy, treats understorey and ladder fuels.
- CORRECTS MY EARLIER DRAFT: linear fuelbreaks and shaded fuelbreaks are stated to be MORE suited to mountainous territory (steep slopes, rough morphology) precisely because retaining vegetation cover limits soil erosion. In very steep terrain, ridgelines or valley bottoms are the preferred locations.
- Widths: primary networks 30-100 m minimum; secondary and tertiary 3 m to more than 50 m.
- Siting criteria already include maintenance cost, accessibility along roads, and firefighting considerations; rivers, croplands, grasslands and previously burned areas can connect segments.
- Large and long linear networks are LESS suited where the WUI is widespread and where forest property is mainly private with small cadastral parcels.
- Also distinguishes (i) linear firebreaks/fuelbreaks, (ii) area-wide fuelbreak treatments, (iii) fuelbreaks at the WUI, with a design table by asset type (villages >100 m; scattered buildings 20-90 m; railway 2-10 m; roads 1-30 m; power lines 2-20 m).
- USE: the backbone of the new fuelbreak subsection. It supplies the terminology, establishes that a mountain-adapted form already exists and is recommended, and independently supplies the small-private-parcel constraint that the Catalan subsidy data quantifies.

## gellrich_agricultural_2007 — Agricultural land abandonment and natural forest re-growth in the Swiss mountains
Gellrich, Baur, Koch, Zimmermann. Agriculture, Ecosystems & Environment 118:93-108, 2007. doi:10.1016/j.agee.2006.05.001. SUPPLIED BY AUTHOR (was #10 on the want-list).
- Multivariate models on geo-physical and socio-economic variables, municipality level with cluster adjustment for spatial autocorrelation.
- Re-growth concentrated where cultivation costs were high and yield potential low; NON-LINEAR relationships between re-growth and both slope steepness and soil stoniness.
- Swiss forest area up by roughly one third over the last 150 years, most of it natural re-growth rather than planting.
- Conclusion: general policy measures across the whole mountain area are unsuitable for preventing abandonment and re-growth; measures must attend to local characteristics and needs.
- Notes higher wildfire probability among the consequences of abandonment, citing Romero-Calcerrada & Perry.
- USE: gives the fuel-accumulation argument an economic mechanism and a spatial prediction — fuel builds where farming is marginal, which is the steep ground where fire runs longest. Also supports scepticism about arc-wide policy prescriptions.

## bayham_economics_2022 — The economics of wildfire in the United States
Bayham, Yoder, Champ, Calkin. Annual Review of Resource Economics 14:379-401, 2022. doi:10.1146/annurev-resource-111920-014804. SUPPLIED BY AUTHOR.
- Reviews mitigation, response and recovery phases. Three drivers of US trends: changing climate, fuel accumulation, growing WUI.
- KEY MECHANISM: a private parcel owner considers only private benefits and costs of mitigation, ignoring spatial spillover externalities (fire transmission across parcels) and conventional externalities (smoke health impacts). Institutions that shape incentives: legal liability for ignition and fuel management, and insurance markets.
- USE: supplies the incentive-theoretic account of why fuel treatment under-provides on fragmented private land — the mechanism behind the Catalan 287 m gap. Non-European; used for mechanism, not magnitude.

## moritz_learning_2014 — Learning to coexist with wildfire
Moritz, Batllori, Bradstock, Gill, Handmer, Hessburg, Leonard, McCaffrey, Odion, Schoennagel, Syphard. Nature 515:58-66, 2014. doi:10.1038/nature13946. SUPPLIED BY AUTHOR.
- Fire is unusual among natural hazards in that the dominant response is to fight it, whereas for earthquakes, hurricanes and floods the emphasis is on identifying vulnerabilities and adaptations. Command-and-control fire management neglects the role of fire regimes in sustaining biodiversity and ecosystem services.
- Argues for socio-ecological systems framing, context-specific and place-based approaches, and coordinated risk management coupled to land-use planning. Draws on the Mediterranean basin, western US and Australia.
- NOTE: this entry was one of the six in references.bib that verified as genuine; the PDF now allows it to be cited properly.
- USE: the framing paragraph in the Discussion. The Alps are building a fire posture now rather than reforming an entrenched suppression regime, so the adaptation-led framing is available to them without a century of path dependency first.

## zong_effectiveness_2026 — Effectiveness of firebreaks: a review (PDF now supplied; previously read via publisher page)
Zong, Wang, Coogan, Flannigan. Fire Ecology 22(1):17, 2026. doi:10.1186/s42408-026-00451-1.
- Southern California 2017-2020: firebreaks stopped only 28% of fires above 200 ha. Western Australia 1998-2001: ~73% of wildfires fully or partially breached firebreaks. Portugal's proposed national network: -13% for fires above 500 ha, still breached in over 35% of such events.
- Firebrand transport (spotting) is the primary breach mechanism. Widths: China 30-90 m; US standard 3-152 m; Canada 30-200 m, wider on steeper slopes. Placement should follow topographic contours to limit erosion. Maintenance burden from regrowth is a recurring failure point.

## muller_forest_2020 — EUSALP white paper, PRIMARY SOURCE now read in full (83 pp.)
EUSALP Action Group 8, prepared by M. M. Muller, L. Vila-Vilardell, H. Vacik. "Forest fires in the Alps: state of knowledge, future challenges and options for an integrated fire management." White paper for policy makers, final version 13 February 2020, Vienna. SUPPLIED BY AUTHOR.
- KEY NEW FACT — heavy tail: a mean Alpine forest fire in France and Italy covers about a hectare, but 10% of fires exceed 10 ha and cause 85% of total burnt area. Slovenia: two-thirds below 1 ha. Austria, Germany, Switzerland: most fires below 0.1 ha. Southern Alps have more and bigger fires than the Northern Alps. This is the statistic that makes "80% under a hectare" safe to cite, because on its own it misleads.
- Two fire seasons: spring peak March-April before new grass, driven by frost-linked dryness and stable springtime weather; summer peak July-August. WINTER fires are an issue in foehn regions north and south of the Alps, caused by the temperature-inversion thermal belt above the fog line (cites Schunk et al. 2013) — independent corroboration of the thermal-belt mechanism.
- Ignition: ~90% human (cigarettes, fires out of control, sparks from trains or work, arson, hot ashes, power lines); ~10% lightning.
- Costs: total direct costs for firefighting and post-fire management across the Alpine region ~EUR 75 million per year, excluding prevention. Proposed integrated fire management framework ~EUR 10 million per year, of which EUR 5-7 million for the prevention recommendation.
- Natural hazards from fire: new avalanche-prone slopes, mudslides, increased runoff, rockfall, soil erosion. Swiss lineage cited: Providoli et al. 2002 (splash erosion, chestnut coppice), Maringer et al. 2016a (protection function of burnt beech against rockfall), Gehring et al. 2019 (shallow landslide disposition in burnt beech). Sass et al. 2012: steep subalpine slopes with thin humus over limestone are particularly unfavourable for regeneration; complete soil erosion may occur.
- Terminology: explicit fire danger / fire hazard / fire vulnerability / fire risk distinction. Danger is dynamic and meteorological; hazard is static, from fuel quantity/continuity and topography; vulnerability is probability of damage including ecological effects, infrastructure, settlements, human losses; risk combines causes, hazard and vulnerability.
- Prescribed burning: applied at scale in France, 1,000-3,000 ha per year in the Alpes-Maritimes, by a trained technician community with annual knowledge exchange — but in the Mediterranean south-east, NOT in the Alpine region. Objectives include hazard reduction, grazing management, habitat maintenance, firefighting training.
- Socio-economic drivers differ by country: rural abandonment dominant in France and Italy; tourism dominant in Austria, Germany and Switzerland; negligence the leading proximate cause throughout. Ageing rural population more vulnerable. Rural abandonment replaces "cultural firebreaks" (pastures) with fire-prone brushland.
- Research landscape: ALPFFIRS launched 2009 after UNISDR Wildland Fire Advisory Group action in 2008; WSL (FireLessII, FireNiche, Swissfire database); AFFRI at BOKU Vienna; Italian regional collaboration (Lombardia, Piemonte, Veneto).
- Also notes: forest fire effects on natural hazards, ecosystems, biodiversity and health "are not well understood"; little information on disturbance/forest-health interactions with fire.
- OVERLAP ASSESSMENT: this is the Alpine consensus statement and the review must position against it explicitly, not around it. Shared framings: protection function as the central concern, topography and accessibility as suppression constraints, rising costs, the bimodal season, the 90/10 ignition split. The review's distinct contribution is (i) the 2020-2026 evidence, which is 73% of its sources, (ii) a documented screening protocol and evidence-weighting convention, (iii) analytic arguments the white paper does not make — consequence decoupled from burnt area, mechanism-transfers-but-coefficients-do-not, fuel break as suppression platform, protective-function versus fuel-treatment trade-off, and the observational-threshold problem.

---

## Expansion set 5 (2026-08-21): biodiversity, fauna, ecological vulnerability, physical vulnerability, pastoral systems

### 94. Papathoma-Köhle, Hausharter, Schlögl & Fuchs (2025) — Towards a wildfire vulnerability index using expert judgement. IJWF 34, WF24114.
- **Region/scale:** Austria; building scale in the wildland–urban interface. Applied to 23 buildings along the Gladiolenweg, Oberschütt (Carinthia), on the lower southern slope of the Dobratsch.
- **Method:** 21 vulnerability indicators from a prior literature review, in three categories (building environment 6, building exterior 7, building characteristics 8). Weighted by pairwise comparison (64 comparisons, reduced four-point ratio scale) using the analytic hierarchy process. Two expert panels: 14 Austrian stakeholders invited, 9 responded (64%); 54 international experts invited, 19 responded (35%). Geometric mean aggregation. Index PVI = Σ w_i·I_i with indicator scores 1–5. Three categories weighted equally.
- **Quantitative results:**
  - Protective strips/firebreaks was the highest-weighted building-environment indicator for both panels: 0.2858 (Austrian), 0.1979 (international), the largest positive difference of any indicator (Δ = 0.088).
  - Building exterior: distance of tree crowns to the building highest (0.2579 AT / 0.2118 INT); distance to forest edge second (0.2443 AT / 0.1629 INT, Δ = 0.0815).
  - Building characteristics: roof material highest in both panels (0.2032 AT / 0.1636 INT); façade second for Austrians (0.1929, Δ = 0.0699) whereas international experts placed door/window material higher.
  - Terrain slope weighted low (0.094 AT / 0.1076 INT) despite being third most important in an earlier Greek study.
  - Austrians rated water sources above vegetation density; international experts the reverse.
  - Consistency: the building-environment cluster showed the highest inconsistency (consistency ratio > 0.1) despite having the fewest indicators.
- **Additional indicators proposed by experts (open questions):** distance to roads, fire suppression infrastructure, presence/absence of people, upwind land use, preventive measures and self-protection systems, ember screens, vegetation connectivity, fuel moisture content, wind speed and direction, population density in the WUI, response time and equipment of the fire brigade, exposure of the building to prevailing wind, slope orientation, altitude.
- **Limitations:** expert judgement substitutes for empirical damage data, which do not exist for Austria; national rather than municipal stakeholders; transferability between countries is limited by construction design (explicit comparison with the Greek empirical study); index does not represent fire intensity or dynamic fire characteristics.
- **How used:** §4.5 physical vulnerability of the built environment; §5.3 the hazard-to-risk gap and the method used where damage records are absent; §6.4 independent support for firebreaks/protective strips as the highest-weighted environment indicator.

### 95. Papathoma-Köhle, Schlögl, Garlichs, Diakakis, Mavroulis & Fuchs (2022) — A wildfire vulnerability index for buildings. Sci Rep 12, 6378.
- **Region/scale:** Mati, Attica, Greece; WUI area of about 7.8 km². Fire of 23 July 2018.
- **Event magnitude:** 102 deaths; 1,400 ha burnt; 1,200 buildings and 300 vehicles destroyed.
- **Method:** post-fire field survey; 13 candidate indicators; Boruta all-relevant feature selection; weights from normalised median importance; PVI on a 1–5 indicator scale.
- **Quantitative results:** 423 buildings assessed. Damage grade 1: 167; grade 2: 212; grade 3: 28. Eight indicators retained, five rejected. Ranked importance: roof material, structural type, terrain slope, main ground cover, leaf accumulation on roof, shutters, surrounding vegetation, roof shape. Validation of the ordinal association between PVI and observed damage: Spearman p = 3.465e−15; Kendall p = 3.25e−14; Goodman–Kruskal γ = 0.4 (95% CI 0.32–0.49); Stuart–Kendall τC = 0.3 (95% CI 0.23–0.36).
- **Limitations:** features already destroyed at survey time; damage recorded qualitatively, not monetarily; fire and fuel parameters excluded; neighbouring-buildings indicator lacked discriminatory power in dense settlement; transferability requires revalidation under different architecture.
- **How used:** §4.5 as the empirical counterpart to the Austrian expert-judgement index and as the only casualty figure available; marked as Mediterranean, imported.

### 96. Arrogante-Funes, Mouillot, Moreira, Aguado & Chuvieco (2024) — Mapping and assessment of ecological vulnerability to wildfires in Europe. Fire Ecology 20, 98.
- **Region/scale:** continental Europe and adjacent islands (Cyprus excluded), 1 km grid; terrestrial ecosystems only, croplands and unburnable cover excluded.
- **Method:** three components combined in a dynamic model: ecological value (biological distinctiveness plus conservation status, combined by PCA), coping capacity (resistance of vegetation to fire from bark thickness, tree height and basal crown height, multiplied by 1 − fireline intensity), and post-fire recovery time (species-specific time to maximum height, modified by climate, soil erosion, management and topography). 69 tree species and 7 non-forested cover types. Fire scenario set at the 95% worst propagation conditions for days on which fires larger than 2,000 ha occurred, 2001–2019, with fireline intensity from BehavePlus driven by ERA5-Land.
- **Definition used:** potential rather than actual loss, that is, the change in ecological value integrated over the recovery period under an assumed extreme fire.
- **Quantitative results:** biome ranking from most to least ecologically vulnerable: tundra; temperate conifer forests; boreal forests; Mediterranean; temperate broadleaf and mixed forests; temperate grasslands, savannas and shrublands. Temperate conifer forest vulnerability is driven by long recovery time despite moderate ecological value and coping capacity. Coping capacity was the component that most influenced the result. Non-forested vegetation (grassland, shrubland, heathland) was assigned zero coping capacity.
- **Inputs to ecological value:** 278 mammals, 711 birds, 156 reptiles, 96 amphibians, about 20,000 vascular plants; forest biomass density; leaf traits; key biodiversity areas; intact forest landscapes (500 km² minimum unit); WDPA protected areas IUCN I–IV; landscape fragmentation index; forest loss 2000–2013; IUCN threatened-species distributions.
- **Limitations:** potential rather than realised loss; validation accuracy not reported; protected areas enter as an indicator rather than being analysed separately. Notes that EFFIS maps ecological vulnerability using ecological value alone, understood as location within Natura 2000.
- **How used:** §4.3 to place Alpine temperate conifer forest in a European ranking of ecological vulnerability, and to identify recovery time rather than resistance as the term that drives it.

### 97. de Miguel & Gutiérrez (2021) — Fires in nature: a review of the challenges for wild animals. Eur J Ecol 7(1).
- **Scope:** global review of fire effects on individual wild animals, with emphasis on welfare, injury and rescue.
- **Mechanisms of direct harm:** burning, smoke inhalation, heat stress; increased risk of injury, disease and stress.
- **Taxa at elevated risk:** small mammals, cavity-nesting birds, reptiles, aquatic organisms; species without evolutionary fire adaptation and with poor dispersal.
- **Indirect effects:** loss of hollows used for denning and nesting; reduction of food resources; habitat fragmentation limiting recolonisation; post-fire sediment loads harming aquatic ecosystems.
- **Magnitude figure available:** approximately 480 million animals killed in New South Wales since September 2019 (Australian bushfires, citing Dickman 2020). Non-European, imported, single event.
- **Stated gap:** studies of fire effects on the welfare of individual animals remain scarce.
- **How used:** §4.3 for the mechanisms of direct faunal harm, explicitly marked as general and non-Alpine; supports the statement that Alpine faunal mortality is unquantified.

### 98. Rey et al. (2019) — Effects of forest wildfire on inner-Alpine bird community dynamics. PLOS ONE (expanded extraction).
- **Sites:** two inner-Alpine Swiss valleys. Leuk, 850–2,100 m, 300 ha burnt, fire of 13 August 2003, high severity. Visp, 600–1,200 m, 100 ha burnt, fire of 26 April 2011, high severity.
- **Design:** simplified territory mapping along transects, three visits between mid-April and end of June. Leuk surveyed 2006–2016 (3–13 years post-fire); Visp 2012–2016 (1–5 years post-fire); control forests surveyed in 2014.
- **Quantitative results:** 5,233 territories across 58 species. Burnt forest 21.68 ± 5.0 territories per 10 ha against 41.59 ± 1.4 in control forest. Species richness did not differ significantly between burnt and control. 14 species significantly more abundant in burnt forest (woodpeckers, wryneck, redstart, finches); 5 significantly more abundant in control forest (thrushes, nutcrackers); 14 species showed time-dependent quadratic responses peaking 8–12 years post-fire and declining as vegetation densified. Red-listed and priority species were substantially more common in burnt than control forest, with the disturbance index strengthening over time (z = 3.67, p < 0.001).
- **Limitations:** two fires only; control data extrapolated using regional correction factors rather than direct monitoring.
- **How used:** §4.3 to state the abundance-versus-composition distinction quantitatively (abundance roughly halved, richness unchanged, conservation-priority species favoured), replacing the earlier qualitative treatment.

### 99. Xu, Zhu, Ren, Yan, Fang, Ahmed & Wang (2026) — How can grazing mitigate wildfires? Sustainability 18(2), 718.
- **Scope:** global review of grazing as fuel management across Mediterranean, temperate, tropical savanna and conifer–grassland systems.
- **Quantitative results:** in southeastern Arizona grazing reduced rate of spread by more than 60% and shortened flame length to within 1.2 m in pure grass communities, with rate of spread reduced by over 50% in mixed grass–shrub areas. Mediterranean studies report fire intensity reduction and burnt-area decrease of 25.9% to 60.9% under effectively managed grazing.
- **Thresholds:** moderate grazing removes less than 50% of photosynthetic leaf area; overgrazing exceeds 50%, particularly during drought. The relationship between grazing pressure and fire risk is non-linear.
- **Species roles:** goats climb and prefer low-fibre leaves and woody shrubs, suited to brush; cattle are non-selective grazers of herbaceous fuel in forest–grassland ecotones; sheep suit shrub–grass transitions and fragmented topography.
- **Trade-offs:** appropriate grazing creates vegetation patchiness acting as a natural firebreak, promotes fire-resistant species and improves soil structure; overgrazing exacerbates erosion, facilitates invasion by flammable annuals such as Bromus tectorum, and can produce continuous homogeneous high-flammability fuel beds.
- **Seasonality in high-altitude systems:** effectiveness is season-dependent, potentially greater in drier spring or autumn periods than in humid summers.
- **Limitations:** pronounced geographical bias toward Mediterranean and North American systems; poor mechanistic understanding of woody-fuel regulation; fire behaviour models represent grazing inadequately; tipping points from risk-reducing to risk-amplifying grazing pressure not identified.
- **How used:** §6.2 to give the pastoral instrument quantitative content and to state the overgrazing trade-off; §7.5 for the cost argument relative to mechanical treatment.

### 100. European Commission Joint Research Centre (2025) — Advance report on forest fires in Europe, Middle East and North Africa 2024.
- **Quantitative results:** total EU burnt area in 2024 was 419,298 ha. Burnt area inside Natura 2000 sites was 147,017 ha, about 35% of the EU total, roughly 70% of the 2023 figure and less than half of 2022. Three countries accounted for about 60% of the protected-area damage: Portugal 36,661 ha (1.53% of national Natura 2000 area), Bulgaria 29,976 ha (0.53%), Spain 17,982 ha (0.11%).
- **How used:** §4.5 to quantify ecological exposure at EU scale, that is, the share of European burning that falls inside the protected-area network; used as context, with the Alpine share not separately reported.

---

## Expansion set 6 (2026-08-21): author-supplied full texts

Seven PDFs were supplied. Three (Xu et al. 2026, Arrogante-Funes et al. 2024, Papathoma-Köhle et al. 2022) were already entered from publisher full-text extraction; the PDFs were used to verify those entries. **Verification outcome: all three checked claims hold.** Grazing figures (>60% rate-of-spread reduction in pure grass, flame length within 1.2 m, 25.9–60.9% burnt-area reduction in Mediterranean studies, the 50%-leaf-area threshold, Bromus tectorum invasion) confirmed at source. Biome ranking (tundra most vulnerable, temperate conifer forest next, boreal third; coping capacity the most influential component; CC set to 0 for non-forested cover) confirmed at source; the source additionally notes that temperate conifer vulnerability concentrates "particularly in northern Europe and mountainous regions" and that the Alps stand out at the 90th percentile within the Mediterranean biome. Mati figures (102 deaths, 1,400 ha, 1,200 buildings, 300 vehicles, 423 buildings surveyed, damage grades 167/212/28, roof material ranked first) confirmed at source.

### 101. European Commission Joint Research Centre (2026) — Advance report on forest fires in Europe, Middle East and North Africa 2025. JRC146199.
- **Continental totals 2025:** 2,242,195 ha mapped across 46 countries in the EFFIS area of interest, a 20% increase on 2024 and almost 2.5 times 2023. EU27: 1,079,538 ha across 25 of 27 member states, the highest burnt area ever recorded by EFFIS, with the peak in July and August.
- **Natura 2000:** 424,023 ha burnt inside EU27 Natura 2000 sites, 39% of the EU27 total, the highest ever recorded, three times 2024 and twice 2023. Spain, Portugal and Romania account for two thirds. Fires in Natura 2000 were mapped in 22 member states.
- **Austria 2025:** four fires mapped, 434 ha total, of which 420.6 ha (0.03% of the national Natura 2000 area) inside Natura 2000. Land cover: agriculture 334 ha (77%), mixed forest 93 ha (21%), coniferous forest 7 ha (1.6%).
- **Slovenia 2025:** 161 ha total.
- **Caveat recorded:** comparisons with previous years use only fires above 30 ha, for temporal consistency; single-year figures use all mapped fires. Mapped fires represent about 95% of the burnt area reported by countries but only a fraction of the number of fires.
- **How used:** §4.5 to replace the 2024 Natura 2000 figures and to add the Austrian case, which shows near-total coincidence of mapped Alpine burnt area with the protected-area network and with agricultural rather than forest land cover, while simultaneously illustrating the detection threshold (four mapped fires against a national annual count in the low hundreds).

### 102. Burton, Di Giuseppe, Jones, Kelley et al. (2026) — State of Wildfires 2025–2026. ESSD Discussions, preprint in open discussion.
- **Global:** burned area 18% below the 2002–2024 average and fire carbon emissions 20–31% below average, one of the lowest global years on record, yet with severe regional extremes.
- **Europe 2025–2026:** burnt area and fire carbon emissions at record levels, concentrated in the Iberian Peninsula. EFFIS estimated 19,629 km² burned across Europe, the largest since records began in 2006 and 301% above the 2006–2024 average, with 58.35 Mt CO2, 165% above the reference mean. During 8–26 August 2025 about 5,247 km² burned in the Iberian Peninsula across 66 fires exceeding 5 km². A record USD 4.11 billion of physical assets was exposed to fire across the continent. Direct fatalities: Türkiye 17, Spain 8, Portugal 6, Cyprus 2, Greece 1, France 1; evacuations in Türkiye, Spain and Greece exceeded 110,000.
- **Attribution:** the three focal regions are Midwestern Canadian Shield Forests, Northwest Iberia and Chilean Temperate Forests and Matorral. Extreme fire weather over Northwest Iberia in August 2025 was about three times more likely because of climate change; under SSP3-7.0 fires of the same return interval are projected to burn about 78% more area by end of century.
- **Mitigation effect:** extreme burnt-area events in the focal regions are projected to be 4–23% smaller under SSP3-7.0 than under SSP5-8.5, and a further 5–36% smaller under SSP1-2.6.
- **How used:** §3.2 for the European record and for the mitigation-conditionality point; §5.5 as a second independent instance of a flagship assessment of the 2025 season selecting non-Alpine focal regions. **Preprint status noted; to be replaced with the published version.**

### 103. Vieira, Girona-García, Kastridis & Martins (2026) — A post-fire management strategy for the EU. Fire Ecology 22:63 (editorial).
- **Diagnosis:** post-fire management is the least developed component of the fire management cycle in the EU despite its role in mitigating soil erosion, nutrient loss, invasive species expansion and ecosystem service degradation. A persistent implementation gap separates scientific knowledge from on-the-ground practice; the tools (stabilisation, restoration, monitoring) already exist.
- **Proposed framework:** risk-based prioritisation, coordinated monitoring, science-policy integration, common assessment procedures, multidisciplinary operational teams, long-term funding structures, knowledge exchange between member states, with flexibility for regional ecological and institutional diversity.
- **Evidence base:** scientific literature plus operational expert experience from Greece, Portugal and Spain, with national legislation examined for those three states.
- **Context figure cited therein:** 1,031,405 ha burnt in the EU in 2025, nearly three times the 2006–2024 annual average of 354,842 ha (attributed to San-Miguel-Ayanz et al. 2025).
- **How used:** §6.5, to establish that the post-fire deficit is institutional as well as evidential at European level, with the transfer to the Alps stated as a question because the evidence base is Mediterranean and the Alpine consequence is gravitational rather than agronomic.

### 104. Miguel, Lines, Tanase, Viana-Soto, Senf & Ruiz-Benito (2026) — Unravelling the spatial and temporal variability of natural disturbances in European forests. J Appl Ecol 63:e70344.
- **Data and method:** Landsat-derived dataset of more than 3.8 million fire, wind and bark beetle disturbance events, 1985–2023. Gaussian finite mixture models characterise regimes by causal agent using disturbance size, severity and frequency, analysed across 50 km grid cells and three biomes (boreal, temperate, Mediterranean) over three periods (1985–1997, 1998–2010, 2011–2023).
- **Fire regimes:** marked north–south gradient. Severe and rare fires in boreal and temperate biomes; moderate, large and frequent fires in Mediterranean and temperate–Mediterranean transitional forests.
- **Wind and bark beetle regimes:** four distinct types. Moderate disturbance predominates in boreal and temperate biomes; mild and rare in Mediterranean and transitional forest; frequent and small in Central Europe; severe and large in the British Isles.
- **Non-stationarity:** approximately 15% of grid cells shifted to a different disturbance regime during the record. Fire frequency and severity declined in temperate and Mediterranean forests between the first and second periods. Wind and beetle disturbances decreased in size in boreal and temperate forests over the same interval, followed by increases in disturbance severity in temperate forests between the second and third periods.
- **Authors' application:** context-dependent management tailored to dominant agent and local conditions, with the Iberian Peninsula (large frequent fires) and Central Europe (increasing bark beetle outbreaks) named as most vulnerable.
- **Limitation for our purposes:** resolution is biome and 50 km grid, not Alpine; no separate Alpine result is reported.
- **How used:** §3.3, for the agent-resolved European baseline and, more importantly, for the demonstration that disturbance regimes are non-stationary within a 39-year record, which is the property static fuel representations assume away.

### 105. Naser & Kodur (2025) — Vulnerability of structures and infrastructure to wildfires. Natural Hazards 121:9995–10015 (perspective).
- **Ignition pathways:** structures ignite by wind-blown embers, radiant heat or direct flame contact; the majority of buildings lost are lost to embers and surface fires rather than to a flame front reaching the structure. Secondary pathways include internal ignition from airborne embers entering through vents, windows and gaps, damaged utilities, and stored flammable liquids or fertilisers.
- **Determinants of loss:** environmental and climate dynamics, urban planning by-laws and building codes, and human-induced factors. Planning practice is noted to overlook defensible space and fire-resistant materials.
- **Case magnitude cited therein:** the 2018 Camp Fire, one of California's deadliest and most destructive, destroyed structures of which 74% were homes, with damage estimated at USD 16.5 billion.
- **How used:** §4.5, to supply the physical reason why roof material ranks first in both the Austrian judgement-based and the Greek damage-based index, which converts a statistical agreement into a mechanistic one. Non-European; used for mechanism only.

### 106. Moftakhari & AghaKouchak (2019) — Increasing exposure of energy infrastructure to compound hazards. Environ Res Lett 14:104018.
- **Region and object:** State of California; natural gas transmission infrastructure.
- **Method:** exposure of pipelines quantified separately for wildfire and for runoff, then jointly as the coincidence likelihood of intense rainfall over recently burnt ground, projected under RCP4.5 and RCP8.5 relative to 1953–2005.
- **Quantitative results:** the likelihood that the historic mean of median compound-hazard values is exceeded rises from 38% in the past to 68% under RCP4.5 and 73% under RCP8.5. For the 90th quantile the likelihood rises from 34% to 90% and 91%, almost three times the historic value. Exposure to runoff roughly doubles in parts of the domain.
- **How used:** §4.5 as the method precedent for quantifying linear-infrastructure exposure to the post-fire compound cascade, explicitly marked as Californian, with the Alpine equivalent (roads, rail and pipelines beneath burnt slopes in confined valleys) identified as not attempted; also added to research agenda item 7 as the exposure half of a cascade-explicit corridor assessment.

---

## Expansion set 7 (2026-09-10): Alpine burn severity retrieval and the practitioner record

### 107. Morresi, Marzano, Lingua, Motta & Garbarino (2022) — Mapping burn severity in the western Italian Alps through phenologically coherent reflectance composites derived from Sentinel-2 imagery. Remote Sensing of Environment 269:112800.
- **Region and events:** 10 wildfires in mountainous areas of the Piedmont Region, Italy, during autumn 2017, following a severe and prolonged drought.
- **Problem addressed:** burn severity maps obtained by thresholding bi-temporal indices built on pre- and post-fire Normalized Burn Ratio vary substantially with matched acquisition and seasonal timing. Plant phenology, solar elevation angle, illumination variation due to topography, and soil and vegetation moisture all contribute non-fire spectral change; cloud, cloud shadow and snow further restrict the usable window in rugged terrain.
- **Method:** pixel-based compositing coupling the weighted geometric median with long-term land surface phenology metrics from MODIS to rank clear observations within the growing season, producing reflectance composites targeted at a specific phenological stage. Processing chain included sub-pixel co-registration, topographic correction and BRDF normalisation. Validation against Composite Burn Index field data.
- **Quantitative results:** CBI against bi-temporal indices R² > 0.85. Overall classification accuracy 76.9% to 83.7%, Kappa 0.61 to 0.72. Relativized differenced NBR performed best among the bi-temporal indices tested. Composites raised overall accuracy over paired cloud-free images by 4.4 to 5.2 percentage points against uncalibrated indices, narrowing to 2.8 to 3.2 points once paired-image indices were calibrated with the dNBR offset. **The extent classified as high severity increased by about 19% when composites were used instead of paired images**, because reduced contrast between healthy and burnt conditions under suboptimal seasonal timing understates burnt area.
- **Field constraint recorded:** some fires yielded fewer than 10 field plots because of steep slopes.
- **How used:** §3.4 to replace a previously speculative statement that no Alpine study had quantified terrain and phenology bias in severity retrieval. The 19% figure now anchors the claim; §3.5 and the synthesis table carry the consequence.

### 108. Müller, Vilà-Vilardell & Vacik / EUSALP Action Group 8 (2020) — practitioner and suppression content (re-extraction of an already-cited source).
Extracted for the firefighting and practitioner audit; all of this was present in the source but unused until now.
- **Extreme-event definition inverts.** In fire-prone regions an extreme fire burns thousands of hectares and its front stops through fuel absence or weather change rather than firefighter effort, that is, beyond suppression capacity limit. In the Alpine region even a relatively small fire is extreme if high severity undermines the protective function of forest above a settlement.
- **Access:** forest road density in the Alpine region is **high** compared with other European regions including the Mediterranean and Scandinavia. The accessibility constraint is therefore vertical separation and slope, not road absence.
- **Ground tactics and equipment:** tank tenders with ponds, hydrants and long hose lines; independent fire pumps where height differences are large; fire rakes and beaters for initial attack in remote terrain; backpack pumps and high-pressure lances for subsurface hot spots; sprinkler systems wetting vegetation at fuel breaks in the southern Alps, especially at night; Italian forest-fire units use all-wheel-drive pickups with firefighting modules sized for narrow Alpine forest roads; French ground suppression is based on all-wheel-drive water tanks in groups.
- **Aerial:** helicopters and fixed-wing support described as essential in the heterogeneous, small-scaled Alpine environment. All Alpine countries maintain units trained for helicopter work, water buckets and winch operations. In Austria and Bavaria helicopters increasingly transport water to mobile basins near the fire from which ground troops lay hoses, especially during mop-up.
- **Safety hazards specific to the range:** rockfall, risk of falling, new fires started by burning tree parts rolling down slope; mountain rescue teams often integrated into the action force to secure firefighters where terrain is steep; buried munition from the First and Second World Wars in some areas. Canyons can produce unexpected increases in fire intensity without accompanying strong wind (citing Viegas and Pita). Dead wood accumulation increases fire residence time and severity and makes firefighting and mop-up more dangerous.
- **Organisation and detection:** dense networks of volunteer fire brigades in Austria and Switzerland, combined with prompt public telephone reporting, mean most Alpine fires are attacked shortly after ignition. Surveillance in Italy by Carabinieri forestali, civil protection, local police and volunteers; in Germany by air observers in small fixed-wing aircraft crewed by a trained firefighter and a forester. Automated camera detection systems reduce burnt area significantly where installed but currently monitor mostly flat areas from towers, and transfer to complex terrain is unresolved. Drones are very sensitive to wind and limited by dense canopy.
- **Tactical burns** (counterfire during an incident) are distinct from prescribed burning and are used mainly in France and Italy, uncommon or absent elsewhere in the Alps.
- **National databases record suppression effort:** the Austrian register (more than 6,000 records at the time of writing, 1993 onward) includes number of action forces, helicopters and fire brigades involved. Slovenia links its forest fire database to action-force records. Switzerland holds more than 10,000 events with records reaching back to 1900.
- **How used:** §8.1, renamed "Suppression, preparedness and the practitioner constraint", gains four paragraphs on tactics, aerial dependency, working hazards and the volunteer organisational base. §3.4 gains the suppression-effort variables and the detection-technology transfer problem. §8.5 gains the extreme-event inversion. One inference is marked as untested in the text: that volunteer response density is part of the explanation for the Alpine fire-size distribution.

---

## Expansion set 8 (2026-09-11): promoted from the screened register

These six were already in the screened pool as metadata stubs. Full text was retrieved and read before citation. Retrieval was cut short by an organisation spend limit that terminated three parallel retrieval agents; a queue of further candidates is listed at the end.

### 109. Gehring, Conedera, Maringer, Giadrossich, Guastini & Schwarz (2019) — Shallow landslide disposition in burnt European beech forests. Sci Rep 9:8638.
- **Region:** southwestern European Alps, Canton Ticino (CH) and Piedmont (IT), 600–1,700 m.
- **Method:** 244 plots of 200 m² (210 burnt, 34 unburnt); 22 field root pullout tests on coarse roots to 2.5 cm and 191 laboratory tensile tests; Root Bundle Model calibration; three-dimensional probabilistic slope stability modelling; space-for-time using fires from 1970–2012.
- **Root reinforcement loss:** low severity stable; moderate severity about 15% loss in the first five years and 60% by year 23; high severity about 70% loss in the first five years and 98% by year 15.
- **Protective capacity:** for shallow cohesive soils on slopes up to 35°, low severity retains more than 50%; moderate severity more than 50% after 35 years; high severity falls below 25% regardless of soil and slope class. On steep slopes (45°) with deep soil (1.5 m), below 25%.
- **Duration:** after high severity, protective capacity vanishes for 15 years and elevated shallow-landslide probability persists at least 40 years.
- **Limitations:** fully saturated soil assumed (conservative); severity proxied by crown volume and basal area loss; space-for-time assumption; results specific to *Fagus sylvatica*.
- **How used:** §6.3, to give the loss of protective function a measured trajectory and to establish that the Alpine consequence window for this process is an order of magnitude longer than the general three-to-four-year figure.

### 110. Maringer, Ascoli, Dorren, Bebi & Conedera (2016) — Temporal trends in the protective capacity of burnt beech forests against rockfall. Eur J Forest Res 135:657–673.
- **Region:** southern European Alps; 34 beech stands burnt over 40 years, all living and dead trees inventoried.
- **Method:** Rockfor.net rockfall model applied for blocks of 0.05, 0.2 and 1 m³ on slopes of 27°, 30° and 35° with lengths of 75 and 150 m.
- **Results:** low-severity burns retain nearly the same protective capacity as unburnt forest; moderate and high severity reduce it significantly, most strongly between 10 and 30 years after fire.
- **How used:** §6.3, alongside Gehring, to establish that the trough in protective capacity arrives years after the fire rather than immediately, because standing dead stems keep working until they fall.

### 111. Sarro, Pérez-Rey, Tomás, Alejano, Hernández-Gutiérrez & Mateos (2021) — Effects of wildfires on rockfall occurrence: a review through actual cases in Spain. Applied Sciences 11:2545.
- **Cases:** Chandebrito (Galicia, granite, 3D LiDAR point cloud and RocFall trajectory simulation); Tramuntana range, Mallorca (2,400 ha burnt, rockfall inventory from road maintenance records on the Ma-10); Los Realejos, Tenerife (barrier degradation on the TF-5).
- **Quantitative:** Galicia mean block volume 0.9 m³, probability of block instability 1/20 per year, rock mass rating about 55.
- **Thermal mechanism:** granites lose strength appreciably above 500 °C approaching the α to β quartz transition at 573 °C; limestones alter from about 300 °C through dehydration and decomposition, mostly decomposed above 700 °C.
- **Three-zone framing:** source area (mineral expansion, microcracking), propagation zone (loss of vegetation that absorbs block energy and reduces frequency), affected area (heat-degraded cables and barriers, raised risk perception).
- **Stated gap:** restitution and friction coefficients for post-fire conditions are not standardised, and input uncertainty is already large before fire. Also notes a lack of information on firefighter fatalities from rockfall.
- **How used:** §5, to attach a temperature scale and a three-zone structure to the rockfall mechanism, and to record that the trajectory-model parameters for a burnt slope are the least constrained.

### 112. Perello, Trucchia, D'Andrea, Meschi, Ghasemiazma, Degli Esposti, Fiorucci, Ferraris, Gollini & Negro (2025) — Fuel-aware forest fire danger rating system RISICO. IJWF 34:WF25023.
- **Region and period:** Italy, national, 2007–2022.
- **What it computes:** dead fuel moisture content (%), rate of spread (m h⁻¹) and fire-line intensity (kW m⁻¹), hourly, on a 1 km grid.
- **Fuel awareness:** a machine-learning susceptibility model classifies the territory into 12 fuel classes (four vegetation types by three susceptibility levels), with topography embedded in the susceptibility term and fire behaviour parameters adjusted spatially.
- **Validation:** compared against 12 established danger indices using 102,904 wildfires over 16 years, by rank-percentile, ROC and territorial coverage of danger classes. Susceptibility model AUC 0.89 at European level; larger fires concentrate in the highest susceptibility quartile.
- **Limitations:** live fuel moisture fixed at a worst-case value; validation uses ignition dates and may miss multi-day duration effects; the 1 km grid implies a minimum resolvable wildfire of about 70 ha; parameter optimisation history undocumented.
- **How used:** §3.3 and the model table, as the counter-example to the generalisation that European danger rating ignores fuel and terrain, and as a further instance of a resolution floor above the size of almost every Alpine fire.

### 113. Bardsley, Bardsley, Conedera, Pütz, Robinson & Weber (2021) — Wildfire, environmental risk and deliberative planning in the Locarnese region of Switzerland. Environmental Management 68:785–801.
- **Region and method:** Locarnese, Canton Ticino; household survey with descriptive and inferential statistics on perceptions of environmental risk exposure within the Swiss deliberative planning system.
- **Findings:** high appreciation of lifestyle and amenity values and strong support for firefighting services; residents recognised increased wildfire and landslide exposure but identified the current level of urban expansion as the primary risk driver at the forest and settlement interface; younger, locally rooted and less formally educated respondents reported particular dissatisfaction with their planning influence despite formal deliberative opportunities; popular referenda may create a disproportionate sense of citizen entitlement over local planning decisions.
- **How used:** §8, to show that public support for suppression capacity is not evidence of support for the land-use restraint that would reduce exposure, and that formal participation rights do not distribute influence evenly among those carrying the risk.

### 114. Shan, Gao, Yin, Shao, Cao, Yu, Cui & Wang (2024) — Influence of terrain slope on sub-surface fire behavior in boreal forests of China. Fire 7:55.
- **Region:** Huzhong National Nature Reserve, Daxing'an Mountains, China; *Ledum palustre* and *Larix gmelinii* forest.
- **Method:** simulated smouldering with a purpose-built temperature acquisition system, 45 measurement points (5 vertical by 9 horizontal), slopes of 0°, 10° and 20°, three repeats each.
- **Results:** maximum temperature 769.02 °C at 0°, 880.69 °C at 10°, 947.91 °C at 20°. Spread rates at 0°: 3.42 cm h⁻¹ vertical and 5.94 cm h⁻¹ horizontal; at 20°: 6.50 and 10.41 cm h⁻¹. Horizontal exceeded vertical throughout and the gap widened with slope. The slope effect on peak temperature exceeded its effect on spread rate.
- **How used:** §3.2, marked as imported (Chinese boreal, not Alpine), to establish that slope acts on sub-surface smouldering as strongly as on flaming spread, which bears on the mop-up phase and on the Alpine observation that lightning fires burn underground and resist extinction.

### Retrieved but not citable
- Esposito & Gariano (2025), *Landslides* 22:2131–2139, doi 10.1007/s10346-025-02516-9, on the first fatal post-fire debris flow recorded in Italy. Only metadata was accessible; the quantitative content sits behind a paywall. **High priority for acquisition**, since it would be the first Alpine-adjacent post-fire debris flow with a recorded fatality.

### Queue identified for the next tranche
From the screened register, Alpine or mountain, fire-related, published 2012 or later, not yet cited: root reinforcement decay in a Scots pine protection forest after fire (10.1016/j.foreco.2017.06.005); fire predisposition assessment in decision support for mountain forest management (10.1016/j.mex.2025.103332); high-severity burns and eruptive fire behaviour in Alpine beech forests (10.1093/forestry/cpad036); alpine pyrogeography from fire statistics (10.1016/j.apgeog.2018.07.011); large-scale weather types and fire danger in the Alps (10.1016/j.agrformet.2012.08.011); lightning-induced forest fires in Austria (10.1007/s00704-012-0653-7 and 10.1016/j.agrformet.2017.03.020); spatio-temporal trends in fire weather in the French Alps (10.1016/j.scitotenv.2017.04.027); rainfall conditions for runoff-initiated post-fire debris flows in Campania (10.1016/j.geomorph.2022.108557). Most sit on hosts that block automated retrieval.

---

## Expansion set 9 (2026-09-11): second tranche from the screened register

### 115. Arpaci, Eastaugh & Vacik (2013) — Selecting the best performing fire weather indices for Austrian ecoregions. Theor Appl Climatol 114:393–406.
- **Region:** Austria, 22 ecoregions across mountainous, hilly and flat terrain, colline to alpine altitudinal zones; 26 meteorological stations.
- **Method:** median-based linear model with percentile analysis comparing index behaviour on fire against non-fire days; 1,331 wildfire records 1993–2007; 22 fire weather indices plus mean daily temperature and relative humidity.
- **Indices compared:** Canadian BUI, FWI, DMC, FFMC, DC; KBDI and KBDISI; M68 and M68dwd; Nesterov; Angström; Carrega I87; Munger; FMI; and others.
- **Results.** Summer (May–November), all fires: mean daily temperature selected at 8 of 26 stations, Canadian BUI 5 times, KBDI/KBDISI 6 combined, I87 3 times. Fires above 100 m²: BUI and temperature 7 times each. Fires above 1,000 m²: BUI at 6 stations, Angström at 5, FMI at 3. Winter (December–April), all fires: M68dwd at 14 of 25 stations, M68 4 times; fires above 100 m²: M68dwd at 17 of 25; fires above 1,000 m²: M68dwd 13, Munger 4.
- **Key finding:** no single index is optimal across Austrian regions in both seasons. Larger fires occurred only when fuel moisture was low enough to sustain them, and index performance improved for larger events.
- **Limitations:** selected stations were warmer and drier than their ecoregion means, potentially overestimating danger; phenological inputs coarse; binary snow treatment described as very basic; stochastic ignition unpredictable; **42% of records lacked size data**.
- **How used:** §3.3, as the strongest Alpine-country demonstration that index choice is conditional on season and ecoregion simultaneously, which is a stronger claim than the usual point about local class boundaries; the 42% figure also supports §3.4.

### 116. Müller, Vacik & Valese (2015) — Anomalies of the Austrian forest fire regime in comparison with other Alpine countries. Forests 6:903–913.
- **Region and period:** Austria, Switzerland, Italy, Slovenia; 1993–2013.
- **Method:** one-way ANOVA with Games-Howell post-hoc testing (p < 0.05), seasonal distribution across spring, summer and autumn.
- **Anomalies:** autumn 2011, 47 fires and 27 ha against a mean of 11 and 3 (p = 0.001); spring 2012, 184 fires and 56.5 ha, the highest spring count recorded, against a mean of 57 and 23; summer 2013, 166–172 fires and 107 ha against a mean of 39 and 22 (p = 0.001), largest single fire 53 ha.
- **Variability:** 20-year standard deviations, spring 50 fires and 33.9 ha; summer 42 and 34.8; autumn 10 and 6.1.
- **Regional pattern:** eastern Alpine countries (Austria, Slovenia, northeastern Italy) show greater anomalies than western regions.
- **Data quality:** approximately 50% of fire records lack burnt-area information; 56% of forest is privately owned with holdings averaging under 5 ha, which complicates standardised data collection.
- **How used:** §3.1, to show that interannual variance in the Alpine regime is large enough that a short record cannot separate trend from fluctuation, and to introduce the east–west contrast; the data-quality figures support §3.4 and the ownership figure supports the governance argument.

### 117. Kirschner, Ascoli, Moore, Clark, Calvani & Boustras (2024) — Governance drivers hinder and support a paradigm shift in wildfire risk management in Italy. Reg Environ Change 24:13.
- **Region and method:** Italy, seven nationally representative regions (Piedmont, Lombardy, Friuli-Venezia-Giulia, Tuscany, Apulia, Campania, Sardinia); qualitative thematic analysis of a structured online questionnaire, 79 participants, 26% response rate, March–April 2022, purposive sampling of wildfire experts across state and regional levels.
- **Hindering:** highly fragmented institutional structure with responsibilities increasingly allocated to disparate organisations; emergency response remaining the default despite formal precautionary commitments; local stakeholder participation displaced by extra-local actors; patchy uncoordinated legislation; state-level resources disproportionate to regional capacity. **46% of coded shortcomings concerned prevention deficits.**
- **Supporting:** national law 155/2021; the 2022 National Forest Strategy; bottom-up networks influencing legislative amendments; regional fire management plans formally committed to precaution. **58% of identified opportunities concerned enhanced prevention.**
- **Limitations:** participation bias; only state and regional experts consulted, excluding community perspectives; non-probability sampling.
- **How used:** §8, as the practitioner-sourced confirmation that the constraint is institutional rather than technical, with the observation that prevention dominates both what is missing and what is thought possible.

### 118. Pala, Melis, Pioli, Sarro, Loddo, Cinus & Brunetti (2025) — Sediment generation through thermal spalling during the 2021 Montiferru-Planargia wildfire. Sci Rep 15:30918.
- **Region:** Montiferru-Planargia, Sardinia, Italy; volcanic geology; 131.64 km² burnt; fire temperatures above 800 °C; 63.1% of the burnt area at moderate-high or high severity.
- **Method:** field survey, Sentinel-2 and aerial imagery, NBR severity mapping, geological mapping, STONE rockfall trajectory modelling, photographic quantification of spalling, sediment connectivity index.
- **Spalling:** recorded at 69 exposures, mostly tephrite-phonolite lava; proportion of rock surface affected from 0.07% to 74.57%, median 20.3 ± 19%; fragment thickness 3 mm to 13 cm; spalled deposit grain size 74.8% pebbles, 23.2% cobbles, 2% gravels. More extensive on fragmented rock (debris avalanche deposits) than on massive lithofacies, and particularly effective on sodium-phonolites, tephrites and trachytes. Occurred equally under high and moderate-high severity and was sparse below.
- **Delivery to channels:** spall fragments made up 1.3% to 36.6% of post-fire debris flow deposits; freshly spalled angular cobbles were 10 to 30% of all cobbles in the deposit. 33 debris flows in 21 sub-basins over two years, triggered by three rainstorms.
- **Limitations:** limited field data on wildfire thermal spalling; physics-based modelling needed; thermal conductivity, porosity, moisture retention, mineralogy and crack density not yet characterised by lithotype; soil erosion measurement uncertainty ±7 cm.
- **How used:** §5, to establish that fire in rocky terrain manufactures transportable sediment rather than only exposing what was already present, and to connect that manufacture quantitatively to debris-flow composition.

### 119. Snyder, Coates, Possinger, Haak, Oishi, Caldwell, Aylward & Barrett (2026) — Taxonomic and multifunctional response of soil microbial communities to wildfire and prescribed fire. Front For Glob Change 9:1694825.
- **Region:** southern Appalachian Mountains, Macon County, North Carolina; Coweeta Hydrological Laboratory (869–1,236 m) and Nantahala National Forest (976–1,628 m).
- **Method:** four watersheds (prescribed fire, fire-excluded, wildfire-impacted, partially harvested); 16S and ITS sequencing, extracellular enzyme activity, metabolic quotients, physicochemical properties.
- **Results:** prescribed-fire and fire-excluded bacterial and fungal communities did not differ significantly (p = 0.535–0.743), while wildfire and partially harvested communities differed from both (PERMANOVA p ≤ 0.003). Ectomycorrhiza abundance 63–66% under prescribed fire and exclusion, 38% after wildfire, 42% after partial harvest; soil saprotrophs 18% prescribed fire, 11% fire-excluded, 24% wildfire, 26% partial harvest. Nitrogen acquisition enzyme activity higher under prescribed fire (59 against 34 nmol mg⁻¹ MBC h⁻¹, p < 0.05). Wildfire sampled about seven years after a 2016 event; no recovery trajectory derived.
- **Limitations:** different times since disturbance prevented direct comparison of prescribed fire against wildfire on identical timescales; elevation confounded severity.
- **How used:** §6.5, marked as imported and confounded, to identify the soil microbial community as an endpoint on which prescribed fire and wildfire separate cleanly, which is the kind of endpoint the Alpine prescribed-burning question currently lacks.

### 120. Wang, Zhao, Yang, Yuan, Wang & Song (2026) — The dynamic influence of mountain-valley breeze circulation on wildfire spread. Fire 9:16.
- **Region:** Greater Khingan Mountains, northwestern Inner Mongolia, China, 200–1,500 m.
- **Method:** ten years of summer meteorological data (2015–2024) with high-resolution WRF-Fire coupled atmosphere-fire modelling of a 2023 event.
- **Circulation:** daily mean wind speeds 0.8–3 m s⁻¹ in valleys, daytime 1.0–1.5 m s⁻¹ above night; daytime up-valley north-westerlies against night-time down-valley south-easterlies; wind persistence index below 0.4 at all stations.
- **Spread by phase:** maximum spread rate 0.3 m s⁻¹ in phase 1, rising to 1.1 m s⁻¹ in phase 2 under nocturnal mountain winds of 3–5 m s⁻¹, then 0.6 m s⁻¹ in phases 3 and 4. Peak heat flux 11 kW m⁻² in phase 1, 159.3 in phase 2, 146.8 in phase 3, 246 in phase 4. Flame heights 0.2 to 0.4 m.
- **Model performance:** Sorensen similarity 0.82 against satellite-derived burnt area.
- **Limitations:** satellite burnt area excludes suppression effects, so the model overestimates burnt area in valleys.
- **How used:** §3.2, marked as imported, to show that coupled modelling resolves the diurnal valley circulation as a spread driver and that the fastest spread coincided with the night-time down-valley phase, which carries an operational implication for crew availability.

---

## Expansion set 11 (2026-09-11): comparative mountain systems

Scope change: mountain fire evidence worldwide is now a first-class tier rather than an exception admitted only where the Alps are silent. Priorities as chosen: European non-Alpine and North American ranges.

### 121. Oliveres, Castellnou, Castellarnau, Rosas-Casals, Brotons & Duane (2025) — The path to strategic fire management planning in the Aran, Pyrenees. Fire Ecology 21:9.
- **Setting:** Aran valley, Spanish Pyrenees, about 634 km²; mean altitude 1,840 m (range 578–2,418 m); mean slope 27°; Atlantic rather than Mediterranean climate; winter-dominant fire season January to March when vegetation is drought-stressed and soil moisture low; deep organic soil layers that smoulder and reignite, a suppression problem the authors compare with boreal regions.
- **Process:** transdisciplinary; community participation with municipal representatives, expert consultation, landscape fire succession modelling, forest fire potential polygons with historical fire data. Strategic Fire Management Plan approved unanimously May 2022.
- **Concept introduced:** "environmental flow of fire", the amount of fire a landscape needs to preserve its formally agreed goals. Eight goals agreed, including preventing high-severity impacts, emergency resilience, extensive livestock farming, biodiversity and cultural heritage.
- **Quantitative:** 0.66% of natural land cover to be managed with fire annually, about 7% over the ten-year horizon; historical fire rotation about 350 years; first implementation (Canejan, March 2023) managed 327 ha over 14 working days inside a pre-planned 346 ha fire management area.
- **Limitations:** two years is too short to evaluate; economic sustainability not assessed; managed fire can escape; explicit wildfire use under technical direction remains far from common European practice.
- **How used:** §6.5. The closest European mountain precedent for the Alpine prescribed-fire question, sharing winter burning windows, transhumant fire use and organic soils, and showing that adoption turned on a participatory and goal-setting process rather than on further ecological evidence.

### 122. Manczinger, Kovács & Kovács (2025) — Fire susceptibility assessment in the Carpathians using an interpretable framework. Sci Rep 15:30207.
- **Setting:** Carpathians across seven countries, 39 administrative units, about 286,295 km²; 5,173 fire incidents 2010–2020, annual average 470, peak 991 in 2012; roughly 78% in Ukrainian and Romanian counties; sharp spring peak with maximum in March.
- **Method:** interpretable machine learning with three feature-selection routes, four algorithms and SHAP attribution; 27 predictors across climatic, vegetation, topographic and socio-economic categories. Best model test ROC AUC 0.941 on 19 predictors; independent 2021–2025 validation showed fires 5.14 times more frequent in high-risk than low-risk hexagons (p < 2.2 × 10⁻¹⁶).
- **Leading predictors:** cropland first, then pasture and grassland, NDVI (negative), mean annual temperature, monthly mean temperature, distance from water, population density, distance from settlements, railway proximity, elevation.
- **Topography, and the contrast with the Alps:** lower elevations more fire-prone; steeper slopes comparatively protected, attributed to easier human access on flat terrain. Regional medians range from 0.22 in the Polish Subcarpathians to 0.77 in Lviv.
- **How used:** §3.1 and §3.5. Establishes that the anthropogenic spring season recurs outside the Alps while the elevation relationship inverts, which supports the conclusion that terrain organises the distribution without fixing the direction.

### 123. Bakke, Wanders, van der Wiel & Tallaksen (2023) — A data-driven model for Fennoscandian wildfire danger. NHESS 23:65–89.
- **Setting and data:** Fennoscandia (Norway, Sweden, Finland), 0.25° monthly, 2001–2019; 444,030 data points containing 1,439 fire occurrences, that is 0.3%.
- **Method and result:** random forest with 100 trees, 15 of 30 candidate predictors selected, shallow soil water anomaly dominant; test ROC-AUC slightly above the Canadian Fire Weather Index.
- **Detection gap:** Norwegian national fire records held about 800 fire data points for 2017–2018 against 24 in the satellite product, and a model trained on the national data performed considerably better.
- **Limitations:** satellite products systematically underestimate burnt area through failure to detect small fires; local records are superior where small fires dominate.
- **How used:** §3.4, as a second European mountain quantification of the satellite detection gap alongside the Austrian figures.

### 124. Morales-Molino, Steffen, Samartin, van Leeuwen, Hürlimann, Vescovi & Tinner (2021) — Long-term responses of Mediterranean mountain forests to climate change, fire and human activities. Ecosystems 24:1361–1377.
- **Setting:** Lago Verdarolo, 1,390 m, Northern Apennines, Italy; multi-proxy palaeoecology with 13 AMS dates, chironomid temperature reconstruction, DCA, PCA, RDA and GAM.
- **Sequence:** species-rich mixed *Abies alba* forest 10,500–5,500 cal BP with fir pollen up to 50%; beech expansion from about 5,500 cal BP; complete *Abies* collapse with no macrofossils since about 1,000 cal BP; monospecific beech today.
- **Fire and temperature attribution:** RDA explained 27.9% of Holocene pollen variation, microscopic charcoal influx 10.8%, temperature anomalies 10.3%, shared 6.9%. *Abies*, *Ulmus*, *Fraxinus excelsior* and *Tilia* strongly negative with charcoal; *Fagus* positive under light to moderate fire and negative under high fire; *Corylus* favoured by high regional burning.
- **Conclusion:** current monospecific Apennine beech forests are the product of multi-millennial land-use intensification superimposed on Late Holocene cooling, not natural succession; the displaced mixed fir forests were more drought-tolerant.
- **Limitations:** extrapolated basal chronology; reduced sample set for RDA and GAM (35 against 81); moderate robustness of the *Fagus* fire model.
- **How used:** §6.1, as independent European mountain evidence that the stand now carrying the fire is a land-use artefact, which is the Alpine claim tested in a different range and on a far longer record.

### 125. Rogeau, Barber & Parisien (2018) — Effect of topography on persistent fire refugia of the Canadian Rocky Mountains. Forests 9:285.
- **Setting:** 911,951 ha in the Alberta Rockies, 47% forested; 557 mapped refugia patches covering 117,515 ha, 27.7% of forested area.
- **Method:** boosted regression trees across three patch-size classes.
- **Relative influence:** elevation 23.7%, proportion of non-fuel at 5,000 m scale 20.3%, solar radiation as aspect proxy 14.6%, topographic position index at 2,000 m 10.1%, distance from rivers 10.1%.
- **Scale dependence:** cross-validated AUC 0.71 for all refugia, 0.77 for patches under 30 ha, 0.84 for patches under 10 ha. About 10% of the forested landscape exceeds 0.7 refugia probability; 46% can host large refugia at above 0.5.
- **Limitations:** random elements in survivorship; unconsidered environmental factors; topography may matter less during large fire events; data reflect the pre-1950 regime.
- **How used:** §3.2, for the other side of the terrain relationship, namely where fire does not go, and for the scale dependence that favours terrain-based prioritisation in a range dominated by small fires.

### 126. Rengers et al. (2024) — Evaluating post-wildfire debris-flow rainfall thresholds and volume models at the 2020 Grizzly Creek Fire. NHESS 24:2093–2114.
- **Setting:** Glenwood Canyon, Colorado; 13,000 ha burn perimeter; 11-gauge rainfall network at 15-minute intervals; pre- and post-event lidar; 2021–2022 debris-flow inventory.
- **Thresholds:** year 1 fire-wide 25.9 mm h⁻¹ (P50); year 2 33.7 mm h⁻¹ (P75); Landsat-modified year 2 40 mm h⁻¹. In year 1, 89% of observed debris flows were triggered above the modelled threshold. In year 2, eight storms exceeded the threshold and produced no debris flows. 49 storms in 2021 of which 9 triggered flows; 56 storms in 2022 of which none did.
- **Volumes:** total measured erosional volume 460,000 ± 16,700 m³; individual watersheds 160 ± 25 to 107,000 ± 3,800 m³; the standard volume model overestimated by a median factor of 4.4, with V_obs = 0.21 V_est + 180.
- **Vegetation recovery:** 24% in year 1 and 57% in year 2 averaged across indices (NBR 53%, NDVI 50%, EVI 68% at year 2).
- **Limitations:** uncertain storm timing for in-channel deposits; lidar detection limit ±10 cm; unknown river bathymetry; post-event sediment removal.
- **How used:** §5.3, as the comparator that places the Bussoleno triggering intensity within a few millimetres per hour of an independently derived threshold on another continent, and as the evidence that the threshold is a recovering quantity rather than a constant. Also used to caution that an imported volume model carries a large positive bias.

### 127. Micheletty, Kinoshita & Hogue (2014) — Application of MODIS snow cover products: wildfire impacts on snow and melt in the Sierra Nevada. HESS 18:4601–4615.
- **Setting:** Moonlight Fire basin against the unburnt Grizzly Ridge control, northern California, water years 2002–2012, fire in 2007; MOD10A1 and MODSCAG fractional snow-covered area.
- **Results:** statistically significant increase in fractional snow-covered area after the fire (p < 0.01 for water years 2008–2011); the number of days exceeding a pre-fire high snow-cover threshold rose by 81%; the increase held regardless of slope or burn severity; basin-average melt-out occurred earlier than in the unburnt control.
- **Limitations:** MOD10A1 overestimates snow cover pre-fire in forest, muting the apparent effect, so MODSCAG is the more reliable product in forested terrain.
- **How used:** §5.4. The review had recorded the effect of fire on the mountain cryosphere as unmeasured. It is measured, though not in the Alps, and the result is two-signed: canopy removal puts more snow on the ground and melts it sooner, which are opposite-signed effects on storage and on timing.

### 128. Bekar, Tavşanoğlu, Pezzatti, Vacik, Pausas, Bugmann & Petter (2020) — Cross-regional modelling of fire occurrence in the Alps and the Mediterranean Basin. International Journal of Wildland Fire. doi:10.1071/WF19158.
- **Setting:** four regions with Maxent occurrence models at 1 km and 100 m: southern Switzerland (9,536 km², 2000–2016, 831 fires), Carinthia, Austria (15,141 km², 2000–2017, 491 fires), Valencia, Spain (23,255 km², 2001–2014, 2,688 fires), south-western Turkey (32,831 km², 2013–2017, 2,512 fires); plus a cross-regional model trained on all four.
- **Results:** fine resolution always performed better (AUC gain 0.020 in Switzerland to 0.082 in Austria); successful regional models 6/8 at coarse and 8/8 at fine resolution; 12/16 transfers successful at coarse and 13/16 at fine, but regional models transferred well only between environmentally similar regions; the cross-regional model was successful in every region and most robust in transfer. Climatic predictors contributed a mean 68% in the cross-regional model against 49% in the regional models. In Austria forest variables rose from 2.3% (coarse) to 42.5% (fine), attributed to the small-scale forest mosaic.
- **Mountain caveat:** in steep terrain the road network follows geomorphology and settlements cluster on valley floors and terraces, so distance from roads or settlements correlates with altitude and temperature, making analysis of anthropogenic drivers through such proxies speculative.
- **Limitations:** predictor choice limited by data harmonised across all regions (no high-resolution population layer); relative-importance comparisons across studies to be treated with caution.
- **How used:** §3.1, for the trade between local skill and transfer robustness, for the effect of resolution, and for the mountain-specific confounding of access proxies with altitude.

### 129. Neumann, Vilà-Vilardell, Müller & Vacik (2022) — Fuel loads and fuel structure in Austrian coniferous forests. International Journal of Wildland Fire 31(7):693–707. doi:10.1071/WF21161.
- **Setting:** 93 plots in three Austrian regions (57 Northern and Central Alps, mostly *Picea abies*; 24 Southern Alps, *Pinus sylvestris*; 12 Eastern plains, *Pinus nigra*), 2009–2013, all next to recent fires; destructive sampling, nested subplots and line-intersect transects.
- **Results:** fuel load depended more on region and forest type than on age class; highest in *Picea abies*, lowest in *Pinus nigra*; dead fuels positively and live understorey fuels negatively correlated with basal area. *Pinus nigra*, the type that burns most often in Austria, had more open canopies, higher shrub and grass cover and the deepest fuel bed. No 1000-h fuel in the managed *Pinus nigra* stands, linked to removal of logs above the 7 cm merchantability limit. Canopy fuel estimated at about 10.0 t ha⁻¹ (*Picea*), 5.5 (*P. sylvestris*), 0.8 (*P. nigra*); duff not sampled but estimated at 15 to 30–70 t ha⁻¹ from the literature and Austrian forest-floor inventories.
- **Context stated:** no fire behaviour fuel models exist for Austria, so modelling relies mainly on North American fuel models; Swiss models were developed in national parks and give higher 1-h loads than these managed forests; prescribed burning is not a management option and lighting fires in forest is prohibited under the Austrian Forestry Act.
- **Limitations:** pilot study; broadleaf, mixed and *Pinus mugo* forests not sampled; time since last fire unknown.
- **How used:** §3.3 (absence of Austrian fuel models), §6.1 (measured fuel loads and structure), §6.4 (legal obstacle to prescribed burning in Austria).

### 130. Spadoni et al. (2023) — Active governance of agro-pastoral, forest and protected areas mitigates wildfire impacts in Italy. Science of the Total Environment 890:164281. doi:10.1016/j.scitotenv.2023.164281.
- **Setting and data:** 48,953 Italian fires larger than 1 ha, 2007–2017 (999,482 ha); 602 of 766 agro-forest districts retained (164 with under 10 ha burnt excluded, mostly in the Po Valley); responses are burnt share of four ecosystem services, burn severity from tree-cover loss normalised by biomass, and burnt share of the wildland–urban interface.
- **Method:** 24 predictors reduced by PCA to 10 (climate, fire weather, FlamMap flammability, society and tenure, abandonment, four land-governance components); Random Forest (R² 0.85, 0.73, 0.83) and GAMM with Alpine, temperate and Mediterranean regions as random factor.
- **Results:** climate and fire behaviour dominated; land abandonment increased all three impacts; active land governance (sustainable forest management and certification, extensive agro-pastoral activity, protected areas, rural development and LIFE funding) was associated with lower impacts, a moderate effect; governance of rural and marginal areas showed a non-linear relation with severity (decline to the median, then increase), attributed to intensive harvesting. Governance mitigation was stronger in warm, dry and flammable districts and weaker as fire weather during events became more severe.
- **Limitations:** governance represented by proxies; harvesting inferred from Landsat disturbance; tree-cover loss as severity proxy; fire assumed to reduce ecosystem services.
- **How used:** §6.3, as empirical, national-scale support for land-use policy as a fire lever and for the dependence of mitigation on fire weather.

### 131. Zumbrunnen, Menéndez, Bugmann, Conedera, Gimmi & Bürgi (2012) — Human impacts on fire occurrence: a case study of hundred years of forest fires in a dry alpine valley in Switzerland. Regional Environmental Change 12:935–949. doi:10.1007/s10113-012-0307-4.
- **Setting:** Canton Valais, 5,200 km² of which about half glacier and rock; forest area rising from 75,000 to about 95,000 ha over the twentieth century; about 900 fires 1904–2006 (85% of known-cause fires human); 2 km cells (787 and 786 forested cells).
- **Method:** GAMM with spatial correlation for 1904–1955 and 1956–2006, predictors population and road density (ignition), livestock grazing pressure and wood harvest (fuel removal), temperature and precipitation.
- **Results:** median fire occurrence 0.003 fires ha⁻¹ in the first period against 0.0015 in the second; adjusted R² 0.33 and 0.31. First period: population density strongest, then wood harvest; livestock and harvest both negatively associated with fire. Second period: temperature strongest, attributed to fire activity shifting to lower elevations after abandonment of middle and high-elevation farming and forestry; fire rose with road density up to about 30 m ha⁻¹ and then fell sharply; high livestock densities coincided with more fires in the dry eastern Valais, more likely a dryness than a fuel signal.
- **Limitations:** proxies rather than direct fuel data; coarse spatial and temporal resolution of some predictors; earlier climate data coarser.
- **How used:** §3.1 (non-linear road density), §4.1 (controls change with the socio-economic regime; temperature signal mediated by land use), §6.3 (only Alpine evidence on grazing and fire, historical and correlational).

### 132. Lockwood, Mitchell, Moore & Clement (2014) — Biodiversity governance and social-ecological system dynamics: transformation in the Australian Alps. Ecology and Society 19(2):13. doi:10.5751/ES-06393-190213.
- **Setting and method:** resilience assessment of the 160,000 ha treeless alpine and subalpine Australian Alps, almost all national park under three state and territory legislations; review of more than 150 sources and a 32-participant stakeholder workshop rating drivers and governance influences.
- **Fire findings reported:** pre-European high-intensity fire in high subalpine and alpine vegetation roughly once per century; fire frequency increased after European occupation; the 2003 fires burnt more than 10,000 ha of treeless vegetation; heathlands recovered within about eight years, but wetlands may take decades or not recover where hydrology is disrupted. After 2009 a 5% annual hazard-reduction burning target became Victorian policy, although such burning is not undertaken in the treeless alpine landscape. A 2010 cattle-grazing trial for fire-risk reduction was criticised, published evidence suggesting grazing has no significant effect on alpine fire risk, and cattle were removed by federal intervention on national heritage grounds.
- **Governance:** since 1986 an MOU between four governments supports a cooperative management programme (Australian Alps Liaison Committee) treating the range as one biogeographic region for fire, water, invasive species and Aboriginal engagement, resilient to political and funding cycles and major fires. The authors argue for network governance in a system they place in a release phase driven by climate change.
- **Limitations:** adaptive-cycle trajectory is a hypothesis; fire findings are secondary, drawn from cited literature.
- **How used:** §8.3, as a single imported case of standing cross-jurisdictional coordination in a mountain range and of the politics of fuel arguments; Table 1.
