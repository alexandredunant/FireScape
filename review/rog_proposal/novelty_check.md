# Novelty check: "We have found no review of mountain fire meteorology, nor of satellite detection limits in steep terrain"

Date: 2026-10-19. Read-only check; nothing in the project was modified.
Methods: OpenAlex API keyword searches (about 45 queries), OpenAlex abstracts, Crossref chapter metadata, WebSearch (standard and extended), and open-access full texts where I could download them (pdftotext + grep).
Every item below was returned by one of these tools. I did not add any paper from memory without a tool hit. I checked the volume, pages and author lists of the 17 most important items against Crossref on 2026-10-19.
Classes: **(i)** dedicated review of the topic; **(ii)** broader review with substantial terrain/mountain content; **(ii-minor)** broader review where terrain is only a passing mention; **(iii)** primary study, climatology or encyclopedia entry.
"Verified" means I checked the full text or table of contents. Otherwise the classification rests on the abstract only.

---

## TASK A: Mountain / complex-terrain fire meteorology

### A.1 Dedicated reviews (class i)

| Item | Scope | Evidence | URL |
|---|---|---|---|
| **Sharples, J.J. (2009).** An overview of mountain meteorological effects relevant to fire behaviour and bushfire risk. *Int. J. Wildland Fire* 18:737-754 | Overview of the mountain-meteorology elements relevant to fire weather: terrain-modified upper winds, turbulence, temperature/humidity variation, and the breakdown of the "fire weather continuum" in rugged terrain. | Abstract (OpenAlex) | https://doi.org/10.1071/WF08041 |
| **Clements, C.B. (2016).** Chapter 2, "Effects of Complex Terrain on Extreme Fire Behavior", in Werth, P.A., Potter, B.E., Alexander, M.E., Cruz, M.G., et al., *Synthesis of knowledge of extreme fire behavior: Vol. 2 for fire behavior specialists, researchers, and meteorologists*. USDA FS Gen. Tech. Rep. PNW-GTR-891, pp. ~5-18 | Sections: Wind systems in mountainous terrain; Dynamically driven winds (foehn, chinook, Santa Ana, sundowner, Diablo, Wasatch winds, foehn pause, gap winds, mountain waves, canyon channelling); Thermally driven winds (slope/valley flows, inversions); Wind modelling tools: WindNinja; Future needs. **This is a dedicated review chapter on mountain fire meteorology.** | **Verified**: table of contents and chapter text (PDF) | https://doi.org/10.2737/pnw-gtr-891 ; PDF https://gacc.nifc.gov/oncc/docs/pnw_gtr891.pdf ; record https://research.fs.usda.gov/treesearch/50530 |
| Same volume, **Werth, P.A. (2016).** Chapter 3, "Critical Fire Weather Patterns" (regional patterns incl. Rocky Mountain/Intermountain, California, Pacific NW, Australia), and Ch. 8, "Vortices and Wildland Fire" | Synoptic and regional fire-weather patterns, many of them terrain-forced (foehn-type winds) | **Verified** TOC | as above |

### A.2 Broader reviews and syntheses with terrain content (class ii)

| Item | Terrain content | URL |
|---|---|---|
| **Hayajneh, S.M., Alzghoul, M.I., Naser, J. (2026).** Wind and slope effects on wildland fire spread: a review of experimental, empirical, mathematical, and physics-based models. *Fire* 9(3):100 | Systematic scoping review of more than 150 studies on wind-slope coupling, canyons/trenches, eruptive acceleration, and coupled LES/CFD. **Verified full text**: it is about fire-spread physics on slopes. Meteorology terms are nearly absent (0 hits for foehn, katabatic, valley wind or inversion; 13 for channelling; 1 for WindNinja). It reviews terrain effects on *fire behaviour*, not mountain *meteorology*. | https://doi.org/10.3390/fire9030100 |
| **Werth, P.A., Potter, B.E., et al. (2011).** Synthesis of knowledge of extreme fire behavior: Vol. I for fire managers. PNW-GTR-854 | A manager-level synthesis of the same material. PDF blocked (403), so I could not verify its contents. | https://doi.org/10.2737/pnw-gtr-854 |
| **Sharples, J.J., Cary, G.J., Fox-Hughes, P., Mooney, S., Evans, J.P., Fletcher, M.-S., Fromm, M., Grierson, P.F., McRae, R., Baker, P. (2016).** Natural hazards in Australia: extreme bushfire. *Climatic Change* 139:85-99 | A review of extreme bushfire. OpenAlex has no abstract and I did not check the full text, so I could not confirm how much terrain content it has (the authors' work on VLS and channelling suggests some). | https://doi.org/10.1007/s10584-016-1811-1 |
| **Forthofer, J.M., Goodrick, S.L. (2011).** Review of vortices in wildland fire. *J. Combustion* 2011:984363 | Fire whirls and horizontal roll vortices, including how they form. Terrain is one mechanism (abstract). | https://doi.org/10.1155/2011/984363 |
| **Viegas, D.X., Simeoni, A. (2010).** Eruptive behaviour of forest fires. *Fire Technology* 47:303-320 | Overview of eruptive fires (slope and canyon). No abstract in OpenAlex; classed as overview/primary. | https://doi.org/10.1007/s10694-010-0193-6 |

### A.3 Broader fire-atmosphere reviews where terrain is not a focus (ii-minor)

| Item | Note | URL |
|---|---|---|
| Potter, B.E. (2012). Atmospheric interactions with wildland fire behaviour - I. Basic surface interactions, vertical profiles and synoptic structures. *IJWF* 21:779-801 | A review of 100 years of work: surface T/RH/wind, synoptic structures, vertical profiles. The abstract has no terrain focus. | https://doi.org/10.1071/WF11128 |
| Potter, B.E. (2012). ... II. Plume and vortex dynamics. *IJWF* 21:802-817 | Plumes and vortices | https://doi.org/10.1071/WF11129 |
| Heilman, W.E. (2023). Atmospheric turbulence and wildland fires: a review. *IJWF* | Turbulence; no terrain focus | https://doi.org/10.1071/WF22053 |
| Bakhshaii, A., Johnson, E.A. (2019). A review of a new generation of wildfire-atmosphere modeling. *Can. J. For. Res.* 49:565-574 | Coupled models; terrain appears only as model input | https://doi.org/10.1139/cjfr-2018-0138 |
| Weber, R.O., Dold, J.W. (2006). Linking landscape fires and local meteorology - a short review. *JSME Int. J. B* 49:590 | Plume theory and a stability index (Canberra 2003); not terrain | https://doi.org/10.1299/jsmeb.49.590 |
| Tohidi, A., Gollner, M.J., Xiao, H. (2018). Fire whirls. *Annu. Rev. Fluid Mech.* 50 | Terrain is one generation mechanism among several | https://doi.org/10.1146/annurev-fluid-122316-045209 |
| Finney, M.A., McAllister, S.S. (2011). A review of fire interactions and mass fires. *J. Combustion* | Not terrain | https://doi.org/10.1155/2011/548328 |
| Alexander, M.E., Cruz, M.G. (2013). Limitations on the accuracy of model predictions of wildland fire behaviour: a state-of-the-knowledge overview. *Forestry Chronicle* 89:372-383 | Terrain/wind error is a subtopic | https://doi.org/10.5558/tfc2013-067 |
| McCarthy, N., Guyot, A., Dowdy, A., McGowan, H. (2018). Wildfire and weather radar: a review. *JGR Atmos.* | Not terrain | https://doi.org/10.1029/2018JD029285 |
| Song, Y., Xu, C., et al. (2024). Lightning-induced wildfires: an overview. *Fire* 7(3):79 | General lightning-fire overview; no mountain focus found | https://doi.org/10.3390/fire7030079 |

### A.4 Mountain-meteorology reviews that omit fire (evidence that the two fields are separate)

- De Wekker, S.F.J., Kossmann, M., Knievel, J.C., Giovannini, L., Gutmann, E.D., Zardi, D. (2018). Meteorological applications benefiting from an improved understanding of atmospheric exchange processes over mountains. *Atmosphere* 9:371. The abstract lists hydrology, ecology, agriculture, urban planning, wind energy, transportation, air pollution and climate change, and **does not mention wildfire**. https://doi.org/10.3390/atmos9100371
- Chow, F.K., De Wekker, S.F.J., Snyder, B.J. (eds.) (2013). *Mountain Weather Research and Forecasting*. Springer. I checked 12 chapters via Crossref (diurnal mountain winds; dynamically driven winds; Alpine foehn; boundary layers/air quality; orographic precipitation; NWP in complex terrain, etc.). **None of them is on fire.** https://doi.org/10.1007/978-94-007-4098-3

### A.5 Encyclopedia entries (iii)

- Sharples, J.J. (2018/2020). "Foehn Winds". In Manzello, S.L. (ed.), *Encyclopedia of Wildfires and Wildland-Urban Interface (WUI) Fires*. Springer. https://doi.org/10.1007/978-3-319-51727-8_71-1
- Manzello, S.L. (ed.) (2020). "Downslope Wind" entry, same encyclopedia. https://doi.org/10.1007/978-3-319-52090-2_300082

### A.6 Key primary studies and climatologies (iii), for context

- Abatzoglou, J.T., Kolden, C.A., Williams, A.P., Sadegh, M., Balch, J.K., Hall, A. (2023). Downslope wind-driven fires in the western United States. *Earth's Future*. https://doi.org/10.1029/2022EF003471
- Abatzoglou, J.T., Hatchett, B.J., et al. (2020). Global climatology of synoptically-forced downslope winds. *Int. J. Climatol.* https://doi.org/10.1002/joc.6607
- Rolinski, T., Capps, S.B., et al. (2019). Santa Ana winds: a descriptive climatology. *Weather Forecast.* https://doi.org/10.1175/WAF-D-18-0160.1
- Mass, C.F., Ovens, D. (2019). The Northern California wildfires of 8-9 October 2017: the role of a major downslope wind event. *BAMS*. https://doi.org/10.1175/BAMS-D-18-0037.1
- Smith, C.M., Hatchett, B.J., Kaplan, M. (2018). A surface observation based climatology of Diablo-like winds... *Fire* 1:25. https://doi.org/10.3390/fire1020025
- Hatchett, B.J., Smith, C.M., et al. (2018). Synoptic-scale differences between Sundowner and Santa Ana wind regimes. *NHESS*. https://doi.org/10.5194/nhess-18-419-2018
- Simpson, C.C., Sharples, J.J., et al. (2014). Resolving vorticity-driven lateral fire spread using WRF-Fire. *NHESS*. https://doi.org/10.5194/nhess-14-2359-2014 ; Simpson et al. (2016) *GRL* https://doi.org/10.1002/2015GL067343
- Wagenbrenner, N.S., Forthofer, J.M., et al. (2016). Downscaling surface wind predictions... with WindNinja. *ACP*. https://doi.org/10.5194/acp-16-5229-2016
- Wastl, C., Schunk, C., et al. (2012). Large-scale weather types, forest fire danger, and wildfire occurrence in the Alps. *AFM*. https://doi.org/10.1016/j.agrformet.2012.08.011
- Schunk, C., Wastl, C., et al. (2013). Forest fire danger rating in complex topography (Bavarian Alps). *NHESS*. https://doi.org/10.5194/nhess-13-2157-2013
- Müller, M.M., Vacik, H., et al. (2013). Analysis of lightning-induced forest fires in Austria. *Theor. Appl. Climatol.* https://doi.org/10.1007/s00704-012-0653-7
- Moris, J.V., Conedera, M., et al. (2020). Lightning-caused fires in the Alps: identifying the igniting strokes. *AFM*. https://doi.org/10.1016/j.agrformet.2020.107990
- Alizadeh, M.R., Abatzoglou, J.T., et al. (2021). Warming enabled upslope advance in western US forest fires. *PNAS*. https://doi.org/10.1073/pnas.2009717118 ; Alizadeh et al. (2023). Elevation-dependent intensification of fire danger in the western US. *Nat. Commun.* https://doi.org/10.1038/s41467-023-37311-4
- Xu, C., You, C., et al. (2022). Climate-linked increasing vegetation fires in global high mountains. *Ecography*. https://doi.org/10.1111/ecog.06527
- Beloiu, M., Loeillot, T., et al. (2026). Climate-driven upward spread of forest fires in European mountain regions. *Nat. Commun.* https://doi.org/10.1038/s41467-026-72551-0
- Vukomanovic, J., Steelman, T. (2019). A systematic review of relationships between mountain wildfire and ecosystem services. *Landscape Ecol.* (a mountain-wildfire review, but not meteorology). https://doi.org/10.1007/s10980-019-00832-9

### Task A verdict

**Claim A is FALSE as written.** There are at least two dedicated reviews of mountain fire meteorology: Sharples (2009, IJWF) and Clements (2016, Chapter 2 of PNW-GTR-891, verified). There is also a 2026 systematic review of wind-slope effects on fire spread (Hayajneh et al.) and an encyclopedia entry on foehn and fire (Sharples 2018).

What my searches support:
- (a) I found no peer-reviewed journal review of mountain fire meteorology after Sharples (2009). Clements (2016) is grey literature (USDA GTR). Hayajneh (2026) covers fire-spread physics, not atmospheric processes.
- (b) No review I found links mountain fire weather to satellite observability or to post-fire consequences (hydrogeomorphic, cascading hazards).
- (c) The general mountain-meteorology reviews (De Wekker et al. 2018; Chow et al. 2013) do not cover fire.

---

## TASK B: Satellite detection limits in steep or mountainous terrain

### B.1 Reviews of burned-area, active-fire and burn-severity remote sensing

| Item | Terrain / steep slopes | Small-fire omission | Evidence | URL |
|---|---|---|---|---|
| **Chuvieco, E., Mouillot, F., van der Werf, G.R., San Miguel, J., Tanase, M., Koutsias, N., García, M., Yebra, M., Padilla, M., Gitas, I., Heil, A., Hawbaker, T.J., Giglio, L. (2019).** Historical background and current developments for mapping burned area from satellite Earth observation. *Remote Sens. Environ.* 225:45-64 | Not mentioned in the abstract. I could not download the full text (403/bot wall), so passing mentions (e.g., topographic shadow as a spectral confusion) cannot be ruled out. **Not a dedicated topic.** | Covered (product evaluation) | Abstract only | https://doi.org/10.1016/j.rse.2019.02.013 |
| **Wooster, M.J., Roberts, G.J., Giglio, L., Roy, D.P., Freeborn, P.H., Boschetti, L., Justice, C., Ichoku, C., Schroeder, W., et al. (2021).** Satellite remote sensing of active fires: history and current status, applications and future requirements. *Remote Sens. Environ.* 267:112694 | Not in the abstract. Full text blocked. **Not a dedicated topic** as far as I could check. | Detection physics and limits are covered | Abstract only | https://doi.org/10.1016/j.rse.2021.112694 |
| **Mouillot, F., Schultz, M.G., Yue, C., Cadule, P., Tansey, K., Ciais, P., Chuvieco, E. (2014).** Ten years of global burned area products from spaceborne remote sensing - a review: analysis of user needs and recommendations for future developments. *Int. J. Appl. Earth Obs. Geoinf.* 26:64-79 | **Verified full text**: "topographic shadow contamination" appears only as a metadata flag that users requested. There is no analysis. **Absent as a topic.** | Discussed (small fires, omission/commission targets) | Full text (figshare) | https://doi.org/10.1016/j.jag.2013.05.014 |
| **Kurbanov, E., Vorobev, O., Lezhnin, S., Sha, J., Wang, J., Li, X., Cole, J., Dergunov, D., Wang, Y. (2022).** Remote sensing of forest burnt area, burn severity, and post-fire recovery: a review. *Remote Sensing* 14:4714 | **Verified full text**: 1-2 sentences ("dNBR ... may be difficult because of variations in topography") plus citations of Verbyla et al. 2008 and Mitri & Gitas 2004. **Passing mention only.** | 1 mention | Full text | https://doi.org/10.3390/rs14194714 |
| **Szpakowski, D.M., Jensen, J.L.R. (2019).** A review of the applications of remote sensing in fire ecology. *Remote Sensing* 11:2638 | **Verified full text**: one paragraph on shadows biasing severity (from one study); topography otherwise appears as a fire-risk predictor. **Passing mention.** | 1 mention | Full text | https://doi.org/10.3390/rs11222638 |
| Lentile, L.B., Holden, Z.A., Smith, A.M.S., Falkowski, M.J., Hudak, A.T., Morgan, P., Lewis, S.A., Gessler, P.E., Benson, N.C. (2006). Remote sensing techniques to assess active fire characteristics and post-fire effects. *IJWF* 15:319-345 | Not in the abstract; full text not checked | - | Abstract | https://doi.org/10.1071/WF05097 |
| Chuvieco, E., Aguado, I., Salas, J., García, M., Yebra, M., Oliva, P. (2020). Satellite remote sensing contributions to wildland fire science and management. *Curr. For. Rep.* 6:81-96 | Full text blocked; abstract does not mention terrain | - | Metadata | https://doi.org/10.1007/s40725-020-00116-5 |
| Miller, C., Harvey, B.J., Kane, V.R., Moskal, L.M., Alvarado, E. (2023). Different approaches make comparing studies of burn severity challenging: a review of methods used to link remotely sensed data with the Composite Burn Index. *IJWF* | Methods review; terrain not in the abstract | - | Abstract | https://doi.org/10.1071/WF22050 |
| Gitas, I.Z., De Santis, A., Mitri, G. (2009). Remote sensing of burn severity. In Chuvieco (ed.), *Earth Observation of Wildland Fires in Mediterranean Ecosystems*, Springer | Book chapter; not checked | - | Metadata | https://doi.org/10.1007/978-3-642-01754-4_10 |
| Ramsey, S., Jones, S., et al. (2024). Review of approaches and challenges for the validation of satellite-based active fire products in savannah ecosystems. *IJWF* | Savannah only; not terrain | - | Metadata | https://doi.org/10.1071/WF23202 |
| Allison, R.S., Johnston, J.M., Craig, G., Jennings, S. (2016). Airborne optical and thermal remote sensing for wildfire detection and monitoring. *Sensors* 16:1310 | Airborne; not checked for terrain | - | Metadata | https://doi.org/10.3390/s16081310 |

**No review dedicated to satellite fire detection or burned-area/burn-severity mapping in steep or mountainous terrain was found** in any OpenAlex query or web search ("mountain/complex terrain/topographic" combined with "burned area/active fire/burn severity" and "review").

### B.2 Primary studies documenting terrain-related detection limits (iii): the evidence base that has not been synthesized

- Gitas, I.Z., Devereux, B.J. (2006). The role of topographic correction in mapping recently burned Mediterranean forest areas from Landsat TM images. *Int. J. Remote Sens.* 27:41-54. Slope illumination and shadowing cause burned/unburned confusion, and topographic correction improves accuracy. https://doi.org/10.1080/01431160500182992
- Mitri, G.H., Gitas, I.Z. (2004). Burned area object-based classification model on topographically and non-topographically corrected TM imagery. *IJRS* 25:2863-2870. https://doi.org/10.1080/01431160410001688321
- Verbyla, D.L., Kasischke, E.S., Hoy, E.E. (2008). Seasonal and topographic effects on estimating fire severity from Landsat TM/ETM+ data. *IJWF* 17:527-534. dNBR is negatively biased in shaded valleys and on steep north-facing slopes. https://doi.org/10.1071/WF08038
- Veraverbeke, S., Verstraeten, W.W., Lhermitte, S., Goossens, R. (2010). Illumination effects on the differenced Normalized Burn Ratio's optimality for assessing fire severity. *Int. J. Appl. Earth Obs. Geoinf.* 12:60-70. https://doi.org/10.1016/j.jag.2009.10.004
- Vanonckelen, S., Lhermitte, S., et al. (2014). Performance of atmospheric and topographic correction methods on Landsat imagery in mountain areas. *IJRS*. https://doi.org/10.1080/01431161.2014.933280
- **Fornacca, D., Ren, G., Xiao, W. (2017).** Performance of three MODIS fire products (MCD45A1, MCD64A1, MCD14ML), and ESA Fire_CCI in a mountainous area of Northwest Yunnan, China, characterized by frequent small fires. *Remote Sensing* 9:1131. Verified text: without topographic adjustment, the real surface area of a pixel on steep slopes is larger, "making the detection of small fires more difficult". https://doi.org/10.3390/rs9111131
- Fornacca, D., Ren, G., Xiao, W. (2018). Evaluating the best spectral indices for the detection of burn scars at several post-fire dates in a mountainous region of Northwest Yunnan. *Remote Sensing* 10:1196. https://doi.org/10.3390/rs10081196
- Ying, L., Shen, Z., Yang, M., Piao, S. (2019). Wildfire detection probability of MODIS fire products under the constraint of environmental factors (Yunnan). *Remote Sensing* 11:3031. Only 11.1% of 5145 confirmed ground fires were detected. https://doi.org/10.3390/rs11243031
- **Jiao, M., Quan, X., Yao, J. (2022).** Evaluation of four satellite-derived fire products in the fire-prone, cloudy, and mountainous area over subtropical China. *IEEE GRSL*. All products scored F1 < 0.5. The authors state that "no study evaluated the performance of these fire products in fire-prone, cloudy and mountainous areas". https://doi.org/10.1109/LGRS.2022.3188259
- Small-fire omission, not terrain-specific: Randerson, J.T., et al. (2012) *JGR-Biogeosci.* https://doi.org/10.1029/2012JG002128 ; Ramo, R., Roteta, E., et al. (2021) *PNAS* https://doi.org/10.1073/pnas.2011160118 ; Khairoun, A., Mouillot, F., et al. (2024) *Sci. Total Environ.* https://doi.org/10.1016/j.scitotenv.2024.170599 ; Zhao, J., Ciais, P., et al. (2025) *Nat. Commun.* https://doi.org/10.1038/s41467-025-56218-w ; Fornacca, D., Ye, Y., et al. (2025) *Fire* 8:422 https://doi.org/10.3390/fire8110422 ; Sofiev, M., Kouznetsov, R., et al. (2026, preprint) Observation operator and detection limits for MODIS and VIIRS FRP products, EGUsphere https://doi.org/10.5194/egusphere-2026-89

### Task B verdict

**Claim B is TRUE only in a narrower form.** I found no dedicated review of satellite fire-detection, burned-area or burn-severity limits in steep or mountainous terrain. The major reviews either treat terrain in passing or, as far as I could verify, not at all: Mouillot 2014 (metadata flag only), Kurbanov 2022 (1-2 sentences), Szpakowski 2019 (one paragraph), and Chuvieco 2019 and Wooster 2021 (abstracts only; full text not accessible). Small-fire omission, however, has been reviewed and quantified at length (Chuvieco 2019; Ramo 2021; Mouillot 2014). A reviewer could therefore falsify "no review of satellite detection limits" if it is read broadly, but not "no synthesis of terrain effects on detection".
**Caveat**: the full texts of Chuvieco et al. (2019) and Wooster et al. (2021) could not be retrieved. The authors should skim them for "topograph*" before submission.

---

## Proposed replacement sentences (≤ 40 words each)

**Claim A** (I could not exhaustively rule out "no review since 2009", so it is phrased as "to our knowledge"):
> "Mountain fire meteorology has been reviewed (Sharples, 2009; Clements, 2016), but to our knowledge no review has since connected terrain-forced fire weather to satellite observability and post-fire consequences across mountain ranges." (31 words)

**Claim B:**
> "Terrain effects on satellite fire detection and burn-severity mapping (illumination, shadowing, pixel foreshortening, small-fire omission) are documented in case studies (e.g., Verbyla et al., 2008; Fornacca et al., 2017) but, to our knowledge, have never been synthesized." (36 words)

**Combined alternative** (if one sentence is needed):
> "Existing reviews treat mountain fire meteorology (Sharples, 2009; Clements, 2016) and satellite fire mapping (Chuvieco et al., 2019; Wooster et al., 2021) separately; none, to our knowledge, examines how steep terrain jointly shapes fire weather, detectability and consequence." (37 words)
