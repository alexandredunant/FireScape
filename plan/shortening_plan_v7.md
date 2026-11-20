# Shortening plan: review_paper_v7 (≈31k → ≈15k words)

Status: **executed 2026-11-20**: main text 13,488 words (was ≈29,900), 45 pp (was 78); 136 cited sources (was 159; 23 dropped, all on the §5 list); search protocol and glossary moved to `review_paper_v7_SI.tex`. Awaiting author review before commit.
Basis: claim-by-claim evidence audit of the restyled v7 (commit `02c52f5`); the per-section audit files were working material and have been removed; their conclusions are consolidated in §3–§5 below.

## 1. Principle

The paper should make the multi-study trends prominent and present single-publication arguments as such.

| Evidence class | Definition (paper's own convention, Sect. 2) | Treatment |
|---|---|---|
| **Robust** | ≥3 independent studies (different groups or datasets; same team or site counts once; a review pooling many studies may count) | Opens its subsection; stated firmly; citations grouped |
| **Supported** | 2 independent studies | Stated with "two independent studies…" |
| **Single** | 1 study, one group, one site or one event | One hedged clause, at most one key number; kept only if it is the only (Alpine) evidence for a mechanism the argument needs, otherwise cut |
| **Uncited** | No source | Cut, or source it; review's own inferences marked "we suggest/argue" |

Imported (non-Alpine) evidence stays marked as imported at the point of use.

## 2. Budgets

| Section | Now | Audit proposal | Budget |
|---|---|---|---|
| Introduction + Review Design | 2,630 | 1,650 | 2,000 |
| Wildfire as a Natural Hazard (incl. Table `tab:models`) | 5,960 (+600 table) | 1,790 (+400 table) | 2,300 |
| Climate + Water | 4,440 | 2,150 | 2,400 |
| Forest | 7,970 | 2,735 | 2,800 |
| Risk + Governance | 5,550 | 2,390 | 2,600 |
| Discussion + Conclusions | 3,390 (+450 table) | 1,770 | 2,100 |
| **Main text** | **≈29,900** | **≈12,500** | **≈14,200** |
| Search protocol + Glossary | 1,200 | → Supporting Information | — |

The ≈1,700-word gap between proposal and budget is reserve for the judgement calls in §6.

## 3. Robust trends to foreground

Grouped by where they open a subsection. ⚑ = caveat to state.

**Regime and observation**
1. Burning is shifting from Mediterranean to temperate and mountain Europe, and upslope (Jones 2022; Miller 2025; Beloiu 2026).
2. The Alpine regime is dominated by sub-hectare fires with a heavy tail, which continental satellite products (≥30 ha) miss (Müller 2020; Conedera 2018; Beloiu 2026; JRC 2026; Papathoma-Köhle 2025). ⚑ Müller, Valese and Conedera share one consortium lineage (ALP FFIRS/EUSALP).
3. Two fire seasons with distinct drivers: human spring surface fires and summer heat/lightning fires (Müller 2020; Valese 2011; Schunk 2013; Vacchiano 2018; Trucchia 2022; Bercak 2024).
4. Ignition is ≈90% human and follows access; lightning forms a distinct high-elevation summer mode (Müller 2020; Conedera 2006; Vacchiano 2018; Gerberding 2025).
5. Locally fitted ignition and danger models perform well locally but transfer poorly, so indices need recalibration (Vacchiano 2018; Arndt 2013; De Angelis 2015; Arpaci 2013; Bekar 2020; Gelabert 2025; Zacharakis 2023; Moreno 2025 preprint).

**Climate and weather**
6. Atmospheric aridity and fuel moisture (VPD, DFMC anomaly, multi-month drought) are the most consistent climatic drivers (Miller 2025; Gelabert 2025; Beloiu 2026; Venäläinen 2014; Ruffault 2018).
7. Fire danger has risen, most clearly in the extremes, and is projected to rise further before mid-century (Wastl 2012; Venäläinen 2014; Gerberding 2025; Miller 2024; Billing 2026). ⚑ Gerberding is Eurac co-authored; north/east emphasis rests on Gerberding + Miller only.
8. Snow cover below ≈2,000 m is declining, relaxing the elevational constraint on fire (Pepin 2022; Rangwala 2012; Pede 2022).
9. Relief makes fire weather discontinuous and wind–slope spread non-linear (Sharples 2009; Hayajneh 2026; Ozaki 2022; Schunk 2013). Imported mechanism.

**Water**
10. Fire lowers debris-flow rainfall thresholds, which recover with regrowth (Santi 2022 review; Rengers 2024; Tiranti 2021; La Porta 2025). Imported mechanism with two Alpine cases.
11. Fire degrades water regulation and quality (Basso 2021; Misseyanni 2025; Lecina-Díaz 2021). Imported; unmeasured in the Alps.
12. Fire alters snow storage and snow/ice albedo (Koshkin 2022; Micheletty 2014; Aubry-Wake 2022; Pepin 2022). Imported.

**Forest**
13. Land-use retreat drives forest expansion and sub-treeline infilling, lengthening continuous fuel runs (Bebi 2017; Garbarino 2020; Gellrich 2007; Frei 2023; Stritih 2023; Conedera 2024).
14. Drought, storms and bark beetles are loading forests with dead fuel (Bebi 2017; Senf 2021; Giuggiola 2010).
15. Protective function is lost for years to decades after fire; recovery is multi-decadal and sometimes arrested (Maringer 2016 + Gehring 2019 [one group]; Stritih 2023; De Giuli 2024; Melzner 2025). ⚑ The 40-year landslide figure is one group's southern-beech result: hedge it.
16. Warming erodes protection in low and montane forests while subalpine stands hold (Schumacher 2006; Moos 2021; Hillebrand 2023).
17. Protection, not timber, is the management objective, and thinning on steep slopes rarely pays (Dincă 2025; Jandl 2013; Accastello 2019).
18. Linear firebreaks are weak barriers alone and work mainly as anchors for ground crews (Zong 2026; Ortega 2024; Ascoli 2018). Imported.
19. Fuel reduction through grazing and agro-pastoral land use lowers fire impact (Rouet-Leduc 2021; Xu 2026; Spadoni 2023). ⚑ Imported; Australian Alps evidence (Lockwood 2014) finds no grazing effect, so hedge.
20. Post-fire management is the least developed phase of the cycle (Gonçalves 2025; Vieira 2026; Mantero 2023).

**Risk and governance**
21. Governance is fragmented across agencies, countries and EU policy (Kirschner 2024; Hoang 2026; Plana 2024; Bardsley 2021).
22. Spending is response-dominated, although prevention is judged more efficient (Müller 2020; Michetti 2019; Lecina-Díaz 2023; Kirschner 2024).
23. Terrain and access are the binding constraint on suppression (Müller 2020; Conedera 2024; Cardil 2025 [imported, California]).
24. Exposure is dynamic, driven by tourism, recreation and the WUI (Gelabert 2025; Pede 2022; Müller 2020; Hoang 2026).
25. Prevention effort is not reaching where it is needed (Dubo 2023; Górriz-Mifsud 2026; Bayham 2022; Kirschner 2024). ⚑ Different mechanisms; state as a convergent direction.

**Discussion opening (proposed):** trends 2, 13, 1/7, 15 and the convergent finding that imported tools lose skill in Alpine terrain.

## 4. Factual corrections found during the audit (apply regardless)

| Location | Issue | Checked against PDF | Fix |
|---|---|---|---|
| Intro, baseline paragraph (≈L170) | Cardil 2025 presented as Alpine operational quantification | Study uses 26,907 Californian ignitions | Mark as imported, or move to the suppression subsection |
| Climate, L387 | "DFMC anomaly … lower at the three Mediterranean sites" (Gelabert 2025) | "For all PS models, the yearly anomaly in DFMC was the most influential variable" | Say it ranked first at every site |
| Conclusions, L770 | 1,350 CHF per ha appears only in the Conclusions | Correct (Moos 2021) | Introduce it in the body or drop it from the Conclusions |
| Hazard, L275 | Beloiu 42% above 1,400 m | Correct | Keep |
| Forest/Governance | Hillebrand 58%; Zong Arnhem Land | Correct (evidence note wrong on Zong) | Keep |
| Several | Uncited claims: fuel-break crews (L128), pioneer broadleaves (L540), browsing precondition (L542), smouldering (L571), suppression equipment (L635–639), Portugal target (L661) | — | Cut, or add a source |

## 5. References that would leave the bibliography (22 + 5 conditional)

Each key below occurs only in material proposed for cutting (grep-checked on the whole file).

- **Hazard:** bercak_combination_2024, wang_seasonal_2023, rogeau_effect_2018, bakke_datadriven_2023, couto_numerical_2024
- **Climate/Water:** baranovskiy_deterministic_2021, bozicek_impact_2023, gomezgallego_interaction_2022, miranda_widespread_2023, wilson_distribution_2022, zumbrunnen_reconstructing_2010
- **Forest:** moralesmolino_longterm_2021, reilly_drivers_2022, cagliero_effect_2022, demiguel_fires_2021, snyder_taxonomic_2026, rigolot_recommendations_2003, ager_optimizing_2023, majlingova_sustainability_2025
- **Risk/Governance:** ghodrat_existing_2021, abrham_wildfires_2024
- **Discussion:** payne_nature_2020
- **Conditional (depends on §6 choices):** malandra_burn_2022, keeping_attribution_2026, lockwood_biodiversity_2014, kalapodis_towards_2025, burton_state_2026 (stays via L371 unless that also goes)

Note: bercak_combination_2024 also corroborates trend 3, so it can stay as a grouped citation there if wanted.

## 6. Decisions (author, 2026-11-20)

- **Burton mitigation paragraph:** CUT from the Introduction (Burton stays cited in Hazard).
- **Table `tab:models`:** SHRINK IN PLACE to ≈400 words.
- **Discussion:** OPEN WITH THE ROBUST TRENDS; the imported-tools contrast table follows as their consequence.
- **Billing 2026:** HEDGE ("one continental modelling study suggests…").
- **Research agenda:** one sentence per discipline (default taken).
- **Repeated points:** one home each, with one-line back-references (default taken). Homes: Bussoleno and the snow–glacier loop → Water; bark-trait ranking → Forest (first half); thinning vs protection → Forest (second half); monetary values (Moos, €/ha) → Governance/Economics; dubo, mandate, Moftakhari, Baselt, Papathoma slope disagreement, grazing → Governance body, briefly referenced in the Discussion.

### Original questions

1. **Mitigation paragraph (Burton 2026, preprint, non-Alpine regions).** Cut, or keep as one hedged clause?
2. **Table `tab:models`.** Shrink in place (≈400 words), or move to Supporting Information?
3. **Discussion order.** Open with the robust trends (evidence first), with the imported-tools contrast table second, or keep the current critique-first order?
4. **Research agenda.** One integrated paragraph (≈310 words), or one sentence per discipline (≈460 words)?
5. **Repeated points, one home each.** Bussoleno sub-threshold debris flow (5×), snow–glacier–soot loop (2×), thinning vs protection (3×), bark-trait ranking (3×). Proposed: the body section holds the detail and other sections back-reference it in one line.
6. **Billing 2026 "management is a primary determinant".** Rests on one modelling study, but the governance argument builds on it. Hedge as "one continental modelling study suggests", or keep it firm?

## 7. Execution (after approval)

1. Apply the §4 corrections on the restyled base.
2. Rewrite section by section in parallel, each against its budget and its audit file. The checker is changed so that dropped citation keys and numbers are allowed but listed; every dropped key must be on the §5 list.
3. Merge, then check: (a) no orphan `\ref`; (b) bibliography diff equals §5; (c) style metrics stay on target; (d) clean compile; (e) word count ≈14–15k.
4. Move the search protocol and glossary to a Supporting Information file.
5. Review the diff with the author; then commit and push.
