# Reviews of Geophysics requirements and dual-track venue strategy

Working note, revised 20 November 2026 (aligned with the shortened, evidence-graded `review_paper_v7.tex` and `review_paper_v7_SI.tex`).

---

## 1. Requirements of *Reviews of Geophysics* (Primary Target)

### 1.1 Proposal and timeline

*Reviews of Geophysics* (RoG) is **invitation-only**:

1. **Send a brief proposal** (e-mail text in `email_to_rog.md`, attachment `RoG_proposal.pdf` exported from `proposal.md`; the longer `cover_letter.md` is an optional alternative) to **reviewsgeophysics@agu.org** containing:
   - proposed paper title and potential authors;
   - scientific questions to be addressed;
   - draft abstract (< 250 words);
   - manuscript outline;
   - analysis of recent similar review articles;
   - how *RoG*'s broad geoscience readership benefits.
2. **Editorial Board review:** reply typically within **two weeks**.
3. **Formal invitation:** issued via AGU's **GEMS** system if accepted.
4. **Submission window:** **three months** from invitation. Because `review_paper_v7.tex` is already fully drafted and citation-audited, no additional literature expansion is required after invitation.

### 1.2 Length, format and required AGU elements

| Element | *RoG* / AGU Rule | Status in `review_paper_v7.tex` (20 Nov 2026) | Action needed if invited |
|---|---|---|---|
| **Page limit & charges** | **None** at *RoG* (AGU's general ~30 PU review guideline is advisory) | 45 pages total (pp. 1–33 main text, ~13,500 words; pp. 34–45 references, 136 cited sources). Estimated ~32 PU (~27 PU of text plus 5 figures and tables), close to the ~30 PU guideline (was 78 pages and ~64 PU) | **Ready**; no further cuts needed |
| **Abstract** | Single paragraph, **< 250 words** | **248 words** | **Ready** |
| **Key Points** | Up to **3 bullets**, each **$\le$ 140 characters**, no abbreviations | **3 bullets** (131, 137 and 127 characters; zero abbreviations), controlled by `\ifaguformat` | **Ready** |
| **Plain Language Summary** | **Required** (~150–200 words for non-specialists) | **Included** (175 words), controlled by `\ifaguformat` | **Ready** |
| **Visual synthesis** | ~5–6 figures/tables recommended | **2 TikZ figures** (`fig:cascade`, `fig:flow`) + **3 synthesis tables** (`tab:regimes`, `tab:models`, `tab:contrast`) = **5 visual items** | Optional map of Alpine Convention pyroregions if desired |
| **Template** | AGU `agujournal2019` LaTeX class | Clean standard `article` class | Convert preamble to `agujournal2019` upon invitation |
| **Supporting Information** | Separate file using the AGU SI template (Text S1, S2…) | `review_paper_v7_SI.tex`: Text S1 search protocol and screening register; Text S2 glossary. Section numbers refer to the main text | Transfer to the AGU SI template; upload the screening register (`supplementary/FireScape_Alps_screening_register.xlsx`) as Data Set S1 |

### 1.3 Changes since the 18 November version

- **Shortened to ~13,500 words of main text** (from ~29,900) and reorganised as an evidence-graded synthesis. Each section opens with the trends supported by at least three independent studies. Findings from two studies are reported as such, and single-study results are kept only where they are the sole Alpine evidence for a mechanism, hedged and with their case count stated. See `plan/shortening_plan_v7.md`.
- **Discussion** now opens with the five findings that recur across independent studies, and the imported-tools contrast table (*Table 3*) follows as their consequence.
- **Corrections verified against the source PDFs:** the Gelabert et al. (2025) fuel-moisture ranking; Cardil et al. (2025) now marked as Californian (imported) evidence; the Moos et al. (2021) per-hectare value introduced in the economics section; and the Avouris et al. (2025) preprint result hedged.
- **136 sources cited** (23 removed, mainly single non-Alpine examples). The search protocol and glossary moved to Supporting Information.
- **Large-fire potential added (20 Nov 2026):** the paper (abstract, climate-projections subsection, Discussion, research agenda, Conclusions, Plain Language Summary) and all RoG material now state that a shift towards more frequent large fires is plausible but not yet quantified, from converging evidence checked in the PDFs: Beloiu 2026 (occurrence and burnt area of fires of at least 30 ha increased 2000-2025, surge after 2015), Wastl 2012 (return period of extreme danger values decreased, western and southern Alps), Billing 2026 (burnt-area hotspots in medium-elevation terrain including the Alps; lower model agreement in alpine regions), Conedera 2024 (fuel connectivity and slope explain large-fire statistics; scenario assumes severe future fire weather). Counter-evidence stated: fewer fires where suppression improved (EUSALP 2020: Alpes-Maritimes, Ticino); no projection of the Alpine fire-size distribution exists. Not yet retrievable automatically and worth adding once obtained: Pimont et al. 2022 (IJWF, 10.1071/WF22103, open access), Dupuy et al. 2022 (Advances in Forest Fire Research), Cane et al. 2013 (Climatic Change, 10.1007/s10584-013-0775-7, closed).
- **Evidence audit of all RoG material (20 Nov 2026):** every factual claim in the proposal, e-mail, cover letter, abstract, Key Points and Plain Language Summary was checked against the review's citations and, for single-source claims, against the source PDFs (EUSALP 2020 80% below 1 ha; Beloiu 2026 shift after 2015; Maringer 2016 10-30 yr; Gehring 2019 40-50 yr; Tiranti 2021 two events after brief moderate rain; Schunk 2013 thermal belt; Negri 2026 repellency). Characterisations of comparison reviews were checked against their full texts (Jones, Pepin, Or, Melzner, Sharples, Mani, Moos, Werth, Wooster, Chuvieco, McGuire; all PDFs in `papers/`); unverifiable negative statements were removed. Tools are no longer described as built for Mediterranean fires (the Fire Weather Index is Canadian), and claims resting on one or two sources are hedged.
- **Claims aligned with the evidence (20 Nov 2026):** the paper and all RoG material no longer state that damage "decouples" from or "does not scale" with burnt area, nor that small fires remove protection. This is now presented as the review's argument (the asset lies downslope; no Alpine study relates damage to burnt area), and protection loss for decades as the finding of the few sites studied (moderate and high-severity fire). The upslope shift is described as marked after 2015, not as 72 m per decade since 2000.
- **Abstract softened (20 Nov 2026):** forest and agricultural policy "may matter as much as climate" in shaping outcomes, consistent with the hedged body text.

### 1.4 Author list for GEMS submission

From the Eurac Research staff pages and ORCID (checked 20 Nov 2026). All authors are at Eurac Research, Bolzano, Italy. **Author list and order confirmed with all co-authors (20 Nov 2026).**

| # | Author | Eurac unit | ORCID |
|---|---|---|---|
| 1 | Alexandre Dunant (corresponding) | Center for Climate Change and Transformation | 0009-0009-4377-4747 |
| 2 | Paulina Bartkowiak | Institute for Earth Observation | 0000-0002-1172-5942 |
| 3 | Alice Crespi | Center for Climate Change and Transformation | 0000-0003-4186-8474 |
| 4 | Christian Hoffmann | Institute for Regional Development | 0000-0002-9291-2570 |
| 5 | Marco Mina | Institute for Alpine Environment | 0000-0002-7489-7688 |
| 6 | Massimiliano Pittore | Center for Climate Change and Transformation | 0000-0003-4940-3444 |
| 7 | Uta Schirpke | Institute for Alpine Environment | 0000-0002-2075-6118 |
| 8 | Stefano Terzi | Center for Climate Change and Transformation | 0000-0003-0491-6772 |
| 9 | Marc Zebisch | Center for Climate Change and Transformation | 0000-0002-3530-7219 |

---

## 2. Dual-Track Redirect Strategy (If *RoG* Declines the Proposal)

Because all nine co-authors are at Eurac Research and the manuscript synthesises hazard, climate, hydrology, silviculture, risk and governance specifically for **the European Alps**, the *RoG* Editorial Board could potentially judge the scope either too regional or too management-oriented for an AGU geophysics journal. `review_paper_v7.tex` has been structured so it can be **redirected within one day** without rewriting the scientific argument or screening new literature.

### Option A (First Redirect): *Earth-Science Reviews* (Elsevier)
- **Why it fits:** Publishes comprehensive, authoritative Earth- and environmental-science syntheses, including major mountain-arc benchmarks (European Alps, Tibetan Plateau, Andes) where physical hazards intersect land-use change and remote sensing; **no strict word limit** for comprehensive reviews.
- **Switch checklist (< 1 hour):**
  1. Set `\aguformatfalse` at line 54 of `review_paper_v7.tex` (hides AGU Key Points and Plain Language Summary; optionally reuse Key Points as Elsevier *Highlights*, trimming each to $\le 85$ characters).
  2. Switch document class to `elsarticle` (`\documentclass[review,authoryear]{elsarticle}`) or submit directly in Elsevier's format-free initial submission mode (`Your Paper Your Way`), which accepts the compiled `review_paper_v7.pdf` as-is!

### Option B (Second Redirect): *Natural Hazards and Earth System Sciences (NHESS)* (EGU / Copernicus, Diamond/Gold OA)
- **Why it fits:** The manuscript is already written in the **NHESS house register** (`STYLE_SHEET.md`), and *NHESS* is the flagship international venue for Alpine multi-hazard cascades, protection forests, early warning, vulnerability and risk governance. Review articles are explicitly welcomed and undergo transparent interactive discussion.
- **Switch checklist (< 2 hours):**
  1. Set `\aguformatfalse` at line 54 of `review_paper_v7.tex` (reuse the Plain Language Summary as the EGU *Short Summary* in the submission portal).
  2. Apply the `copernicus.cls` two-column template (`\documentclass[nhess, manuscript]{copernicus}`).

---

## 3. Note on Retired GMBA Global Range Screening

The October 2026 GMBA v2 Gower-distance screening of 235 global mountain ranges (`range_screening/`, `range_selection_diagram.*`, `search_strategy.md`) was retired from this review on 18 November 2026 to eliminate scope confusion and keep the paper squarely focused on the European Alps. All scripts and outputs are preserved intact in `review/_archive/global_range_screening_2026-11/` in case the team wishes to publish the Global Fire Atlas small-fire mountain omission analysis separately.
