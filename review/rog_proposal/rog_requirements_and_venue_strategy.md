# Reviews of Geophysics requirements and dual-track venue strategy

Working note, revised 20 November 2026 (aligned with the shortened, evidence-graded `review_paper_v7.tex` and `review_paper_v7_SI.tex`).

---

## 1. Requirements of *Reviews of Geophysics* (Primary Target)

### 1.1 Proposal and timeline

*Reviews of Geophysics* (RoG) is **invitation-only**:

1. **Send a brief proposal** (`proposal.md` + `cover_letter.md`) to **reviewsgeophysics@agu.org** containing:
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
| **Abstract** | Single paragraph, **< 250 words** | **244 words** | **Ready** (review the "rival climate" sentence, see §1.3) |
| **Key Points** | Up to **3 bullets**, each **$\le$ 140 characters**, no abbreviations | **3 bullets** (127, 124 and 127 characters; zero abbreviations), controlled by `\ifaguformat` | **Ready** |
| **Plain Language Summary** | **Required** (~150–200 words for non-specialists) | **Included** (146 words), controlled by `\ifaguformat` | **Ready** |
| **Visual synthesis** | ~5–6 figures/tables recommended | **2 TikZ figures** (`fig:cascade`, `fig:flow`) + **3 synthesis tables** (`tab:regimes`, `tab:models`, `tab:contrast`) = **5 visual items** | Optional map of Alpine Convention pyroregions if desired |
| **Template** | AGU `agujournal2019` LaTeX class | Clean standard `article` class | Convert preamble to `agujournal2019` upon invitation |
| **Supporting Information** | Separate file using the AGU SI template (Text S1, S2…) | `review_paper_v7_SI.tex`: Text S1 search protocol and screening register; Text S2 glossary. Section numbers refer to the main text | Transfer to the AGU SI template; upload the screening register (`supplementary/FireScape_Alps_screening_register.xlsx`) as Data Set S1 |

### 1.3 Changes since the 18 November version

- **Shortened to ~13,500 words of main text** (from ~29,900) and reorganised as an evidence-graded synthesis. Each section opens with the trends supported by at least three independent studies. Findings from two studies are reported as such, and single-study results are kept only where they are the sole Alpine evidence for a mechanism, hedged and with their case count stated. See `plan/shortening_plan_v7.md`.
- **Discussion** now opens with the five findings that recur across independent studies, and the imported-tools contrast table (*Table 3*) follows as their consequence.
- **Corrections verified against the source PDFs:** the Gelabert et al. (2025) fuel-moisture ranking; Cardil et al. (2025) now marked as Californian (imported) evidence; the Moos et al. (2021) per-hectare value introduced in the economics section; and the Avouris et al. (2025) preprint result hedged.
- **136 sources cited** (23 removed, mainly single non-Alpine examples). The search protocol and glossary moved to Supporting Information.
- **Open point for the author team:** the abstract states that forest and agricultural policy "rival climate in shaping outcomes", whereas the body now presents the one continental modelling study behind the management-versus-climate comparison as suggestive. Decide whether to soften the abstract or keep the stronger claim, which the abstract derives from the protective-service argument.

### 1.4 Author list for GEMS submission

From the Eurac Research staff pages and ORCID (checked 20 Nov 2026). All authors are at Eurac Research, Bolzano, Italy.

| # | Author | Eurac unit | ORCID |
|---|---|---|---|
| 1 | Alexandre Dunant (corresponding) | Center for Climate Change and Transformation | *to confirm*: 0000-0002-8961-5292 (Durham/GNS history) or 0009-0009-4377-4747 (Eurac) |
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
