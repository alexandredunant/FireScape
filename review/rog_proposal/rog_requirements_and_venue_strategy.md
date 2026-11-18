# Reviews of Geophysics requirements and dual-track venue strategy

Working note, revised 18 November 2026 (aligned with `review_paper_v7.tex`).

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

| Element | *RoG* / AGU Rule | Status in `review_paper_v7.tex` (18 Nov 2026) | Action needed if invited |
|---|---|---|---|
| **Page limit & charges** | **None** at *RoG* (AGU's general ~30 PU review guideline is advisory) | 77 pages total (pp. 1–63 main + appendices; pp. 64–77 references, 159 cited sources) | Optional tightening only |
| **Abstract** | Single paragraph, **< 250 words** | **229 words** | **Ready** |
| **Key Points** | Up to **3 bullets**, each **$\le$ 140 characters**, no abbreviations | **3 bullets** (127, 124 and 127 characters; zero abbreviations), controlled by `\ifaguformat` | **Ready** |
| **Plain Language Summary** | **Required** (~150–200 words for non-specialists) | **Included** (148 words), controlled by `\ifaguformat` | **Ready** |
| **Visual synthesis** | ~5–6 figures/tables recommended | **2 TikZ figures** (`fig:cascade`, `fig:flow`) + **3 synthesis tables** (`tab:regimes`, `tab:models`, `tab:contrast`) = **5 visual items** | Optional map of Alpine Convention pyroregions if desired |
| **Template** | AGU `agujournal2019` LaTeX class | Clean standard `article` class | Convert preamble to `agujournal2019` upon invitation |

---

## 2. Dual-Track Redirect Strategy (If *RoG* Declines the Proposal)

Because all ten co-authors are at Eurac Research and the manuscript synthesises hazard, climate, hydrology, silviculture, risk and governance specifically for **the European Alps**, the *RoG* Editorial Board could potentially judge the scope either too regional or too management-oriented for an AGU geophysics journal. `review_paper_v7.tex` has been structured so it can be **redirected within one day** without rewriting the scientific argument or screening new literature.

### Option A (First Redirect): *Earth-Science Reviews* (Elsevier)
- **Why it fits:** Publishes comprehensive, authoritative Earth- and environmental-science syntheses, including major mountain-arc benchmarks (European Alps, Tibetan Plateau, Andes) where physical hazards intersect land-use change and remote sensing; **no strict word limit** for comprehensive reviews.
- **Switch checklist (< 1 hour):**
  1. Set `\aguformatfalse` at line 53 of `review_paper_v7.tex` (hides AGU Key Points and Plain Language Summary; optionally reuse Key Points as Elsevier *Highlights*, trimming each to $\le 85$ characters).
  2. Switch document class to `elsarticle` (`\documentclass[review,authoryear]{elsarticle}`) or submit directly in Elsevier's format-free initial submission mode (`Your Paper Your Way`), which accepts the compiled `review_paper_v7.pdf` as-is!

### Option B (Second Redirect): *Natural Hazards and Earth System Sciences (NHESS)* (EGU / Copernicus, Diamond/Gold OA)
- **Why it fits:** The manuscript is already written in the **NHESS house register** (`STYLE_SHEET.md`), and *NHESS* is the flagship international venue for Alpine multi-hazard cascades, protection forests, early warning, vulnerability and risk governance. Review articles are explicitly welcomed and undergo transparent interactive discussion.
- **Switch checklist (< 2 hours):**
  1. Set `\aguformatfalse` at line 53 of `review_paper_v7.tex` (reuse the Plain Language Summary as the EGU *Short Summary* in the submission portal).
  2. Apply the `copernicus.cls` two-column template (`\documentclass[nhess, manuscript]{copernicus}`).

---

## 3. Note on Retired GMBA Global Range Screening

The October 2026 GMBA v2 Gower-distance screening of 235 global mountain ranges (`range_screening/`, `range_selection_diagram.*`, `search_strategy.md`) was retired from this review on 18 November 2026 to eliminate scope confusion and keep the paper squarely focused on the European Alps. All scripts and outputs are preserved intact in `review/_archive/global_range_screening_2026-11/` in case the team wishes to publish the Global Fire Atlas small-fire mountain omission analysis separately.
