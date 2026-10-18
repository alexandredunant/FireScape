# FireScape Alps Review: folder guide

Review paper: *What Makes Mountain Wildfire Different? Evidence from the European Alps and Other Ranges* (Eurac Research). Folder tidied on 7 October 2026.

## Current working set (this folder)
- `review_paper_v5.tex` and `review_paper_v5.pdf`: the current manuscript (v26 of the drafting history). Compile with pdflatex, then bibtex, then pdflatex twice.
- `FireScape_master_v6.bib`: the bibliography the manuscript compiles against. It must stay next to the `.tex` file.
- `STYLE_SHEET.md`: the writing rules for the manuscript.

## Subfolders
- `papers/`: the PDF of every cited source, named `FirstAuthor_Year_First_five_title_words.pdf`, plus `FireScape_review_reference_list.docx/.csv`, which maps each citation key to its DOI and file.
- `supplementary/`: material described in Appendix A of the paper.
  - `FireScape_Alps_screening_register.xlsx`: every screened record with its decision and reason.
  - `FireScape_evidence_notes.md`: a structured note for each cited source.
  - `FireScape.bib`: the initial set of 49 publications from Step 2.
- `contents/`: a stand-alone table of contents (`review_paper_contents.tex/.toc/.pdf`). To regenerate it, see the project status notes.
- `_archive/`: superseded material, kept for reference.
  - Earlier drafts: `review_paper.tex`, and `review_paper_v2` to `v4`.
  - The parallel revision of 7 October: `review_paper_v5_parallel_0953.tex`.
  - Earlier bibliographies: `FireScape_master.bib`, and `FireScape_master_v2` to `v5`.
  - `references.bib`: most entries failed verification, so do not cite from it.
  - `PDF_WANTLIST.md`: outdated.
  - `tmp/`: working files, including the full-text source audit of 7 October.
- `_to_delete/`: exact duplicates and failed downloads, safe to delete.
  - `review_paper_v5_alt`: identical to the current manuscript.
  - `Claude outputs/`: an outdated PDF copy.
  - `_to_review/`: four 3 KB failed downloads, and duplicate copies of the Gerberding 2025, Reilly 2022 and Conedera 2024 PDFs.
