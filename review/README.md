# FireScape Alps Review: folder guide

Review paper: *Wildfires in the European Alps: An Interdisciplinary Review* (Eurac Research). Updated on 20 November 2026 (`review_paper_v7`, shortened to ≈13.5k words of main text around multi-study trends).

## Current working set (this folder)
- `review_paper_v7.tex` and `review_paper_v7.pdf`: the current manuscript (`\aguformattrue` toggle at the top controls AGU Key Points and Plain Language Summary for *Reviews of Geophysics* vs. *Earth-Science Reviews* / *NHESS* export). Compile inside `review/` with `pdflatex`, `bibtex`, and `pdflatex` twice.
- `review_paper_v7_SI.tex`: Supporting Information (Text S1 search protocol and screening register; Text S2 glossary). Compiles on its own against the same `.bib`.
- `FireScape_master_v7.bib`: the Crossref-audited bibliography (138 sources cited in the main text; the `.bib` also holds entries no longer cited) that `review_paper_v7.tex` compiles against. Keep next to the `.tex` file.
- `STYLE_SHEET.md`: writing and register conventions for the manuscript.

## Subfolders
- `rog_proposal/`: pre-submission inquiry package (`proposal.md`, `RoG_proposal.pdf`, `cover_letter.md/.pdf/.docx`, `rog_requirements_and_venue_strategy.md/.pdf`, `novelty_check.md`) aligned with `review_paper_v7` for *Reviews of Geophysics*, with dual-track redirect notes for *Earth-Science Reviews* and *NHESS*.
- `papers/`: PDFs of cited sources (`FirstAuthor_Year_First_five_title_words.pdf`, shared via Zotero) plus `FireScape_review_reference_list.docx/.csv` mapping citation keys to DOIs.
- `supplementary/`: material described in Supporting Information Text S1 (`FireScape_Alps_screening_register.xlsx`, `FireScape_evidence_notes.md`, `FireScape.bib`).
- `contents/`: standalone table of contents (`review_paper_contents.tex/.toc/.pdf`), synchronised with `review_paper_v7`.
- `_archive/`: superseded material kept for reference:
  - Earlier drafts (`review_paper.tex`, `review_paper_v2` to `v6`) and bibliographies (`FireScape_master.bib`, `v2` to `v6`).
  - `global_range_screening_2026-11/`: retired GMBA v2 global mountain range screening scripts, outputs, diagram, and search strategy.
  - `audit_2026-10/` and `tmp/`: full-text citation and metadata audit working files from October 2026.
