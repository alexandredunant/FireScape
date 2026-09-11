# Publishing the citable release

The focused repository was published at `https://github.com/alexandredunant/FireScape`. The previous default-branch state is preserved at `archive/legacy-firescape`, and release `v1.0.0` was published on 11 September 2026.

## Checks completed for v1.0.0

- The report built successfully from its LaTeX source.
- Repository integrity, Python syntax, local links, checksums, and file sizes were checked.
- The old GitHub default branch was preserved as `archive/legacy-firescape` before `main` changed.
- The final PDF and `SHA256SUMS` were attached to the GitHub release.

## Remaining DOI step

1. Enable `alexandredunant/FireScape` in the Zenodo GitHub integration.
2. Ask Zenodo to ingest release `v1.0.0` and assign a DOI.
3. Add the version DOI to `CITATION.cff`, `CITATION.bib`, and `README.md`, then publish a metadata-only patch release.

GitHub reads a root-level `CITATION.cff` and exposes a “Cite this repository” control. Zenodo can read the same metadata when archiving a GitHub release.
