# Publishing the citable release

This clean repository is intended to replace the public-facing contents at `https://github.com/alexandredunant/FireScape` without deleting the legacy research history.

## Before publishing

- Confirm the author list, affiliations, release date, and license in `CITATION.cff`.
- Confirm that the repository may be made public and that every figure is cleared for redistribution.
- Run `make report`, `make check`, and `python -m py_compile src/*.py scripts/*.py`.
- Preserve the old GitHub default branch as `archive/legacy-firescape` before changing `main`.

Replacing a default branch is a repository-administration operation. Do not force-push until the archive branch is visible on GitHub and collaborators have been notified.

## Release and DOI

1. Publish the clean `main` branch and make the repository public.
2. Enable `alexandredunant/FireScape` in the Zenodo GitHub integration.
3. Create the GitHub tag and release `v1.0.0`.
4. Wait for Zenodo to ingest the release and assign a DOI.
5. Add the version DOI to `CITATION.cff`, `CITATION.bib`, and `README.md`, then publish a metadata-only patch release if needed.

GitHub reads a root-level `CITATION.cff` and exposes a “Cite this repository” control. Zenodo can read the same metadata when archiving a GitHub release.
