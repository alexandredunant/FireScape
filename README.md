# FireScape Forestry Report

Reproducibility package for the technical report **“Suscettibilità agli incendi boschivi su scala stagionale in Alto Adige”** (Seasonal wildfire susceptibility in South Tyrol).

The report presents seasonal susceptibility maps at 250 m resolution. An Explainable Boosting Machine (EBM), trained on wildfire occurrences, sampled pseudo-absences, and environmental covariates for 1999–2024, was applied to daily meteorological inputs for 2015–2024. Daily scores were then aggregated by season. Because the training design does not reflect population prevalence, the outputs are relative susceptibility scores—not calibrated probabilities of fire occurrence.

## Read and cite

- [Final report (PDF)](report/FireScape_forestry_report.pdf)
- [Version 1.0.0 release](https://github.com/alexandredunant/FireScape/releases/tag/v1.0.0)
- [LaTeX source](report/report.tex)
- [Citation metadata](CITATION.cff)
- [BibTeX entry](CITATION.bib)
- [Release-file checksums](SHA256SUMS)

Please cite [version 1.0.0](https://github.com/alexandredunant/FireScape/releases/tag/v1.0.0) rather than the moving `main` branch. A Zenodo DOI will supersede the release URL once archival is enabled.

## Repository contents

```text
.
├── report/                 # Final PDF, LaTeX source, and report figures
├── src/                    # Ordered analysis workflow
│   ├── 01_prepare_training_data.py
│   ├── 02_train_model.py
│   ├── 03_generate_seasonal_risk.py
│   └── validate_inputs.py
├── data/README.md          # Required inputs, expected layout, and access limits
├── archive/                # Superseded report drafts and legacy tests
├── CITATION.cff
├── CITATION.bib
├── environment.yml
└── Makefile
```

The numbered scripts document the pathway used to create the results. They replace machine-specific absolute paths with a portable project-root setting.

## Reproduce the analysis

Create the software environment:

```bash
conda env create -f environment.yml
conda activate firescape-forestry
```

Place the inputs described in [`data/README.md`](data/README.md) under `data/`, or point to an existing FireScape data tree:

```bash
export FIRESCAPE_ROOT=/path/to/firescape-data-root
python src/validate_inputs.py
python src/01_prepare_training_data.py
python src/02_train_model.py 250m
python src/03_generate_seasonal_risk.py
```

The complete numerical analysis cannot be rerun from the GitHub checkout alone because some wildfire, meteorological, and administrative inputs are not redistributed. The included report, figures, scripts, software specification, and input manifest make that boundary explicit.

To rebuild only the PDF:

```bash
make report
```

## Versioning and preservation

Release `v1.0.0` is public and contains the final PDF plus its checksum manifest. The repository state that preceded this refocus is preserved on [`archive/legacy-firescape`](https://github.com/alexandredunant/FireScape/tree/archive/legacy-firescape).

To add a DOI, enable this repository in the Zenodo GitHub integration. After Zenodo ingests the release, add the assigned version DOI to `CITATION.cff`, `CITATION.bib`, and this README in a metadata-only follow-up release.

## License

Code and original documentation are released under the [MIT License](LICENSE). Input datasets and third-party basemaps remain governed by their providers’ terms; see [`data/README.md`](data/README.md).

## Contact

Alexandre Dunant — alexandre.dunant@eurac.edu
