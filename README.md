# Predicting smoking lapse from EMA without leakage

Code for the paper:

> Modiridovom, E., Akopian, D., & Chalela, P. *Predicting smoking lapse from ecological momentary assessment without leakage: a benchmark of 24 machine-learning configurations, fixed-horizon survival risk and cross-study transfer.* (under review)

The code benchmarks 12 model families (logistic regression, random forest, XGBoost, SVM, LSTM, GRU, GRU-D, TCN, GRU-TCN, Transformer, and self-supervised GRU and GRU-D) with two feature sets on 37,002 EMA entries from 791 Smoke Free app users. Evaluation is leakage-free throughout:

- participant-level 10-fold cross-validation (KFold, seed 42);
- nested tuning on 20% of each training fold's participants;
- standardisation, tuning, early stopping and self-supervised pretraining on training participants only;
- features that use the current and earlier entries only.

It also fits discrete-time survival models, replicates the main checks in two prompted EMA studies (EMASENS and an alcohol use disorder study), and tests cross-study transfer to small new studies.

## Repository layout

| Path | What it does | Paper |
|---|---|---|
| `notebooks/01_benchmark_24_configurations.ipynb` | The 24-configuration benchmark with nested tuning; per-entry predictions; per-participant AUC; self-supervised early-stopping check | Tables 6–7, Figures 2–3, Supplements S1, S3, S4, Appendix A |
| `notebooks/02_leakage_check_and_survival.ipynb` | Entry-level vs participant-level splits; discrete-time survival models with three encoder strategies | Tables 5 and 8, Figure 4 |
| `notebooks/03_replication_EMASENS.ipynb` | Replication in EMASENS (leave-one-participant-out) | Table 9, Figures 5 and 8 |
| `notebooks/04_replication_alcohol.ipynb` | Replication in the alcohol study (participant-level 10-fold) | Table 9, Figures 5 and 8 |
| `notebooks/05_cross_study_transfer.ipynb` | Transfer between studies, pooling, partial pooling, small target studies, Table 10 intervals | Table 10, Figure 6, Supplement S2 |
| `scripts/make_figures.py` | Redraws all eight figures from the reported values (no data needed) | Figures 1–8 |

## Setup

```bash
git clone https://github.com/<your-account>/lapse-prediction-benchmark.git
cd lapse-prediction-benchmark
pip install -r requirements.txt
```

The analyses used Python 3.13 with NumPy 2.1, pandas 2.2, SciPy 1.16, scikit-learn 1.6, XGBoost 3.4 and PyTorch 2.9. A GPU is recommended for notebooks 01 and 02; the other notebooks run on a CPU.

## Data

The three datasets are public, but they are not redistributed here. See [`data/README.md`](data/README.md) for where to download each one and which file names the notebooks expect.

## Running

1. Download the data into `data/` (or next to the notebooks).
2. Open a notebook, set `SAVE_DIR` in its first code cell, and run once with `QUICK_TEST = True`. This is a fast check on a few participants.
3. Set `QUICK_TEST = False` and *Run all*. Results, per-entry predictions and fold assignments are written to `SAVE_DIR`. Finished folds and configurations are skipped after a restart.

Run notebook 01 before 02: notebook 02 reads the benchmark folder for the survival comparison. Notebooks 03–05 are independent.

On Google Colab, mount Drive and point `SAVE_DIR` to a Drive folder. The commented lines in each first cell show how.

## Reproducing the figures only

```bash
python scripts/make_figures.py      # writes PNG and PDF files to ./paper_figures
```

## Notes

- Notebook 05 also contains exploratory analyses (Parts 5–7, 10, 11) that are not reported in the paper; its first cell lists which parts are.
- Random seeds are fixed per fold. Deep-model results can still differ slightly between GPU and CPU or between library versions.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). Please also cite the original datasets:

- Smoke Free: Perski et al. (2023), *Nicotine & Tobacco Research*, 25(7), 1330–1339.
- EMASENS: Perski et al. (2024), *PLOS Digital Health*, 3(8), e0000594.
- Alcohol: Wyant et al. (2024), *Journal of Psychopathology and Clinical Science*, 133(7), 527–540.

## Funding

Supported in part by the Cancer Prevention & Research Institute of Texas (CPRIT), grant RP230117.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences.
