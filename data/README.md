# Data

No data are included in this repository. Download each dataset from its source and place the files in this folder (or next to the notebooks).

| Study | Source | Files the notebooks expect | Licence |
|---|---|---|---|
| Smoke Free (Perski et al., 2023) | https://doi.org/10.5281/zenodo.8423920 | `smokefree_data.xlsx` (sheet `All_Data`) | CC BY 4.0 |
| EMASENS (Perski et al., 2024) | https://doi.org/10.5281/zenodo.12532026 | `analytic_sample_1.csv` (converted from `analytic_sample_1.rds`, see below) | CC BY 4.0 |
| Alcohol use disorder (Wyant et al., 2024) | https://osf.io/w5h9y/ | `ema.csv`, `baseline_features_1week_v3.csv` | See the OSF project |

## Converting the EMASENS file to CSV

The EMASENS data are distributed as an R data file. Convert it once, in R:

```r
x <- readRDS("analytic_sample_1.rds")
write.csv(x, "analytic_sample_1.csv", row.names = FALSE)
```

or in Python:

```python
import pyreadr
pyreadr.read_r("analytic_sample_1.rds")[None].to_csv("analytic_sample_1.csv", index=False)
```

## Notes on the analysed samples

- **Smoke Free:** participants with at least 20 entries (791 participants, 37,002 entries).
- **EMASENS:** answered prompts only (4,722 of 6,124 rows). Missed prompts carry imputed outcomes in the published file and are excluded.
- **Alcohol:** the authors' analytic sample of 151 participants (18 IDs excluded, as listed in notebook 04). EMAs are kept within each participant's study period.
