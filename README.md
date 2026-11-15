# Assistive Technologies on Safety Monitoring for Community-Dwelling Persons Living with Dementia — analysis code

Analysis code for the participant characteristics (Table 1), the sleep and in-home sensor comparison (Table 2), the participant-level prediction models (Table 3), and Figures 1–3 of the manuscript.

No participant records or private model state are distributed. The data are available upon reasonable request to the corresponding author.

## Scope

- Table 1 uses Welch's t-test for continuous variables and the chi-square test for categorical variables, or Fisher's exact test where any expected cell count is below five.
- Table 2 accounts for the clustering of days within dyads using linear mixed models with a participant random intercept, falls back to generalized estimating equations with an exchangeable working correlation when a mixed model does not converge, and applies the Benjamini–Hochberg correction across the 21 primary P values.
- Participant-level models use penalized logistic regression with an L1 or L2 penalty selected in an inner cross-validation loop, stratified group k-fold cross-validation grouped by dyad, and out-of-fold predictions averaged over two independent partitions. Uncertainty is estimated by bootstrap resampling of dyads.
- Continuous variables are scaled by min-max normalization, and missing values are imputed with the training-fold median; all transformations are fitted on the training folds only.
- Discrimination is tested against matched permutation null distributions (2,000 permutations), in which the full model-selection procedure is re-run.
- Classification thresholds are selected on the pooled out-of-fold predictions; the threshold-dependent metrics are descriptive.
- Feature contributions are examined using SHAP values computed on the out-of-fold predictions.

Supplementary analyses are not included.

## Run

Install the packages in `requirements.txt`. Restricted input contracts are documented in `INPUT_SCHEMA.md`; required fields are defined in `DATA_DICTIONARY.csv`. Use the approved 67-dyad input only. Both input and output directories must be outside this package; `code/config.py` refuses to run otherwise.

Set the private input and work directories, then run the scripts in order from the `code` directory:

```powershell
$env:PASS_PRIVATE_DATA = "D:/restricted/PASS/inputs"
$env:PASS_WORK_DIR = "D:/restricted/PASS/analysis"
cd code
python table1.py              # Table 1
python table2.py              # Table 2
python participant_models.py  # Table 3, bootstrap CIs, and 2,000 matched permutations
python figures.py             # Figures 1–3 (requires participant_models.py output)
```

Results are written to `results/`, `state/`, and `figures/` inside the work directory. The permutation count defaults to 2,000; set `$env:PASS_PERMUTATIONS` to override it, for example for a quick check.

The figure script reads only state generated in the selected work directory. Generated figures and state include participant-level information and must remain private.

## License

The code is released under the MIT License (see `LICENSE`). The license covers the code and documentation only; it does not grant access to the restricted study data.
