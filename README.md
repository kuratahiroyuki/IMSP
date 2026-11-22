# SCI-IMSP: AIS grade prediction after spinal cord injury

Logistic regression models based on individual motor and sensory predictors (IMSP models) for binary prediction of AIS grade six months after SCI using measurements obtained two weeks after injury. AIS grades A–C are encoded as 0 and grade D as 1.

## Installation

```bash
conda env create -f environment.yml
conda activate sci-imsp
```

Alternatively, use a Python environment and `python -m pip install -r requirements.txt`. The dependency versions reflect the packaging/test environment, not a verified record of the original study environment.

## Quick start (synthetic data)

```bash
python data/generate_example.py
python l1_lr_calibration_1.py --input data/example.xlsx --preselect both --out_dir result/example --n_time 1 --bootstrap_samples 20 --n_splits 3
```

Synthetic data are only for software testing, not for demonstrating clinical predictive accuracy.

## Study analysis

Place the authorized workbook at `data/data_SCI_eng.xlsx` (not included). Run all six representations from the repository root:

```bash
bash main.sh data/data_SCI_eng.xlsx result
```

Or run one strategy:

```bash
python l1_lr_calibration_1.py --input data/data_SCI_eng.xlsx --preselect both --out_dir result/both --n_time 50 --bootstrap_samples 10000 --calibration_bins 10 --n_splits 3
```

Choices for `--preselect`: `both`, `left`, `right`, `mean`, `smaller`, `larger`. Use `--help` for other arguments. The shell script uses 50 repetitions and 10,000 bootstrap samples by default, matching the supplied shell script. For a short smoke run, set `N_TIME=1 BOOTSTRAP_SAMPLES=20` before invoking `bash main.sh`.

## Analysis workflow

The supplied Python analysis files are unchanged. The main program uses 20 motor and 112 sensory measurements, bilateral preselection, domain grouping, absolute Pearson correlation filtering (threshold 0.85), median imputation, standardization, L1 logistic regression feature selection (three predictors), and logistic regression classification. Preprocessing and feature selection are fitted within nested stratified cross-validation. The default starting random seed is 42.

**Implementation detail:** Domain grouping in the supplied code recognizes `TR_`, `TL_`, `AR_`, and `AL_` prefixes. It retains one representative per group. Aggregated representations use `T_` and `A_` prefixes, which are not grouped at this stage. Thus the expected counts after domain grouping are 48 for both, 38 for left/right, and 66 for mean/smaller/larger. This behavior is preserved, not silently changed.

## Outputs

Each representation has its own results directory. Key outputs include `all_fold_results.csv`, `repeat_metrics.csv`, `metrics_mean_95CI.csv`, `all_oof_predictions.csv`, `feature_selection_frequency_overall.csv`, `calibration_metrics_mean_95CI.csv`, and calibration figures. Per-repetition outputs are stored in numbered subdirectories. Clinical outputs may contain sensitive patient-level predictions; do not publish them without authorization.

## Repository contents

- `l1_lr_calibration_1.py`: supplied main analysis program.
- `preselect_predictor.py`: supplied bilateral predictor helper; imported by the main program. Its standalone demonstration is not the study analysis entry point (its example outcome coding differs from the main program, and it prints dataset rows).
- `main.sh`: launcher corrected to reference the supplied Python filename.
- `data/`: input schema documentation and synthetic data generator.
- `tests/`: preprocessing smoke tests.

Run preprocessing tests with `python -m unittest discover -s tests -v`.

## Data availability and license

The supplied clinical workbook is excluded from this public-code package. No claim of public access to patient data is made. See `data/README.md` for input requirements. The original MIT license is retained in `LICENSE`. Add the verified manuscript citation and repository DOI when available.
