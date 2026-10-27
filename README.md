# Mediastinitis after heart transplantation vs. standard cardiac surgery — analysis code

Analysis pipeline for a single-centre retrospective cohort comparing the microbiology
and outcomes of post-sternotomy mediastinitis in heart-transplant recipients and in
patients undergoing other cardiac surgery with cardiopulmonary bypass.

**This repository contains code only.** No patient-level data — raw, de-identified or
otherwise — are distributed here. The scripts are published to document the methods
(data cleaning rules, endpoint definitions, statistical models) and are reproducible by
anyone holding a dataset in the format described by the data dictionary produced by
`deid.py`.

## Pipeline

Python 3.12 — `pip install -r requirements.txt`

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `python clean.py crf.xlsx` | raw CRF (Excel) | `clean_full.pkl` (identifiable, internal use only) |
| 2 | `python deid.py` | `clean_full.pkl` | `mediastinitis_deidentified.csv`, `mediastinitis_data_dictionary.csv` |
| 3 | `python analysis.py` | de-identified CSV | `results.json`, `ps.pkl` |
| 4 | `python tables.py` | `results.json` | `tables_report.md` (Tables 1–3, Supplementary Tables 1–6) |
| 5 | `python flowchart.py` | de-identified CSV + counts set in `FLOW` | `fig1_flowchart.png` |
| 6 | `python figures.py` | `results.json`, de-identified CSV | `fig2_main.png`, `sfig1–3.png` |

Steps 3–6 read only the de-identified CSV and never touch the source CRF.

## Methods summary

- Microbiology coded as presence/absence per organism from structured CRF columns and
  free text (regular expressions in `clean.py`); fungal infection and ESBL-producing
  Enterobacterales are the primary endpoints.
- Group comparisons: chi-square / Fisher exact test, Mann–Whitney U test.
- Unadjusted odds ratios with Haldane–Anscombe correction when a cell is empty;
  Benjamini–Hochberg q values computed for the univariate microbiology family.
- Multivariable logistic regression (heart-transplant status adjusted for age and
  SAPS II; sensitivity model adding antibiotics within 7 days before diagnosis) on
  20 multiply-imputed datasets (chained equations, Bayesian ridge), Rubin's rules.
- Propensity score (age, sex, SAPS II) with overlap weights; balance assessed with
  standardised mean differences; effective sample size reported.

## Data availability

Individual patient data cannot be shared publicly (French health-data regulation).
Requests for data access should be addressed to the corresponding author.

## License

MIT — see `LICENSE`.
