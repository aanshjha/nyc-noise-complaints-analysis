# NYC Noise Complaints Analysis

Midterm analysis of NYPD-handled NYC 311 noise complaints created from **June
30 through July 6, 2024**. The project studies closure times, compares groups,
and builds a leakage-safe logistic-regression baseline for whether a complaint
is closed in two hours or more.

## Submission files

- `midtermnew.qmd` - executable Quarto source
- `midtermnew.pdf` - rendered report for grading
- `src/analysis.py` - reusable cleaning, testing, and modeling functions
- `tests/test_analysis.py` - deterministic unit tests
- `data/raw/` - committed source snapshot used by the report
- `scripts/download_data.py` - optional refresh from NYC Open Data

## Reproduce the project

Python 3.11+ and Quarto 1.4+ are recommended. A TeX engine is required for PDF
output; Quarto can install TinyTeX with `quarto install tinytex` if needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
quarto render midtermnew.qmd --to pdf
```

Or, after activating the environment:

```bash
make test
make render
```

The report uses the committed snapshot by default, so rendering does not need
network access. To refresh the same date window from the live API:

```bash
python scripts/download_data.py
```

NYC Open Data is revised over time, so a refreshed file may not exactly match
the committed snapshot or the submitted PDF.

## Data source

[311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/d/erm2-nwe9),
provided by NYC 311 through NYC Open Data. The download is filtered to NYPD
records whose complaint type begins with `Noise` and whose creation timestamp
falls in the stated seven-day window.

## Method notes

- Invalid or missing creation/closure dates, duplicate unique keys, and negative
  closure intervals are excluded with counts reported in the PDF.
- Response-time comparisons use Kruskal-Wallis tests because the outcome is
  strongly right-skewed.
- Chi-square tests include Cramer's V effect sizes.
- Model features are restricted to information available when a complaint is
  created; preprocessing is fitted only on the training split.

## Academic integrity

The report states the data source, assumptions, exclusions, methods, and
limitations. Review your course's submission rules before uploading.
