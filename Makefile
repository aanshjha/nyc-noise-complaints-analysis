PYTHON ?= python3
QUARTO ?= quarto

.PHONY: test render refresh-data clean

test:
	$(PYTHON) -m pytest -q

render:
	$(QUARTO) render midtermnew.qmd --to pdf

refresh-data:
	$(PYTHON) scripts/download_data.py

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__ midtermnew_files
