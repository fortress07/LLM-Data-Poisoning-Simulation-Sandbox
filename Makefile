PYTHON ?= python
export PYTHONPATH := src

.PHONY: help setup test test-node test-all native bench demo sweep audit study figures check-figures clean lint

help:
	@echo "make setup      install the package in editable mode"
	@echo "make test       run the python test suite"
	@echo "make test-node  run the viewer test suite"
	@echo "make test-all   run both suites"
	@echo "make native     build the C kernels"
	@echo "make bench      compare the C and python kernels"
	@echo "make demo       run one campaign and render the html report"
	@echo "make sweep      run a dose response sweep"
	@echo "make audit      triage a corpus with no ground truth"
	@echo "make study      reproduce every number in docs/RESULTS.md"
	@echo "make figures    regenerate the README charts from the study json"
	@echo "make check-figures  verify every figure fits its canvas"
	@echo "make clean      remove build and run artefacts"

setup:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

test-node:
	cd viewer && node --test

test-all: test test-node

native:
	$(MAKE) -C native/poisonscan

bench:
	$(PYTHON) -m poisonlab benchmark --size 20000

demo:
	$(PYTHON) -m poisonlab run configs/backdoor.toml --html

sweep:
	$(PYTHON) -m poisonlab sweep configs/backdoor.toml \
		--axis attack.poison_rate=0.002,0.005,0.01,0.02,0.05 \
		--seeds 5 --html

study:
	$(PYTHON) scripts/experiments.py --seeds 6 --size 6000

figures:
	$(PYTHON) scripts/figures.py
	$(PYTHON) scripts/check_figures.py

check-figures:
	$(PYTHON) scripts/check_figures.py

audit:
	$(PYTHON) -m poisonlab data --out corpus.jsonl --set data.size=2000
	$(PYTHON) -m poisonlab forge --in corpus.jsonl --out suspicious.jsonl --set attack.poison_rate=0.02
	$(PYTHON) -m poisonlab audit --in suspicious.jsonl --quiet

lint:
	$(PYTHON) -m compileall -q src scripts tests

clean:
	rm -rf runs build dist *.egg-info .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf src/poisonlab/accel/_bin
