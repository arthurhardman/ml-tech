.PHONY: install test run run-fast run-quick lint

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run:
	python run_pipeline.py

run-fast:
	python run_pipeline.py --no-grid --no-shap

run-quick:
	python run_pipeline.py --no-grid --no-shap --nrows 10000

lint:
	python -m py_compile src/load_data.py src/preprocessing.py src/modeling.py src/evaluation.py run_pipeline.py tests/test_preprocessing.py tests/test_modeling.py && echo "OK"
