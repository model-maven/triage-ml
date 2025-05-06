.PHONY: install data train test lint serve docker drift clean

install:
	pip install -e ".[dev]"

data:
	python scripts/generate_dataset.py

train:
	python -m triageml.train_baseline

train-transformer:
	pip install -e ".[transformer]" && python -m triageml.train_transformer

test:
	pytest --cov=triageml --cov-report=term-missing

lint:
	ruff check src tests scripts

serve:
	uvicorn triageml.api.main:app --reload

docker:
	docker compose up --build

drift:
	python scripts/check_drift.py

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .coverage
