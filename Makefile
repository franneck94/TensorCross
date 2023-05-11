install:
	pip install -e .

doc:
	mkdocs gh-deploy

test:
	pytest tests/ --disable-pytest-warnings

coverage:
	pytest tests/ --cov=tensorcross --disable-pytest-warnings

check:
	pre-commit run --all-files
