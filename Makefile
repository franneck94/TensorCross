SRC_CORE=tensorcross
SRC_TEST=tests
SRC_DOC=docs

ifeq ($(OS), Windows_NT)
	PYTHON=python
	BUILD_DOC=cd $(SRC_DOC) && build_docs.bat
else
	PYTHON=python3
	BUILD_DOC=cd $(SRC_DOC) && ./build_docs.sh
endif

help:
	@echo "Some available commands:"
	@echo " * test         		- Run unit tests."
	@echo " * test-coverage  	- Run unit tests and test coverage."
	@echo " * test-coverage-html- Run unit tests and test coverage (html)."
	@echo " * doc          		- Document code (pydoc)."

install:
	@$(PYTHON) setup.py install

test:
	@$(PYTHON) -m pytest $(SRC_TEST)

test-coverage:
	@$(PYTHON) -m pytest --cov=$(SRC_CORE) $(SRC_TEST)
	@$(PYTHON) -m codecov

test-coverage-html:
	@$(PYTHON) -m pytest --cov=$(SRC_CORE) $(SRC_TEST) --cov-report=html

doc:
	@$(BUILD_DOC)
