SRC_CORE=tensorcross
SRC_TEST=tests
SRC_DOC=docs

ifeq ($(OS), Windows_NT)
	BUILD_DOC=cd $(SRC_DOC) && build_docs.bat
else
	BUILD_DOC=cd $(SRC_DOC) && ./build_docs.sh
endif

help:
	@echo "Some available commands:"
	@echo " * test         		- Run unit tests."
	@echo " * test-coverage  	- Run unit tests and test coverage."
	@echo " * doc          		- Document code (pydoc)."

test:
	@pytest $(SRC_TEST)

test-coverage:
	@pytest --cov=$(SRC_CORE) $(SRC_TEST)
	@codecov

doc:
	@$(BUILD_DOC)
