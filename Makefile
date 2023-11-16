.PHONY: install clean lint style format test build publish publish-test

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = energy-forecat-benchmark-toolkit
PACKAGE_NAME = enfobench
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create python virtual environment
venv/bin/python:
	( \
		$(PYTHON_INTERPRETER) -m venv $(PROJECT_DIR)/venv; \
		source $(PROJECT_DIR)/venv/bin/activate; \
		pip install --upgrade pip; \
	)

## Install project dependencies
install: venv/bin/python
	(\
		source $(PROJECT_DIR)/venv/bin/activate; \
		pip install -e .; \
    )

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff, mypy, black, and isort
lint:
	hatch run lint:all


## Check style using ruff, black, and isort
style:
	hatch run lint:style

## Format using black
format:
	hatch run lint:fmt

## Run pytest with coverage
test:
	hatch run cov

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Build source distribution and wheel
build: style
	hatch build

## Upload source distribution and wheel to PyPI
publish: build
	hatch publish --repo main

## Upload source distribution and wheel to TestPyPI
publish-test: build
	hatch publish --repo test


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
