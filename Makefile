#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = energy-forecast-benchmark-toolkit
PACKAGE_NAME = enfobench
PYTHON_INTERPRETER ?= python3

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

.PHONY: install
## Install project dependencies
install: venv/bin/python
	(\
		source $(PROJECT_DIR)/venv/bin/activate; \
		pip install -e .; \
    )

.PHONY: clean
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: lint
## Lint using ruff, mypy, black, and isort
lint:
	hatch run lint:all

.PHONY: style
## Check style using ruff, black, and isort
style:
	hatch run lint:style

.PHONY: format
## Format using black
format:
	hatch run lint:fmt

.PHONY: tests
## Run pytest with coverage
tests:
	hatch run cov

.PHONY: docs
## Create documentation
docs:
	hatch run docs:serve

.PHONY: docs-build
## Build documentation
docs-build:
	hatch run docs:build

#################################################################################
# PACKAGING RULES                                                               #
#################################################################################

.PHONY: build
## Build source distribution and wheel
build:
	hatch build

.PHONY: publish
## Upload source distribution and wheel to PyPI
publish: build
	hatch publish --repo main

.PHONY: publish-test
## Upload source distribution and wheel to TestPyPI
publish-test: build
	hatch publish --repo test


#################################################################################
# MODEL RULES                                                                   #
#################################################################################

DOCKER_HUB_REPOSITORY := $(DOCKER_HUB_REPOSITORY)
ENFOBENCH_VERSION := $(shell hatch version)
MODEL_NAME := sf-naive
IMAGE_TAG := $(ENFOBENCH_VERSION)-$(MODEL_NAME)
DEFAULT_PORT := 3000

.PHONY: base-images
## Create docker base image
base-images:
	docker build --build-arg DARTS_VERSION=0.27.0 -t $(DOCKER_HUB_REPOSITORY):base-u8darts-0.27.0 ./docker/base/darts
	docker build --build-arg SKTIME_VERSION=0.24.1 -t $(DOCKER_HUB_REPOSITORY):base-sktime-0.24.1 ./docker/base/sktime
	docker build --build-arg STATSFORECAST_VERSION=1.5.0 -t $(DOCKER_HUB_REPOSITORY):base-statsforecast-1.5.0 ./docker/base/statsforecast

.PHONY: image
## Create docker image
image:
	docker build -t $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG) ./models/$(MODEL_NAME)

.PHONY: push-image
## Push docker image to Docker Hub
push-image: image
	docker push $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)

.PHONY: run-image
run-image: image
	docker run -it --rm -p $(DEFAULT_PORT):3000 $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)


MODELS = $(shell ls -d ./models/* | xargs -n 1 basename)
images:
	$(foreach var,$(MODELS), $(MAKE) image MODEL_NAME=$(var);)

push-images:
	$(foreach var,$(MODELS), $(MAKE) push-image MODEL_NAME=$(var);)


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
