#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = energy-forecast-benchmark-toolkit
PACKAGE_NAME = enfobench
PYTHON_INTERPRETER ?= python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create python virtual environment
venv/bin/python:
	( \
		$(PYTHON_INTERPRETER) -m venv ./venv; \
		source ./venv/bin/activate; \
		pip install --upgrade pip; \
	)

.PHONY: install
## Install project dependencies
install: venv/bin/python
	(\
		source ./venv/bin/activate; \
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
# CLONING RULES                                                                 #
#################################################################################


models/amazon-chronos/models/chronos-t5-tiny:
	git clone https://huggingface.co/amazon/chronos-t5-tiny ./models/amazon-chronos/models/chronos-t5-tiny

models/amazon-chronos/models/chronos-t5-mini:
	git clone https://huggingface.co/amazon/chronos-t5-mini ./models/amazon-chronos/models/chronos-t5-mini

models/amazon-chronos/models/chronos-t5-small:
	git clone https://huggingface.co/amazon/chronos-t5-small ./models/amazon-chronos/models/chronos-t5-small

models/amazon-chronos/models/chronos-t5-base:
	git clone https://huggingface.co/amazon/chronos-t5-base ./models/amazon-chronos/models/chronos-t5-base

models/amazon-chronos/models/chronos-t5-large:
	git clone https://huggingface.co/amazon/chronos-t5-large ./models/amazon-chronos/models/chronos-t5-large --progress

download-amazon-chronos: models/amazon-chronos/models/chronos-t5-tiny \
						 models/amazon-chronos/models/chronos-t5-mini \
						 models/amazon-chronos/models/chronos-t5-small \
						 models/amazon-chronos/models/chronos-t5-base \
						 models/amazon-chronos/models/chronos-t5-large



models/salesforce-moirai/models/moirai-1.0-R-small:
	git clone https://huggingface.co/salesforce/moirai-1.0-R-small ./models/salesforce-moirai/models/moirai-1.0-R-small

models/salesforce-moirai/models/moirai-1.0-R-base:
	git clone https://huggingface.co/salesforce/moirai-1.0-R-base ./models/salesforce-moirai/models/moirai-1.0-R-base

models/salesforce-moirai/models/moirai-1.0-R-large:
	git clone https://huggingface.co/salesforce/moirai-1.0-R-large ./models/salesforce-moirai/models/moirai-1.0-R-large

download-salesforce-moirai: models/salesforce-moirai/models/moirai-1.0-R-small \
							models/salesforce-moirai/models/moirai-1.0-R-base \
							models/salesforce-moirai/models/moirai-1.0-R-large

#################################################################################
# MODEL RULES                                                                   #
#################################################################################

DOCKER_HUB_REPOSITORY := $(DOCKER_HUB_REPOSITORY)
ENFOBENCH_VERSION := $(shell hatch version)
MODEL := sf-naive
IMAGE_TAG := $(ENFOBENCH_VERSION)-$(MODEL)
DEFAULT_PORT := 3000

DARTS_VERSION := 0.30.0
SKTIME_VERSION := 0.30.1
STATSFORECAST_VERSION := 1.7.5
CHRONOS_VERSION := 1.2.0


.PHONY: base-image-darts
## Build base image for darts
base-image-darts:
	docker build --build-arg DARTS_VERSION=$(DARTS_VERSION) -t $(DOCKER_HUB_REPOSITORY):base-u8darts-$(DARTS_VERSION) ./docker/base/darts
	docker push $(DOCKER_HUB_REPOSITORY):base-u8darts-$(DARTS_VERSION)


.PHONY: base-image-sktime
## Build base image for sktime
base-image-sktime:
	docker build --build-arg SKTIME_VERSION=$(SKTIME_VERSION) -t $(DOCKER_HUB_REPOSITORY):base-sktime-$(SKTIME_VERSION) ./docker/base/sktime
	docker push $(DOCKER_HUB_REPOSITORY):base-sktime-$(SKTIME_VERSION)


.PHONY: base-image-statsforecast
## Build base image for statsforecast
base-image-statsforecast:
	docker build --build-arg STATSFORECAST_VERSION=$(STATSFORECAST_VERSION) -t $(DOCKER_HUB_REPOSITORY):base-statsforecast-$(STATSFORECAST_VERSION) ./docker/base/statsforecast
	docker push $(DOCKER_HUB_REPOSITORY):base-statsforecast-$(STATSFORECAST_VERSION)


.PHONY: base-image-amazon-chronos
base-image-amazon-chronos:
	docker build --build-arg CHRONOS_VERSION=$(CHRONOS_VERSION) -t $(DOCKER_HUB_REPOSITORY):base-amazon-chronos-$(CHRONOS_VERSION) ./docker/base/amazon-chronos
	docker push $(DOCKER_HUB_REPOSITORY):base-amazon-chronos-$(CHRONOS_VERSION)


.PHONY: base-image-salesforce-moirai
base-image-salesforce-moirai:
	docker build -t $(DOCKER_HUB_REPOSITORY):base-salesforce-moirai ./docker/base/salesforce-moirai
	docker push $(DOCKER_HUB_REPOSITORY):base-salesforce-moirai


## Build base images
base-images: base-image-darts \
			 base-image-sktime \
			 base-image-statsforecast \
			 base-image-amazon-chronos \
			 base-image-salesforce-moirai


.PHONY: build-image
## Build a model image
build-image:
	docker build -t $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG) ./models/$(MODEL)


.PHONY: push-image
## Push a model image to Docker Hub
push-image:
	docker push $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)


.PHONY: run-image
## Run a model image
run-image:
	docker run -it --rm -p $(DEFAULT_PORT):3000 $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)


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
