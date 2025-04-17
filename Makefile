## XAI for Skin Lesion Classification - Makefile

.PHONY: clean data lint download_data organize_data extract_features train explain test all

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python3
PROJECT_NAME = skin-lesion-xai
PROJECT_DIR = $(shell pwd)
VENV_DIR = $(PROJECT_DIR)/venv

ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
download_data:
	$(PYTHON_INTERPRETER) -c "from XAI.dataset import download_and_extract_ham10000; download_and_extract_ham10000()"

organize_data:
	$(PYTHON_INTERPRETER) -c "from XAI.dataset import organize_data; organize_data()"

extract_features:
	$(PYTHON_INTERPRETER) -c "from XAI.features import main; main()"

data: download_data organize_data extract_features

## Train Model
train:
	$(PYTHON_INTERPRETER) -c "from XAI.modeling.train import main; main()"

## Explain Model Predictions
explain:
	$(PYTHON_INTERPRETER) -c "from XAI.modeling.predict import main; main()"

## Test Model
test:
	$(PYTHON_INTERPRETER) -c "from XAI.modeling.predict import test_model; test_model()"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete

## Lint using flake8
lint:
	flake8 XAI

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name $(PROJECT_NAME) python=3.9
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	@echo ">>> Creating virtualenv environment in $(VENV_DIR)"
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo ">>> Activate the virtualenv with:\nsource $(VENV_DIR)/bin/activate"
endif

## Run all steps in sequence
all: requirements download_data organize_data extract_features train explain

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
