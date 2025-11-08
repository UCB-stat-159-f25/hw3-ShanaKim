ENV_NAME = ligo
PYTHON = python
MYS = myst

.PHONY: env html clean help

env:
	@if conda env list | grep -q "^$(ENV_NAME) "; then \
		conda env update -n $(ENV_NAME) -f environment.yml; \
	else \
		conda env create -f environment.yml; \
	fi

# Build myst site as html
html:
	myst build --html

# Delete generated files
clean:
	rm -rf figures/*
	rm -rf audio/*
	rm -rf _build/*