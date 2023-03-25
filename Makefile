format:
	nbqa black nbs/
	nbqa isort nbs/
	nbqa flake8 nbs/

mypy:
	nbqa mypy nbs/ --ignore-missing-imports --check-untyped-defs

all: format mypy nbdev_prepare
