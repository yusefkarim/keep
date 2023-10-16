.PHONY: build upload test check clean

install:
	python -m pip install .

dev-install:
	python -m pip install -e .[dev]

build:
	python -m build

upload: clean build
	twine upload dist/*

test:
	python -m unittest discover tests

format-check:
	black --check ZeroFine

type-check:
	mypy --config mypy.ini ZeroFine

check: type-check format-check

clean:
	# Clean up build artifacts.
	-rm -R build dist ZeroFine.egg-info
	# Clean up bytecode leftovers.
	find . -type f -name '*.pyc' -print0 | xargs -0 rm
	find . -type d -name '__pycache__' -print0 | xargs -0 rmdir
