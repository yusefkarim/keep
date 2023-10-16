# ZeroFine

Coming soon...

## Checklist

For completing the tasks below, you can refer to [Scapy](https://github.com/secdev/scapy) or [pyRSKtools](https://github.com/yusefkarim/RBR-pyRSKtools) as decent Python project examples.

1. [x] Create project skeleton
  - Create a folder called `zerofine` with an `__init__.py` file inside it, but a simple print statement there. The `zerofine` folder will be where all your main project code exists.
  - Create a `setup.py` file. Again, see aforementioned project examples linked above for ideas and refer to [Packaging and distributing projects](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
2. [x] Create a testing skeleton
  - Create a folder called `tests` with a dummy test inside it, this test should be run when you execute `python -m unittest discover tests`
3. [x] Create a good (local) build flow
  - Install [Black](https://black.readthedocs.io/en/stable/), [mypy](https://www.mypy-lang.org/), [Twine](https://twine.readthedocs.io/en/stable/index.html), and [Sphinx](https://www.sphinx-doc.org/en/master/) as dev/extra dependencies (should be reflected in your `setup.py` file)
  - Create some form of automated script to do the following common tasks: installing dependencies, building code, uploading code to PyPi, doing a code format check using Black, doing a code type check with mypy, running tests
  - You can see pyRSKtools' [Makefile](https://github.com/yusefkarim/RBR-pyRSKtools/blob/master/Makefile) as one example to follow, but you don't necessarily have to use [make](https://www.gnu.org/software/make/) for this
4. [ ] Setup Github actions for the project, the CI should do format check, type check, and run tests. PRs should only be allowed to be merged if all actions pass.
5. [ ] Start writing actual code, maybe start with a technical spec describing what you are actually trying to achieve and some of the libraries/frameworks you already know will be needed
6. [ ] Write unit tests at the same time you write production code (test-driven development)
