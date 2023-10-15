# Keep

`<Write introduction here>`

## Checklist

For completing the tasks below, you can refer to [Scapy](https://github.com/secdev/scapy) or [pyRSKtools](https://github.com/yusefkarim/RBR-pyRSKtools) as decent Python project examples.

- [ ] Create project skeleton
  - Update this README, replacing `<Write introduction here>` with a proper (brief) introduction about your app and what it will do. Try your best to be grammatically correct.
  - Create a folder called `keep` with an `__init__.py` file inside it, but a simple print statement there. The `keep` folder will be where all your main project code exists.
  - Create a `setup.py` file. Again, see aforementioned project examples linked above for ideas and refer to [Packaging and distributing projects](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
- [ ] Testing skeleton
  - Create a folder called `tests` with a dummy test inside it, this test should be run when you execute `python -m unittest discover tests`
- [ ] Create a good (local) build flow
  - Install [Black](https://black.readthedocs.io/en/stable/), [mypy](https://www.mypy-lang.org/), [Twine](https://twine.readthedocs.io/en/stable/index.html), and [Sphinx](https://www.sphinx-doc.org/en/master/) as dev/extra dependencies (should be reflected in your `setup.py` file)
  - Create some form of automated script to do the following common tasks: installing dependencies, building code, uploading code to PyPi, doing a code format check using Black, doing a code type check with mypy, running tests
  - You cane see pyRSKtools' [Makefile](https://github.com/yusefkarim/RBR-pyRSKtools/blob/master/Makefile) as one example to follow, but you don't necessarily have to use [make](https://www.gnu.org/software/make/) for this
- [ ] ToDo: Github actions
- [ ] ToDo: Start writing actual code
- [ ] ToDo: Write your first real unit testing
- [ ] ToDo: TBD
