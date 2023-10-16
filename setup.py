#https://www.activestate.com/resources/quick-reads/how-to-package-python-dependencies-with-pip-setuptools/

from distutils.core import setup
from setuptools import find_packages
import os
# Optional project description in README.md:
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''
setup(
	# Project name:
    name='ZeroFine',
    # Packages to include in the distribution:
    packages=find_packages('ZeroFine'),
    # Project version number:
    version='0.0',
    # List a license for the project, eg. MIT License
    #license='',
    # Short description of your library:
    description='For helping piggy get the street parking tickets less',
    # Long description of your library:
    long_description=long_description,
    long_description_content_type='text/markdown',
    # Your name:
    author='Piggy',
    # Your email address:
    author_email='hmu026@icloud.com',
    # Link to your github repository or website:
    #url='',
    # Download Link from where the project can be downloaded from:
    #download_url='',
    # List of keywords:
    keywords=["street parking", "computer vision", "artificial intelligence"],
    # List project dependencies:
    install_requires=["numpy", "scikit-learn", "matplotlib","requests", "pillow", "pydantic"],
    extras_require={
        "dev": ["twine", "black", "mypy", "sphinx", "pre-commit"],
    },
    # https://pypi.org/classifiers/
    classifiers=[
        "Intended Audience :: Engineer/Application",
        "Topic :: Engineering/AI",
        "Programming Language :: Python :: 3",
        ]
)
