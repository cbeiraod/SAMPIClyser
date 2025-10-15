"""""""""""""
SAMPIClyser
"""""""""""""

.. start-badges

|pre-commit|

|black| |flake8| |isort|

|docs|

|version| |wheel| |supported-versions| |supported-implementations|

|commits-since|

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
    :target: https://github.com/psf/black
    :alt: black

.. |flake8| image:: https://img.shields.io/badge/flake8-checked-blueviolet
    :target: https://github.com/PyCQA/flake8
    :alt: flake8

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/
    :alt: isort

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

.. |docs| image:: https://readthedocs.org/projects/sampiclyser/badge/?version=latest
    :target: https://sampiclyser.readthedocs.io/
    :alt: Documentation Status

.. |version| image:: https://img.shields.io/pypi/v/SAMPIClyser.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/SAMPIClyser

.. |wheel| image:: https://img.shields.io/pypi/wheel/SAMPIClyser.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/SAMPIClyser

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/SAMPIClyser.svg
    :alt: Supported versions
    :target: https://pypi.org/project/SAMPIClyser

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/SAMPIClyser.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/SAMPIClyser

.. |commits-since| image:: https://img.shields.io/github/commits-since/cbeiraod/SAMPIClyser/v0.1.1.svg
    :alt: Commits since latest release
    :target: https://github.com/cbeiraod/SAMPIClyser/compare/v0.1.1...main

.. end-badges

Python library to help decoding and analysing data from a SAMPIC system

* Free software: Zlib license


----------


.. contents:: Table of Contents
   :depth: 3


----------


====================
Running the software
====================
We will use a venv to isolate the SAMPIClyser python installation and dependencies from that of the system and other venvs.
Create the venv with: ``python -m venv venv``.
Venv creation only needs to be done once and occasionally when you upgrade the python version installed on the system.

Before running any commands related to the SAMPIClyser, activate the venv with: ``source venv/bin/activate``

Once you are finished, you may deactivate the venv with: ``deactivate``

---------------------------
Dependencies & Installation
---------------------------
There are two options, you can run the SAMPIClyser from source or from `PyPI <https://pypi.org/>`_ (Recommended).

From PyPI
---------
Make sure you activate the venv first, then simply run the command: ``python -m pip install SAMPIClyser``

From Source
-----------
Fetch the source from github, you may for instance clone the git repository to a local directory.
Make sure the venv is activated, then install this application with all its dependencies by running this command from the source directory: ``python -m pip install .``

----------------------
Developer Dependencies
----------------------
If you are working on developing the code for SAMPIClyser you will need to follow the "Running the Software" instructions for setup and then for installing use the "From Source" instructions but instead of the install command in the "Dependencies" instructions, use the following command: ``python -m pip install -e .``.
Then install a few other dependencies which are only used in development:

.. code:: bash

  python -m pip install --upgrade pytest
  python -m pip install --upgrade pytest-cov
  python -m pip install --upgrade pytest-mock
  python -m pip install --upgrade pre-commit
  python -m pip install --upgrade bump2version
  python -m pip install --upgrade black
  python -m pip install --upgrade flake8-pyproject
  python -m pip install --upgrade isort
  python -m pip install --upgrade build
  python -m pip install --upgrade twine


Then install the pre-commit hooks so that things are ran automatically before any commit is done: ``pre-commit install``.


------------


==============
Developer Info
==============

-----
Tools
-----

pytest
------
We are using pytest to run unit tests on the software.
See `here <https://docs.pytest.org/en/7.4.x/getting-started.html>`_ for ideas on how to get started.
Use the command ``pytest`` to run all the tests.

We use the pytest-cov plugin to get coverage reports from pytest.
Use the command ``pytest --cov --cov-report term-missing`` to run all the tests and get a coverage report.
Use the command ``pytest --cov --cov-report term-missing --cov-report html`` to get an html report with detailed information.

We use the pytest-mock plugin in order to use the mock class in our tests.

pre-commit
----------
**pre-commit may need to be installed in the global python environment for things to work correctly.**
This tool allows to setup hooks into the git workflow, in particular for the Pre-Commit Hook, allowing to run automated tests before committing code.
This functionality is used to automatically run black, flake8 and isort before any commit is made, thus guaranteeing a consistent style and formatting for all committed code (according to these tools).
If you want to run all the checks individually as if a commit were about to be made, you can use ``pre-commit run --all-files``.
You can find more information on pre-commit `here <https://pre-commit.com/>`_.

bump2version
------------
We are using bump2version to manage the version string of the software.
bump2version will automatically create a commit and a tag with the version when you use it:

- To increase the major version, use: ``bump2version major``; for example 0.1.3 to 1.0.0
- To increase the minor version, use: ``bump2version minor``; for example 0.1.3 to 0.2.0
- To increase the patch version, use: ``bump2version patch``; for example 0.1.3 to 0.1.4

black
-----
We are using black to automatically format the python code.
To run black standalone use: ``black .`` in the root directory.
You may also use ``black --check -v .`` to get a list of which changes would be made, without actually making them.
More information on black can be found `here <https://pypi.org/project/black/>`_.

flake8
------
We are using flake8 to check the code for style and syntax errors (i.e. a linter tool).
The flake8-pyproject enables flake8 to read configuration from the pyproject.toml file, and pulls flake8 as a dependency.
This is why flake8 is not explicitly installed.
To run flake8 standalone, use: ``flake8`` in the root directory.
More information on flake8 can be found `here <https://flake8.pycqa.org/en/latest/index.html#quickstart>`_.
A list of all options and configuration for the toml file can be found `here <https://flake8.pycqa.org/en/latest/user/options.html>`_.

isort
-----
We are using isort to automatically sort the include statements at the top of the python files.
To run isort standalone use: ``isort .`` in the root directory.
More information on isort can be found `here <https://pycqa.github.io/isort/>`_.

build
-----
The build tool is used to package the code for publishing on PyPI.
Build the release with: ``python -m build``

twine
-----
The twine tool is used to upload the package to PyPI.
Once the distribution files are generated with the build tool, then upload them with: ``python -m twine upload --repository testpypi dist/*``

----------------
Github Workflows
----------------
Currently there is a single github workflow which builds and publishes a properly tagged version of the repo onto PyPI.
We plan to use other github workflows as our CI tools, to be added.

publish-to-test-pypi
--------------------
This workflow publishes to PyPI a properly tagged commit of the repository.
Some initial setup in needed on PyPI so that things run smoothly, but once running it should continue to function without intervention needed.

-----------------
Restructured Text
-----------------
For information on how to use restructured text, see the cheatsheet `here <https://github.com/DevDungeon/reStructuredText-Documentation-Reference>`_ for example.
But there are other resources on the internet if you prefer.

----------
Docstrings
----------
Please use docstrings in the "NumPy/SciPy docstrings" style: `link <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

--------------
pyproject.toml
--------------
Get classifiers from `here <https://pypi.org/classifiers/>`_.

More information on packaging can be found `here <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_.

There are entry points as explained in: https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata
There is an entry point to run the binary conversion tool, for example: ``sampic-convert`` tool, use the help to see the options.
