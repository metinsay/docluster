# Contributing


Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways.

## Types of Contributions

### Report Bugs


Report bugs at https://github.com/metinsay/docluster/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

**NOTE:** Please check out [Code Style](./CODE_STYLE.md) before you start implementing.

After setting up your local repo of `docluster` with [Get Started](##Get-Started!), these are the things you can choose to do:

1) Create a new `Model` (See [model implementation](./MODEL_IMPLEMENTATION.md)).
2) Improve a `Model` that already exists in docluster. Perform the changes and send a pull request.
3) Contribute to any other core features (Contact [maintainers](./AUTHORS.md)).

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/metinsay/docluster/issues.


## Get Started!


Ready to contribute? Here's how to set up `docluster` for local development.

1. Fork the `docluster` repo on GitHub.

2. Clone your fork locally

    ```shell
    $ git clone git@github.com:your_name_here/docluster.git
    ```

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development.

    ```shell
    $ mkvirtualenv docluster
    $ cd docluster/
    $ python3 setup.py develop
    ```

4. Create a branch for local development.

    ```shell
    $ git checkout -b name-of-your-bugfix-or-feature
    ```
   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox.

    ```shell
    $ flake8 docluster tests
    $ python setup.py test or py.test
    $ tox
    ```

   To get `flake8` and `tox`, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub.

    ```shell
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines


Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for Python 2.6, 2.7, 3.3, 3.4 and 3.5, and for PyPy. Check https://travis-ci.org/metinsay/docluster/pull_requests and make sure that the tests pass for all supported Python versions.
