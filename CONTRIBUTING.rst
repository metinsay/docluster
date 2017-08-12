.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/metinsay/docluster/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

One of the main goal of docluster is to have the most recent research models in the library. If you like to implement a model that you think will be useful, start by browsing docluster/core. Make sure the model is not already implemented. Note that many files need implementations. You can choose to implement one of the mock files or choose to add your model. In any case:

1) Create a file with filename being the name of your model. The filename should be all lowercase and shouldn't have any underscores.

2) Copy paste the following template inside the python file::

     from docluster.core import Model

    class YourModelName(Model):
      def __init__(self, model_name='YourModelName'):
          <One-liner description of your model>

          Credits:
          --------
          This was adapted from a post/code from <Name of author (which can be you)> that can be
          found here:
          <Website of author/code/white paper>

          Authors:
          --------
          - <Your name>
          - <Another name>

          Paramaters:
          -----------
          <Any parameters>

          model_name : str
              Name of the model that will be used for printing and saving purposes.

          Attributes:
          -----------
          <Any Attributes>

          """

          # Initialize any instance variables
          self.model_name = model_name

      def fit(self, data):
          """
          <One-liner description of what this method achieves>

          Paramaters:
          -----------
          data : <type>
              The data that is going to be <what will it be used for>.

          Return:
          --------
          <Any Return values>
          """

          # Apply your model and return the most essential result of your model.
          # This is usually a vector of floats, integers or strings.
          pass


      # Create any other methods you think are useful.
      # You can have helper methods as well, but put a '_' in front of the method name
      # to make it private to the class.

Important Notes:

* Check out docluster/utils. There are many helper methods and data structures that might be useful. Try to use them as much as possible.
* Please use autopep8 with vertical line wrapping. This will ensure docluster has a standard across each module.


Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/metinsay/docluster/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `docluster` for local development.

1. Fork the `docluster` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/docluster.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv docluster
    $ cd docluster/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox::

    $ flake8 docluster tests
    $ python setup.py test or py.test
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.6, 2.7, 3.3, 3.4 and 3.5, and for PyPy. Check
   https://travis-ci.org/metinsay/docluster/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ py.test tests.test_docluster
