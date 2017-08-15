# Code Style

We care about code style quality as much as features. If you want to contribute in any way, please try to follow these as much as possible:

* Docluster has some standard parameter naming:

    numbers -> n_name (ex. n_epochs, n_workers, n_words, n_tokens)
    boolean -> do_name (ex. do_plot, do_stem, do_analytics)

* Try to follow [pep8](https://www.python.org/dev/peps/pep-0008/) as the python code style. We recommend using [autopep8](https://pypi.python.org/pypi/autopep8) that automatically stylizes your code to conform to pep8. Many popular text editors such as Atom and Sublime have plugins that use autopep8 as a backend.

* Try to use existing modules. For example, do not try to implement PCA for a quick dimensionality reduction.

* We love organized and readable code. Also try to comment when necessary.
