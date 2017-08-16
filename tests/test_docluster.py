#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `docluster` package."""

import pytest
from docluster import docluster


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


# Hint 3: That was easier than the first one. I surely do look like I follow a constant path.
