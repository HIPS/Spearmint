import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "spearmint",
    version = "0.1",
    author = "Jasper Snoek, Ryan Adams, Kevin Swersky, Michael Gelbart, and Hugo Larochelle",
    author_email = "rpa@seas.harvard.edu, jsnoek@seas.harvard.edu, kswersky@cs.toronto.edu, mgelbart@seas.harvard.edu, hugo.larochelle@usherbrooke.ca",
    description = ("A package for Bayesian optimization."),
    keywords = "Bayesian Optimization, Magic, Minty Freshness",
    packages=['spearmint',
              'spearmint.choosers',
              'spearmint.grids',
              'spearmint.kernels',
              'spearmint.models',
              'spearmint.resources',
              'spearmint.sampling',
              'spearmint.schedulers',
              'spearmint.tasks',
              'spearmint.transformations',
              'spearmint.utils',
              'spearmint.utils.database',],
    long_description=read('README.md'),
)
