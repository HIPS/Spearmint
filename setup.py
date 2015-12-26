import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

from numpy.distutils.core import Extension

# This is needed for the SUR acquisition function for multi-objective optimization

ext1 = Extension(name = 'pbivnorm', sources = ['./pbivnorm/pbivnorm.pyf', './pbivnorm/pbivnorm.f'])

from numpy.distutils.core import setup

setup(
    name = "spearmint",
    version = "0.1",
    author = "Jasper Snoek, Ryan Adams, Kevin Swersky, Michael Gelbart, Hugo Larochelle, Daniel Hernandez-Lobato and Jose Miguel Hernandez-Lobato",
    author_email = "rpa@seas.harvard.edu, jsnoek@seas.harvard.edu, kswersky@cs.toronto.edu, mgelbart@seas.harvard.edu, hugo.larochelle@usherbrooke.ca, daniel.herandnez@uam.es, jmh@seas.harvard.edu",
    description = ("A package for Bayesian optimization."),
    keywords = "Bayesian Optimization, Magic, Minty Freshness",
    packages=['spearmint',
              'spearmint.acquisition_functions',
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
              'spearmint.visualizations',
              'spearmint.utils.database',],
    long_description=read('README.md'),
    ext_modules = [ ext1 ]
)
