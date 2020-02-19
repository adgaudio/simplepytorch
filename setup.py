#!/usr/bin/env python
from setuptools import setup


setup(
    name='SimplePyTorch',
    version='0.01',
    description='Setup and train deep nets with PyTorch. Opinionated and Simple.',
    author='Alex Gaudio',
    packages=['simplepytorch'],
    scripts=['./bin/simplepytorch_plot', 'bin/simplepytorch', './bin/simplepytorch_debug']
)
