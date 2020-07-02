#!/usr/bin/env python
from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='simplepytorch',
    version='0.1.0',
    description='Setup and train deep nets with PyTorch. Opinionated and Simple.',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/adgaudio/simplepytorch",
    author='Alex Gaudio',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    packages=['simplepytorch', 'simplepytorch.datasets'],
    scripts=['./bin/simplepytorch_plot', 'bin/simplepytorch', './bin/simplepytorch_debug'],
    install_requires=[
        "torchvision", "torch", "pretrainedmodels", "pandas", "numpy",
        "configargparse", "pillow", "scikit-learn", "matplotlib",
        "efficientnet_pytorch"],
    extras_require={
        'dataset_qualdr': ["pyjq", ]
    }
)
