#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


with open('requirements.txt') as requirements:
    REQUIREMENTS = requirements.readlines()

# explicitly config
test_args = [
    '--cov-report=term',
    '--cov-report=html',
    '--cov=tefla',
    'tefla/tests'
]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='tefla',
    packages=find_packages(),
    version='1.0.0',
    description='Simple end-to-end deep learning with tensorflow. Datasets, data-augmentation, models, training, prediction, and metrics',
    author='Tefla contributors',
    author_email='mrinalhaloi11@gmail.com',
    url='https://github.com/n3011/tefla',
    download_url='https://github.com/n3011/tefla/tarball/1.0.0',
    keywords=['tensorflow', 'deeplearning', 'cnn', 'deepcnn'],
    classifiers=[],
    install_requires=REQUIREMENTS,
    test_suite='tefla/tests',
    cmdclass={'test': PyTest},
    license='MIT',
    platforms=['linux'],
)
