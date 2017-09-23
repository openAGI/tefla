#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


# with open('requirements.txt') as requirements:
#    REQUIREMENTS = requirements.readlines()

# explicitly config
test_args = [
    '--cov-report=term',
    '--cov-report=html',
    '--cov=tefla',
    'tests'
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
    version='1.3.9',
    description='Simple end-to-end deep learning with tensorflow. Datasets, data-augmentation, models, training, prediction, and metrics',
    author='Tefla contributors',
    author_email='mrinalhaloi11@gmail.com',
    url='https://github.com/n3011/tefla',
    download_url='https://github.com/n3011/tefla/tarball/1.3.9',
    keywords=['tensorflow', 'deeplearning', 'cnn', 'deepcnn'],
    classifiers=[],
    install_requires=['numpy>=1.11.1', 'pandas==0.18.1', 'SharedArray==1.0', 'click==6.6', 'scikit-image==0.12.3',
                      'scikit-learn==0.18.2', 'six==1.10.0', 'setuptools==28.8.0', 'ghalton==0.6', 'Pillow==2.3.0', 'progress', 'opencv-python'],
    test_suite='tests',
    cmdclass={'test': PyTest},
    license='MIT',
    platforms=['linux'],
)
