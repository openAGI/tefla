from distutils.core import setup


#with open('requirements.txt') as requirements:
#    REQUIREMENTS = requirements.readlines()

setup(
    name='tefla',
    packages=['tefla'],
    version='0.1',
    description='Simple end-to-end deep learning with tensorflow. Datasets, data-augmentation, models, training, prediction, and metrics',
    author='Tefla contributors',
    author_email='mrinalhaloi11@gmail.com',
    url='https://github.com/n3011/tefla',
    keywords=['tensorflow', 'deeplearning', 'cnn', 'deepcnn'],
    classifiers=[],
    install_requires=REQUIREMENTS,
    license='MIT',
    platforms=['linux'],
)
