import minipgm
from distutils.core import setup
import setuptools

setup(
    name='minipgm',
    description='A minimalistic probabilistic programming framework.',
    long_description=open('README.md').read(),
    version='0.1',
    author='Arnaud Rachez',
    author_email='arnaud.rachez@gmail.com',
    packages=['minipgm'],
    requires = ['numpy', 'scipy'],
)