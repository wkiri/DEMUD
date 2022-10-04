#!/usr/bin/env python

from setuptools import setup

exec(open('demud/version.py').read())  # brings in a "version" variable

with open('README.md') as f:
    long_description = f.read()

setup(name='DEMUD',
      version=version,
      description='DEMUD novelty detection',
      long_description=long_description,
      author='Kiri Wagstaff',
      author_email='kiri.wagstaff@jpl.nasa.gov',
      url='https://github.com/wkiri/DEMUD',
      packages=['demud', 'demud.dataset', 'demud.log'],
      install_requires=['matplotlib',
                        'numpy',
                        'scipy',
                        'pillow'],
      entry_points={'console_scripts': ['demud=demud.demud:main']},
      python_requires='>=3.0',
     )
