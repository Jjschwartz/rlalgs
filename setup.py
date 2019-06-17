import sys
from setuptools import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 7, \
    "The rlalgs repo is designed to work with Python 3.7 and greater." \
    + "Please install it before proceeding."

setup(name="rlalgs",
      version="0.0.1",
      install_requires=[
          'gym',
          'scipy',
          'numpy',
          'mpi4py',
          'pandas',
          'pyyaml',
          'psutil',
          'matplotlib',
          'prettytable',
          'tensorflow==2.0.0-beta0'
      ],
      description="Implementations of some RL algorithms, based off of OpenAI spinningup tutorials",
      author="Jonathon Schwartz",
      )
