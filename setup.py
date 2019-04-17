from setuptools import setup

setup(name="rlalgs",
      version="0.0.1",
      install_requires=[
        'gym',
        'scipy',
        'numpy',
        'mpi4py',
        'pandas',
        'pyyaml',
        'matplotlib',
        'prettytable'
      ],
      description="Implementations of some RL algorithms, based off of OpenAI spinningup tutorials",
      author="Jonathon Schwartz",
      )
