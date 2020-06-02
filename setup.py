#!/usr/bin/env python
import os
from distutils.core import setup
import numpy as np

setup(name='jPCA',
      version='0.0.1',
      description='jPCA for analyzing neural state trajectories',
      author='Benjamin Antin',
      author_email='benjaminantin1@gmail.com',
      url='https://github.com/bantin/jPCA',
      install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn'],
      packages=['jPCA'],
      include_dirs=[np.get_include(),],
      )
