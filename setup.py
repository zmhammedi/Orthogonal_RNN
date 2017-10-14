#!/usr/bin/env
# -*- coding: UTF-8 -*-

#  Run python setup.py build_ext --inplace

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('C_fun', sources = ['C_fun.c'])]

setup(
        name = 'C_fun',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )
