from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'cyclic decoding',
  ext_modules = cythonize("decoding.pyx"),
  include_dirs=[numpy.get_include()]
)