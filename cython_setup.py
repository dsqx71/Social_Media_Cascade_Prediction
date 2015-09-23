try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize    
setup(
    ext_modules = cythonize("exp_feature.py")
)

#python cython_setup.py build_ext --inplace