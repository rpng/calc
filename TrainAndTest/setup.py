from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules = cythonize("makeNet.py"))
setup(ext_modules = cythonize("writeDatabase.py"))
setup(ext_modules = cythonize("testNet.py"))



















