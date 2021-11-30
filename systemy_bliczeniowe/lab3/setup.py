from setuptools import setup
from Cython.Build import cythonize

MODULE_PATH = 'systemy_bliczeniowe/lab3'

setup(
    ext_modules=cythonize(f'{MODULE_PATH}/lista3.pyx')
)