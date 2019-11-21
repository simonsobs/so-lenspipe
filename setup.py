#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools
from distutils.errors import DistutilsError
from numpy.distutils.core import setup, Extension, build_ext, build_src
from distutils.sysconfig import get_config_var, get_config_vars
import versioneer
import os, sys
import subprocess as sp
import numpy as np
build_ext = build_ext.build_ext
build_src = build_src.build_src

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements       = []
setup_requirements = []
test_requirements  = []

# The following copied from healpy might be necessary for libsharp
# Apple switched default C++ standard libraries (from gcc's libstdc++ to
# clang's libc++), but some pre-packaged Python environments such as Anaconda
# are built against the old C++ standard library. Luckily, we don't have to
# actually detect which C++ standard library was used to build the Python
# interpreter. We just have to propagate MACOSX_DEPLOYMENT_TARGET from the
# configuration variables to the environment.
#
# This workaround fixes <https://github.com/healpy/healpy/issues/151>.
if (
    get_config_var("MACOSX_DEPLOYMENT_TARGET")
    and not "MACOSX_DEPLOYMENT_TARGET" in os.environ
):
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = get_config_var("MACOSX_DEPLOYMENT_TARGET")


compile_opts = {
    'extra_compile_args': ['-std=c99','-fopenmp', '-Wno-strict-aliasing', '-g'],
    'extra_f90_compile_args': ['-fopenmp', '-Wno-conversion', '-Wno-tabs'],
    'extra_link_args': ['-fopenmp', '-g']
    }

fcflags = os.getenv('FCFLAGS')
if fcflags is None or fcflags.strip() == '':
    fcflags = ['-O3']
else:
    print('User supplied fortran flags: ', fcflags)
    print('These will supersede other optimization flags.')
    fcflags = fcflags.split()
    
compile_opts['extra_f90_compile_args'].extend(fcflags)

    


class CustomBuild(build_ext):
    def run(self):
        # Then let setuptools do its thing.
        return build_ext.run(self)

class CustomSrc(build_src):
    def run(self):
        # Then let setuptools do its thing.
        return build_src.run(self)

class CustomEggInfo(setuptools.command.egg_info.egg_info):
    def run(self):
        return setuptools.command.egg_info.egg_info.run(self)   

# Cascade your overrides here.
cmdclass = {
    'build_ext': CustomBuild,
    'build_src': CustomSrc,
    'egg_info': CustomEggInfo,
}
cmdclass = versioneer.get_cmdclass(cmdclass)


setup(
    author="Simons Observatory Collaboration L3 Working Group",
    author_email='mathewsyriac@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="SO Lensing Pipeline",
    package_dir={"solenspipe": "solenspipe"},
    ext_modules=[
        Extension('solenspipe._lensing_biases',
            sources=['fortran/LensingBiases.f90'],
            **compile_opts),
    ],
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='Lensing',
    name='solenspipe',
    packages=['solenspipe'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/simonsobs/so-lenspipe',
    version=versioneer.get_version(),
    zip_safe=False,
    cmdclass=cmdclass
)

print('\n[setup.py request was successful.]')

