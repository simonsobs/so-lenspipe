project(
  'solenspipe',
  ['c', 'fortran'],
  version: '0.1.0',                 # Keep in sync with pyproject.toml
  meson_version: '>=1.2.3',         # meson-python on Py3.12+ expects >=1.2.3
  default_options: [
    'c_std=c99',                     # mirrors your -std=c99
    'buildtype=release'              # O3 etc. controlled by buildtype/optimization
  ],
)

py = import('python').find_installation(pure: false)
py_dep    = py.dependency()          # libpython + include path

# --- NumPy include dir -------------------------------------------------
numpy_inc = run_command(
  py, '-c', 'import numpy,sys;print(numpy.get_include())'
).stdout().strip()
inc_numpy = include_directories(numpy_inc)

# --- locate fortranobject.c correctly ---------------------------------
fortranobject_c_path = run_command(
  py, '-c',
  '''
import importlib.util, pathlib, sys, numpy
spec = importlib.util.find_spec("numpy.f2py")
print(pathlib.Path(spec.origin).parent / "src" / "fortranobject.c")
'''
).stdout().strip()

fortranobject_c = files(fortranobject_c_path)


f2py_src_inc = include_directories(
  run_command(
    py, '-c',
    '''
import importlib.util, pathlib
spec = importlib.util.find_spec("numpy.f2py")
print(pathlib.Path(spec.origin).parent / "src")
'''
  ).stdout().strip()
)


# ----------  common compile / link flags ----------
openmp_dep = dependency('openmp', required : true)

# Allow users to override optimisation flags like the old FCFLAGS
fc_env = run_command('sh', '-c', 'echo ${FCFLAGS:-}').stdout().strip().split()
if fc_env.length() == 0
  fc_env = ['-O3', '-fPIC']
endif

extra_fortran_args = [
  '-fopenmp', '-Wno-conversion', '-Wno-tabs', '-fPIC',
] + fc_env

add_project_arguments(extra_fortran_args, language : 'fortran')
add_project_link_arguments('-fopenmp', language : ['c','fortran'])

# ----------  libalkernel static library ----------
lb_src = files(
  'fortran/LensingBiases.f90',
)

lb_lib = static_library(
  'lensing_biases',
  lb_src,
  dependencies : openmp_dep,
  install      : false,          # it is only for internal linking
)

lb_dep = declare_dependency(link_with : lb_lib)

lb_sources = files(
  'fortran/LensingBiases.f90',
)

# Tell f2py to generate the wrapper code for all *.f90 files in one shot
f2py_wrappers = custom_target(
  '_lensing_biases_f2py',
  input  : lb_sources,
  output : ['_lensing_biases-f2pywrappers2.f90', '_lensing_biasesmodule.c'],
  command: [py, '-m', 'numpy.f2py', '-m', '_lensing_biases', '--lower'] + lb_sources,
  depend_files : lb_sources,
)

py.extension_module(
  '_lensing_biases',
  # order is important: f2py wrappers, raw sources, fortranobject helper
  sources             : lb_sources + f2py_wrappers + [fortranobject_c],
  include_directories : [inc_numpy, f2py_src_inc],  
  dependencies        : [py_dep, openmp_dep, lb_dep],
  link_args           : ['-fopenmp'],
  subdir              : 'solenspipe',      
  install             : true,
)



# Install your pure-Python package files alongside the extension.
# This copies the whole solenspipe/ directory to site-packages.
install_subdir(
  'solenspipe',
  install_dir: py.get_install_dir(),    # site-packages location for this interpreter
)
