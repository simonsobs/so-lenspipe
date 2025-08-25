project('solenspipe',
  ['c', 'fortran'],
  version: '0.1.0',
  meson_version: '>= 0.64.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c99',
    'fortran_std=legacy',
  ],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Get numpy include directories
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

# Get f2py include directory
incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check : true
).stdout().strip()

# Explicitly add the numpy subdirectory that contains npy_os.h
incdir_numpy_headers = run_command(py,
  ['-c', 'import numpy, os; print(os.path.join(numpy.get_include(), "numpy"))'],
  check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py, incdir_numpy_headers)

numpy_dep = dependency('numpy', required: false)
if not numpy_dep.found()
  numpy_dep = declare_dependency()
endif

subdir('solenspipe')