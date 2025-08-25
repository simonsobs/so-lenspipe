# Install Python files explicitly
py.install_sources(
  files(
    '__init__.py',
    'analysis_tools.py',
    'bias.py',
    'biastheory.py',
    'shear.py',
    'solenspipe.py',
    'utility.py'

  ),
  subdir: 'solenspipe'
)

# Install the pre-built f2py extension
py.install_sources(
  ['_lensing_biases.cpython-311-x86_64-linux-gnu.so'],
  subdir: 'solenspipe'
)