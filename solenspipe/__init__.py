"""Top-level package for SO-lenspipe."""

__author__ = """Mathew Madhavacheril"""
__email__ = 'mathewsyriac@gmail.com'
__version__ = '0.1.0'

# Import everything from all modules in this package
import importlib
import pkgutil
import warnings

__all__ = ['__version__', '__author__', '__email__']

# Import the Fortran extension
try:
    from . import _lensing_biases
    __all__.append('_lensing_biases')
except ImportError as e:
    warnings.warn(f"Fortran extension _lensing_biases could not be imported: {e}")

# Automatically discover and import all Python modules in this package
for finder, module_name, ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
    if not module_name.endswith('.__init__') and not module_name.split('.')[-1].startswith('_'):
        try:
            # Import the module
            module = importlib.import_module(module_name)
            # Get the short name (without package prefix)
            short_name = module_name.split('.')[-1]
            # Add to current namespace
            globals()[short_name] = module
            __all__.append(short_name)
            
            # Also import all public attributes from the module
            if hasattr(module, '__all__'):
                for attr in module.__all__:
                    if not attr.startswith('_'):
                        globals()[attr] = getattr(module, attr)
                        __all__.append(attr)
            else:
                # If no __all__, import all public attributes
                for attr in dir(module):
                    if not attr.startswith('_'):
                        globals()[attr] = getattr(module, attr)
                        __all__.append(attr)
                        
        except ImportError as e:
            warnings.warn(f"Could not import module {module_name}: {e}")