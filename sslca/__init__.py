"""Sets up the cython environment correctly and in a way that is multi-machine
safe when using a network drive.
"""

import functools
import numpy as np
import os
import socket
import sys

def importCython():
    import pyximport.pyximport

    # Hack pyximport to have default options for profiling and embedding
    # signatures in docstrings.
    # Anytime pyximport needs to build a file, it ends up calling
    # pyximport.pyximport.get_distutils_extension.  This function returns an
    # object which has a cython_directives attribute that may be set to a
    # dictionary of compiler directives for cython.
    _old_get_distutils_extension = pyximport.pyximport.get_distutils_extension
    @functools.wraps(_old_get_distutils_extension)
    def _get_distutils_extension_new(*args, **kwargs):
        extension_mod, setup_args = _old_get_distutils_extension(*args,
                **kwargs)

        # Use g++, not gcc
        extension_mod.language = 'c++'

        if not hasattr(extension_mod, 'cython_directives'):
            extension_mod.cython_directives = {}
        extension_mod.cython_directives.setdefault('embedsignature', True)
        extension_mod.cython_directives.setdefault('profile', True)
        return extension_mod, setup_args
    pyximport.pyximport.get_distutils_extension = _get_distutils_extension_new

    # Finally, install pyximport so that each machine has its own build
    # directory (prevents errors with OpenMPI)
    _, pyxImporter = pyximport.install(build_dir = os.path.expanduser(
            '~/.pyxbld/{}'.format(socket.gethostname())),
            setup_args={'include_dirs':[np.get_include()]})
    _pyx_oFind = pyxImporter.find_module
    def _pyx_nFind(fullname, package_path=None):
        if fullname in ['cuda_ndarray.ProcessLookupErrorda_ndarray']:
            return None
        return _pyx_oFind(fullname, package_path)
    pyxImporter.find_module = _pyx_nFind
if 'pyximport' not in sys.modules:
    importCython()

from .sslca import LcaSpikingWoodsAnalyticalInhibition

