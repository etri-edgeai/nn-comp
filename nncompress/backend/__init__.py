from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
import importlib

# Default backend: tf.Keras
_BACKEND = 'tensorflow'

if 'NNCOMPRESS_HOME' in os.environ:
    _nncompress_dir = os.environ.get('NNCOMPRESS_HOME')
else:
    _nncompress_base_dir = os.path.expanduser('~')
    if not os.access(_nncompress_base_dir, os.W_OK):
        _nncompress_base_dir = '/tmp'
    _nncompress_dir = os.path.join(_nncompress_base_dir, '.nncps')

# Attempt to read nncompress config file.
_config_path = os.path.expanduser(os.path.join(_nncompress_dir, 'mlck.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _BACKEND = _config.get('backend', _BACKEND)

# Save config file, if possible.
if not os.path.exists(_nncompress_dir):
    try:
        os.makedirs(_nncompress_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        'backend': _BACKEND,
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on NNCOMPRESS_BACKEND flag, if applicable.
if 'NNCOMPRESS_BACKEND' in os.environ:
    _backend = os.environ['NNCOMPRESS_BACKEND']
    if _backend:
        _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend\n')
    from .tensorflow_backend import *
elif _BACKEND == 'torch':
    sys.stderr.write('Using torch backend.\n')
    from .torch_backend import *
else:
    raise ValueError('Unable to import backend : ' + str(_BACKEND))


def backend():
    """Returns the name of the current backend (e.g. "tensorflow").

    # Returns
        String, the name of the backend is currently using.

    # Example
    ```python
        >>> nncompress.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND
