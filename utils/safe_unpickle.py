"""Load a pandas/numpy pickle while refusing every global outside an allowlist (no arbitrary code execution)."""
import importlib
import pickle

ALLOWED_MODULE_PREFIXES = ('numpy', 'pandas')
ALLOWED_BUILTINS = {'slice', 'range', 'complex', 'set', 'frozenset', 'bytearray', 'list', 'dict', 'tuple', 'object'}
REFUSED_NAMES = {'eval', 'exec', 'system', 'popen', 'load', 'loads', 'read_pickle', 'getattr', '__import__', 'open'}


class SafeUnpickler(pickle.Unpickler):
    seen = set()

    def find_class(self, module, name):
        self.seen.add((module, name))
        if name in REFUSED_NAMES:
            raise pickle.UnpicklingError(f'refused {module}.{name}')
        if module == 'builtins':
            if name not in ALLOWED_BUILTINS:
                raise pickle.UnpicklingError(f'refused builtins.{name}')
            return getattr(importlib.import_module('builtins'), name)
        if module == 'datetime' and name in ('timezone', 'timedelta', 'datetime', 'date', 'time'):
            return getattr(importlib.import_module('datetime'), name)
        if not module.startswith(ALLOWED_MODULE_PREFIXES):
            raise pickle.UnpicklingError(f'refused {module}.{name}')
        if module.startswith('numpy._core'):  # pickled with numpy 2, possibly loaded with numpy 1
            try:
                importlib.import_module(module)
            except ImportError:
                module = module.replace('numpy._core', 'numpy.core', 1)
        return getattr(importlib.import_module(module), name)


def safe_load(path):
    with open(path, 'rb') as f:
        return SafeUnpickler(f).load()
