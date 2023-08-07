import os
import sys
import io
import contextlib

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)


result_dict = {1: 'Network is PAC-model robust.',
               0: 'Unsafe. Adversarial Example Found.',
               2: 'Unknown. Potential Counter-Example exists.'}


class Silence:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


@contextlib.contextmanager
def silence():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

class DummyFile:
    def write(self, x): pass