import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_STATS_PATH = {
    'fineweb_100b': os.path.join(_PACKAGE_DIR, 'data_stats', 'fineweb_100b'),
}

MAX_LENGTH = 128
