import os
import gensim.downloader as dl
from gensim.models import KeyedVectors

WIKI = "glove-wiki-gigaword-200"
TWITTER = "glove-twitter-200"
CACHE = os.path.expanduser("~/gensim-data/kv-cache")   # outside the repo


def load(name):
    """Load a gensim model; the slow text parse happens once, then it is cached as a mmap-able .kv file."""
    path = f"{CACHE}/{name}.kv"
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        dl.load(name).save(path)
    return KeyedVectors.load(path, mmap="r")


def load_models():
    return load(WIKI), load(TWITTER)
