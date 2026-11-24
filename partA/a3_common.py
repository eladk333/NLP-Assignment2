import gensim.downloader as dl

WIKI = "glove-wiki-gigaword-200"
TWITTER = "glove-twitter-200"


def load_models():
    wiki = dl.load(WIKI)
    twitter = dl.load(TWITTER)
    return wiki, twitter
