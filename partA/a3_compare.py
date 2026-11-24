"""Usage: python a3_compare.py word1 word2 ...
Prints top-10 neighbors in the wiki+news GloVe and the twitter GloVe, and their overlap."""
import sys
from a3_common import load_models

wiki, twitter = load_models()
for w in sys.argv[1:]:
    if w not in wiki.key_to_index or w not in twitter.key_to_index:
        print(f"RES {w}: NOT IN BOTH VOCABS (wiki={w in wiki.key_to_index}, twitter={w in twitter.key_to_index})")
        continue
    nw = [n for n, _ in wiki.most_similar(w, topn=10)]
    nt = [n for n, _ in twitter.most_similar(w, topn=10)]
    common = [n for n in nw if n in set(nt)]
    print(f"RES {w}: overlap={len(common)}/10")
    print(f"RES   wiki   : {', '.join(nw)}")
    print(f"RES   twitter: {', '.join(nt)}")
    print(f"RES   shared : {', '.join(common) if common else '-'}")
