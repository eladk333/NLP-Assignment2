"""Automatic strategy: for the N most frequent words (of the wiki vocab) that exist in both models,
compute |top10_wiki ∩ top10_twitter| and rank. Usage: python a3_scan.py [N]  -> a3_scan_results.csv"""
import sys
import numpy as np
from a3_common import load_models

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
K = 10


def neighbors(model, words, k=K, chunk=100):
    vecs = model.get_normed_vectors()
    idx_all = np.array([model.key_to_index[w] for w in words])
    out = []
    for i in range(0, len(idx_all), chunk):
        idx = idx_all[i:i + chunk]
        sims = vecs[idx] @ vecs.T
        sims[np.arange(len(idx)), idx] = -np.inf          # exclude the word itself
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(len(idx)):
            order = part[r][np.argsort(-sims[r, part[r]])]
            out.append([model.index_to_key[j] for j in order])
    return out


wiki, twitter = load_models()
words = [w for w in wiki.index_to_key
         if w.isalpha() and len(w) >= 3 and w in twitter.key_to_index][:N]
print(f"RES scanning {len(words)} words")
nw = neighbors(wiki, words)
nt = neighbors(twitter, words)
rows = sorted(((len(set(a) & set(b)), w) for w, a, b in zip(words, nw, nt)), reverse=True)
with open("a3_scan_results.csv", "w") as f:
    f.write("word,overlap\n")
    for o, w in rows:
        f.write(f"{w},{o}\n")
print("RES most similar :", ", ".join(f"{w}({o})" for o, w in rows[:30]))
print("RES most different:", ", ".join(f"{w}({o})" for o, w in rows[-30:]))
