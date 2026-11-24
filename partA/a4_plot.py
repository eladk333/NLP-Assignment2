import gensim.downloader as dl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = dl.load("word2vec-google-news-300")
words = [w for w in model.index_to_key[1:5000] if w.endswith("ed") or w.endswith("ing")]
y = np.array([w.endswith("ing") for w in words])
X = np.stack([model[w] for w in words])
print(len(words), "words;", (~y).sum(), "ed,", y.sum(), "ing")

pca = decomposition.PCA(n_components=2).fit(X)
Z = pca.transform(X)
print("explained variance ratio:", pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())

plt.figure(figsize=(10, 8))
plt.scatter(*Z[~y].T, c="blue", s=12, label="ends with 'ed'")
plt.scatter(*Z[y].T, c="green", s=12, label="ends with 'ing'")
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (2-D) of word2vec vectors: words ending in 'ed' vs 'ing'")
plt.savefig("partA/a4_plot.png", dpi=150)

clf = LogisticRegression(max_iter=2000)
for name, F in [("2-D PCA", Z), ("full 300-D", X)]:
    print(f"logreg 5-fold CV acc, {name}: {cross_val_score(clf, F, y, cv=5).mean():.3f}")
print("majority baseline:", max(y.mean(), 1 - y.mean()))
