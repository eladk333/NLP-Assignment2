import sys
import gensim.downloader as dl

model = dl.load("word2vec-google-news-300")
w1, w2, w3 = sys.argv[1:4]
s12 = model.similarity(w1, w2)
s13 = model.similarity(w1, w3)
print(f"sim({w1},{w2}) = {s12:.4f}  (synonym)")
print(f"sim({w1},{w3}) = {s13:.4f}  (antonym)")
print("condition sim(w1,w2) < sim(w1,w3):", s12 < s13)
