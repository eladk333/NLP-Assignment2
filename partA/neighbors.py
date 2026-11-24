import sys
import gensim.downloader as dl

model = dl.load("word2vec-google-news-300")
for w in sys.argv[1:]:
    if w not in model.key_to_index:
        print(f"{w}: NOT IN VOCAB")
        continue
    nn = [n for n, _ in model.most_similar(w, topn=10)]
    print(f"{w}: {', '.join(nn)}")
