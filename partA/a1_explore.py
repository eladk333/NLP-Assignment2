import gensim.downloader as dl

model = dl.load("word2vec-google-news-300")

candidates = ["bank", "bat", "apple", "java", "mouse", "python", "crane", "bass",
              "spring", "match", "light", "palm", "jaguar", "mercury", "club",
              "seal", "date", "table", "bow", "cell", "rock", "pitcher", "nut",
              "Turkey", "chair", "organ", "star", "current", "plant", "fan"]

for w in candidates:
    if w not in model.key_to_index:
        print(f"{w}: NOT IN VOCAB"); continue
    nn = [n for n, _ in model.most_similar(w, topn=10)]
    print(f"{w}: {', '.join(nn)}")
