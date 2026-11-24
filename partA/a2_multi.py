import gensim.downloader as dl

model = dl.load("word2vec-google-news-300")
triplets = [("hot","scorching","cold"), ("hot","warm","cold"), ("big","huge","small"),
            ("happy","joyful","sad"), ("good","great","bad"), ("rich","wealthy","poor"),
            ("fast","quick","slow"), ("smart","intelligent","stupid")]
for w1, w2, w3 in triplets:
    s12 = model.similarity(w1, w2); s13 = model.similarity(w1, w3)
    print(f"RES ({w1}, {w2}, {w3}): syn={s12:.4f} ant={s13:.4f} holds={s12 < s13}")
