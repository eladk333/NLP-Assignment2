"""B.0 warmup with roberta-base. Usage: python b0_warmup.py"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tok = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base").eval()


def encode(sentence):
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return enc, out.hidden_states[-1][0], out.logits[0]   # vectors: last layer, (seq, 768)


def word_vec(sentence, word):
    """Mean of the last-layer vectors of the subword tokens of `word` (first occurrence)."""
    words = sentence.split()
    enc, vecs, _ = encode(sentence)
    pos = [i for i, w in enumerate(enc.word_ids()) if w is not None and words[w] == word]
    return vecs[pos].mean(0)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# 1) "I am so <mask>"
sent = f"I am so {tok.mask_token}"
enc, vecs, logits = encode(sent)
toks = tok.convert_ids_to_tokens(enc.input_ids[0])
print("1) tokens:", toks)
for target in ["Ġam", tok.mask_token]:
    i = toks.index(target)
    print(f"1.1) vector for {target!r} (pos {i}): shape {tuple(vecs[i].shape)}, first 5 = {vecs[i][:5].tolist()}")
    probs = logits[i].softmax(-1)
    p, ids = probs.topk(5)
    preds = ", ".join(f"{tok.decode([j]).strip()!r}:{q:.3f}" for j, q in zip(ids.tolist(), p.tolist()))
    print(f"1.2) top-5 for {target!r}: {preds}")

# 2) + 3) same word in two sentences: cosine similarity of its contextual vectors
pairs = [
    ("bank", "I deposited my salary at the bank today .", "I withdrew some cash at the bank yesterday ."),
    ("bank", "I deposited my salary at the bank today .", "We had a picnic on the bank of the river ."),
    ("play", "The kids play football in the park .", "The children play soccer in the yard ."),
    ("play", "The kids play football in the park .", "We saw a great play at the theater ."),
    ("can", "I can swim very fast .", "She opened a can of beans ."),
    ("lead", "Exposure to lead is dangerous for children .", "She will lead the team to victory ."),
    ("fly", "Birds fly south in winter .", "There is a fly in my soup ."),
    ("right", "Turn right at the corner .", "You are right about that ."),
    # last-layer vectors are anisotropic (all share a common direction), so natural sentences stay >0.8;
    # a degenerate context pushes the vector further away:
    ("run", "He scored a run in the ninth inning .", "Hello world run"),
    ("bank", "We sat on the bank of the river .", "xyz qwe bank ! ! !"),
]
print("\n2/3) contextual cosine similarity of the same word:")
for w, s1, s2 in sorted(pairs, key=lambda p: -cos(word_vec(p[1], p[0]), word_vec(p[2], p[0]))):
    print(f"  {cos(word_vec(s1, w), word_vec(s2, w)):.3f}  {w!r}: {s1!r} | {s2!r}")

# 4) n words -> m > n tokens
s = "The unbelievably overcomplicated antidisestablishmentarianism argument bewildered everyone ."
toks = tok.tokenize(s)
print(f"\n4) {len(s.split())} words -> {len(toks)} tokens: {toks}")
