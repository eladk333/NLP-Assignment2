# B.0 Warmup on contextualized vectors

Code: `b0_warmup.py` (roberta-base, `AutoModelForMaskedLM`). Word vectors are the last hidden layer; a word split into several subword tokens gets the mean of its tokens' vectors.

## 1. "I am so `<mask>`"

Tokens: `<s> I Ġam Ġso <mask> </s>` (`Ġ` marks a leading space). The mask is fed as `tok.mask_token` (`<mask>`).

1.1 The vectors for `am` (position 2) and `<mask>` (position 4) are 768-d rows of the last hidden layer.

1.2 Top-5 predictions:

| Position | Predictions (probability) |
|---|---|
| `am` | am 1.000, is 0.000, 'm 0.000, was 0.000, feel 0.000 |
| `<mask>` | sorry 0.308, proud 0.065, grateful 0.058, happy 0.045, blessed 0.032 |

`am` is not masked, so the model sees it and simply predicts it back. For the mask it predicts emotion/state adjectives, as expected after "I am so".

## 2. High similarity (same word, same sense)

`play`: "The kids play football in the park ." vs "The children play soccer in the yard ." → cosine **0.995**

## 3. Low similarity

| Word | Sentence 1 | Sentence 2 | Cosine |
|---|---|---|---|
| lead | Exposure to lead is dangerous for children . | She will lead the team to victory . | 0.846 |
| bank | We sat on the bank of the river . | xyz qwe bank ! ! ! | **0.678** |

Different senses do get lower similarity (bank money vs. river: 0.909; can verb vs. noun: 0.890), but between natural sentences we could not get below ~0.85. RoBERTa's last-layer vectors are anisotropic: they all share a strong common direction, so the cosine between any two of them is high. To go lower we had to put the word in a meaningless context. The relative order is what matters: same sense ≈ 0.99, different senses ≈ 0.85–0.91.

## 4. n words → m > n tokens

"The unbelievably overcomplicated antidisestablishmentarianism argument bewildered everyone ." has **8 words → 15 tokens**:

`The Ġunbelievably Ġover compl icated Ġant idis establishment arian ism Ġargument Ġbewild ered Ġeveryone Ġ.`
