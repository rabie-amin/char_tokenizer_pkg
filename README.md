# char_tokenizer_pkg
Arabic NLP / Character-level tokenizer
# char_tokenizer

Character-level tokenizer and vocabulary builder for NLP projects.

Usage:

```python
from char_tokenizer import build_vocab_from_files, CharTokenizer

vocab, id2char = build_vocab_from_files([train.csv, val.csv, test.csv], column="Input", save_dir="vocab_dir")
tokenizer = CharTokenizer.from_json("vocab_dir/char_vocab.json")

ids = tokenizer.encode("السلام", max_length=50)
text = tokenizer.decode(ids)

---

# 2) How to use these modules locally (examples)

```python
from char_tokenizer import build_vocab_from_files, CharTokenizer, load_vocab

# 1) Build vocab from CSVs (run once)
vocab, id2char = build_vocab_from_files(
    paths=["/path/to/train.csv", "/path/to/val.csv", "/path/to/test.csv"],
    column="Input",
    save_dir="/path/to/save/vocab"
)

# 2) Load tokenizer
tokenizer = CharTokenizer.from_json("/path/to/save/vocab/char_vocab.json")

# 3) Encode / decode
s = "قوله :"
ids = tokenizer.encode(s, max_length=128)
decoded = tokenizer.decode(ids)
print(ids[:len(s)], decoded)
