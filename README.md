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
