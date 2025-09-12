# char_tokenizer/tokenizer.py
from typing import List, Optional


class CharTokenizer:
    """
    Simple character-level tokenizer.

    vocab: dict mapping char -> int (e.g. {"[PAD]":0, "[UNK]":1, "ا": 2, ...})
    """

    def __init__(self, vocab: dict, pad_token: str = "[PAD]", unk_token: str = "[UNK]"):
        self.vocab = vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = vocab[pad_token]
        self.unk_id = vocab[unk_token]
        # id2char: int -> char
        self.id2char = {v: k for k, v in vocab.items()}

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode a single string into list of ids (pad/truncate if max_length given)."""
        ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        if max_length is not None:
            if len(ids) < max_length:
                ids += [self.pad_id] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
        return ids

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """Encode a batch of texts into list of list of ids."""
        return [self.encode(t, max_length=max_length) for t in texts]

    def decode(self, ids: List[int], strip_pad: bool = True, strip_unk: bool = True) -> str:
        """Convert list of ids back to string. Default skips PAD and UNK tokens."""
        chars = []
        for i in ids:
            if strip_pad and i == self.pad_id:
                continue
            if strip_unk and i == self.unk_id:
                continue
            ch = self.id2char.get(i, "")
            chars.append(ch)
        return "".join(chars)

    @classmethod
    def from_json(cls, vocab_path: str, pad_token: str = "[PAD]", unk_token: str = "[UNK]"):
        """Load a tokenizer directly from a saved vocab.json file."""
        import json
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab, pad_token=pad_token, unk_token=unk_token)
