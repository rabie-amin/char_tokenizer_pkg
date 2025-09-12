# char_tokenizer/tokenizer.py
import json
import os
import hashlib
from typing import List, Optional
from .vocab import build_vocab_from_files, load_vocab


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
        self.id2char = {v: k for k, v in vocab.items()}  # int -> char

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        if max_length is not None:
            if len(ids) < max_length:
                ids += [self.pad_id] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
        return ids

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        return [self.encode(t, max_length=max_length) for t in texts]

    def decode(self, ids: List[int], strip_pad: bool = True, strip_unk: bool = True) -> str:
        chars = []
        for i in ids:
            if strip_pad and i == self.pad_id:
                continue
            if strip_unk and i == self.unk_id:
                continue
            chars.append(self.id2char.get(i, ""))
        return "".join(chars)

    @classmethod
    def from_json(cls, vocab_path: str, pad_token: str = "[PAD]", unk_token: str = "[UNK]"):
        vocab = load_vocab(vocab_path)
        return cls(vocab=vocab, pad_token=pad_token, unk_token=unk_token)

    @classmethod
    def from_dataset(
        cls,
        paths: List[str],
        column: str = "Input",
        save_root: Optional[str] = None,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
    ):
        """
        Build or load a vocab from dataset paths.
        - paths: list of CSV/TXT files
        - column: column name if CSV
        - save_root: where to save vocab (default ~/.char_tokenizer)
        """
        if save_root is None:
            save_root = os.path.expanduser("~/.char_tokenizer")

        # Hash dataset paths for uniqueness
        dataset_id = hashlib.md5("".join(paths).encode("utf-8")).hexdigest()[:8]
        save_dir = os.path.join(save_root, dataset_id)
        vocab_path = os.path.join(save_dir, "char_vocab.json")

        if os.path.exists(vocab_path):
            print(f"[CharTokenizer] Reusing existing vocab from {vocab_path}")
            vocab = load_vocab(vocab_path)
        else:
            print(f"[CharTokenizer] Building new vocab from dataset → {save_dir}")
            vocab, _ = build_vocab_from_files(
                paths=paths,
                column=column,
                save_dir=save_dir,
                pad_token=pad_token,
                unk_token=unk_token,
            )

        return cls(vocab=vocab, pad_token=pad_token, unk_token=unk_token)
