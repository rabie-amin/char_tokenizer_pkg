# char_tokenizer/vocab.py
import os
import json
import pandas as pd
from typing import List, Tuple, Dict


def collect_texts_from_path(path: str, column: str = "Input", encoding: str = "utf-8") -> List[str]:
    """Read text content from CSV or TXT file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, encoding=encoding)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {path}. Found: {list(df.columns)}")
        return df[column].astype(str).tolist()
    elif ext == ".txt":
        with open(path, "r", encoding=encoding) as f:
            return [f.read()]
    else:
        raise ValueError("Unsupported file type: use .csv or .txt")


def build_vocab_from_files(
    paths: List[str],
    column: str = "Input",
    encoding: str = "utf-8",
    pad_token: str = "[PAD]",
    unk_token: str = "[UNK]",
    save_dir: str = ".",
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Build a char-level vocab from files. Saves vocab, id2char, and debug char list.
    Returns (vocab, id2char).
    """
    all_chars = set()
    total_texts = 0
    total_chars = 0

    for p in paths:
        texts = collect_texts_from_path(p, column=column, encoding=encoding)
        total_texts += len(texts)
        for t in texts:
            all_chars.update(t)
            total_chars += len(t)

    unique_chars_sorted = sorted(all_chars)

    vocab = {pad_token: 0, unk_token: 1}
    for i, ch in enumerate(unique_chars_sorted, start=2):
        vocab[ch] = i

    id2char = {str(v): k for k, v in vocab.items()}

    os.makedirs(save_dir, exist_ok=True)
    vocab_path = os.path.join(save_dir, "char_vocab.json")
    id2char_path = os.path.join(save_dir, "id2char.json")
    charlist_path = os.path.join(save_dir, "char_list.txt")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(id2char_path, "w", encoding="utf-8") as f:
        json.dump(id2char, f, ensure_ascii=False, indent=2)
    with open(charlist_path, "w", encoding="utf-8") as f:
        for ch in unique_chars_sorted:
            visible = (
                "[SPACE]" if ch == " " else
                "[TAB]" if ch == "\t" else
                "[NEWLINE]" if ch == "\n" else
                ch
            )
            f.write(f"{visible}\n")

    print("==== Vocab build summary ====")
    print(f"Files processed        : {len(paths)}")
    print(f"Total text segments    : {total_texts}")
    print(f"Total characters seen  : {total_chars}")
    print(f"Unique characters      : {len(unique_chars_sorted)}")
    print(f"Vocab size (incl specials): {len(vocab)}")
    print(f"Saved: {vocab_path}")
    print(f"Saved: {id2char_path}")
    print(f"Saved (debug list): {charlist_path}")

    return vocab, id2char


def load_vocab(vocab_path: str) -> dict:
    """Load vocab (char -> id) JSON."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vocab(vocab: dict, save_dir: str = ".") -> str:
    """Save vocab (char -> id) as JSON to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    vocab_path = os.path.join(save_dir, "char_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab_path
