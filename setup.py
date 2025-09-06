from setuptools import setup, find_packages

setup(
    name="char-tokenizer",
    version="0.1.0",
    description="Character-level tokenizer for NLP tasks",
    author="Rabie Otoum",
    author_email="rabie.amin.otoum@gmail.com",
    url="https://github.com/rabie-amin/char_tokenizer_pkg",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0",
    ],
    python_requires=">=3.7",
)
