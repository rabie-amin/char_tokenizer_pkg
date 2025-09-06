# setup.py
from setuptools import setup, find_packages

setup(
    name="char_tokenizer",
    version="0.1.0",
    description="Character-level tokenizer and vocab utilities",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.0.0"
    ],
    python_requires=">=3.8",
)
