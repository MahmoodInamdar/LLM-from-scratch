# Chapter 2 File Descriptions

This document provides a summary of the purpose and usage of each file and subfolder in the `ch02` directory.

---

## 01_main-chapter-code/

- **ch02.ipynb**: The main notebook for Chapter 2, covering text data preparation, tokenization, embedding, and data loading for LLMs. It includes explanations, code, and visualizations for each step, from raw text to token embeddings, and demonstrates both custom and library-based tokenizers.
- **dataloader.ipynb**: A minimal notebook summarizing the main data loading pipeline from the chapter, focusing on the essential code for creating datasets and dataloaders for LLM training.
- **exercise-solutions.ipynb**: Contains solutions to the exercises from Chapter 2, including code for tokenization and data loading tasks.
- **README.md**: Briefly describes the contents of this folder and points to the main and optional notebooks.
- **the-verdict.txt**: The public domain short story "The Verdict" by Edith Wharton, used as sample text for tokenization and embedding demonstrations.

---

## 02_bonus_bytepair-encoder/

- **compare-bpe-tiktoken.ipynb**: Benchmarks and compares various byte pair encoding (BPE) implementations, including OpenAI's original, tiktoken, Hugging Face, and a from-scratch version. Includes performance and output comparisons.
- **bpe_openai_gpt2.py**: The original BPE encoder code used by OpenAI for GPT-2, adapted for educational use. Provides a Python implementation of the GPT-2 tokenizer.
- **requirements-extra.txt**: Lists extra Python dependencies needed for running the bonus notebook (e.g., `requests`, `tqdm`, `transformers`).
- **README.md**: Describes the contents and purpose of this bonus folder.
- **gpt2_model/**: Stores downloaded model files (e.g., `encoder.json`, `vocab.bpe`) required for the original GPT-2 BPE implementation.

---

## 03_bonus_embedding-vs-matmul/

- **embeddings-and-linear-layers.ipynb**: Explains and demonstrates the equivalence between embedding layers and linear layers applied to one-hot encoded vectors in PyTorch, with code and visualizations. Shows why embedding layers are more efficient for LLMs.
- **README.md**: Describes the purpose of this bonus folder and its notebook.

---

## 04_bonus_dataloader-intuition/

- **dataloader-intuition.ipynb**: Provides an intuitive explanation of the data loader and sliding window approach using simple number sequences instead of text. Helps clarify how input/target pairs are generated for LLM training.
- **README.md**: Describes the purpose of this bonus folder and its notebook.

---

## 05_bpe-from-scratch/

- **bpe-from-scratch.ipynb**: A standalone, educational notebook implementing the byte pair encoding (BPE) tokenization algorithm from scratch. Explains the algorithm, provides a trainable tokenizer, and demonstrates loading/saving vocabularies and merges. Also shows how to load the original GPT-2 tokenizer files.
- **README.md**: Describes the purpose of this bonus folder and its notebook.
- **tests/tests.py**: Contains pytest-based tests for the from-scratch BPE tokenizer, including training, encoding, decoding, and saving/loading functionality. Also handles downloading required files for testing.

---

## ch02/README.md

- **README.md**: Top-level readme for Chapter 2, summarizing the structure and purpose of the main and bonus code folders.

---

This document should help you quickly understand the role and usage of each file in the `ch02` directory and its subfolders. 