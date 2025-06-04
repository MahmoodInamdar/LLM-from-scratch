# LLMs from Scratch: Building Large Language Models from the Ground Up

## Project Overview

**LLMs from Scratch** is a comprehensive, hands-on codebase and educational resource for building, pretraining, and fine-tuning GPT-like Large Language Models (LLMs) entirely from first principles.

The project demystifies the inner workings of modern LLMs by guiding users through every stage of development, from text preprocessing and attention mechanisms to full GPT model implementation, pretraining, and advanced fine-tuning techniques. All code is written in Python, leveraging PyTorch for deep learning, and is designed to be accessible on standard laptops, with optional GPU acceleration.

---

## Key Features

- **Step-by-Step LLM Construction:** Learn to build a transformer-based GPT model from scratch, mirroring the architecture and training strategies of state-of-the-art models like ChatGPT.
- **Educational Focus:** Each chapter and appendix is structured to teach both the theory and practical implementation of LLMs, making this repository ideal for students, researchers, and practitioners.
- **Pretraining & Fine-tuning:** Includes scripts and notebooks for unsupervised pretraining, supervised fine-tuning for classification, and instruction-following tasks.
- **Parameter-Efficient Techniques:** Explore advanced topics such as LoRA (Low-Rank Adaptation) for efficient fine-tuning and distributed training.
- **Extensive Bonus Material:** Dive into appendices and bonus folders covering PyTorch fundamentals, tokenizer implementation, performance optimization, and more.
- **Cross-Platform & Cloud Ready:** Designed to run on Linux, macOS, and Windows, with support for cloud platforms like Google Colab and Lightning AI Studio.

---

## Repository Structure

- **Chapters 1–7:** Each chapter folder contains Jupyter notebooks and Python scripts corresponding to a key stage in LLM development:
  - **ch01:** LLM theory and architecture overview
  - **ch02:** Text data processing, tokenization, and dataloaders
  - **ch03:** Attention mechanisms and multi-head attention
  - **ch04:** Full GPT model implementation and transformer blocks
  - **ch05:** Unsupervised pretraining, training loops, and evaluation
  - **ch06:** Fine-tuning for text classification and transfer learning
  - **ch07:** Instruction-following fine-tuning and evaluation

- **Appendices:**
  - **appendix-A:** PyTorch introduction, distributed training, and deep learning basics
  - **appendix-D:** Advanced training techniques, learning rate scheduling, and performance monitoring
  - **appendix-E:** Parameter-efficient fine-tuning with LoRA

- **Setup:** Guides for Python environment setup, Docker, VSCode extensions, and cloud deployment.

- **Bonus Material:** Additional experiments, tokenizer comparisons, user interface demos, and performance tips.

---

## Technical Stack

- **Programming Language:** Python (>=3.10)
- **Deep Learning Framework:** PyTorch (>=2.3.0)
- **Other Key Libraries:** JupyterLab, tiktoken, matplotlib, tensorflow, tqdm, numpy, pandas, psutil
- **Testing & CI:** Automated tests for Linux, Windows, and macOS via GitHub Actions

---

## Hardware & Accessibility

- **Runs on Standard Laptops:** All main code is optimized for CPU execution, with automatic GPU utilization if available.
- **Cloud Compatible:** Easily run notebooks on Google Colab or Lightning AI Studio for faster training and experimentation.

---

## Educational Value

This project is designed for:

- **Students & Learners:** Gain a deep, practical understanding of LLMs by building every component yourself.
- **Researchers:** Use as a reference implementation for custom LLM research and experimentation.
- **Developers & Practitioners:** Adapt the codebase for real-world NLP projects, custom model development, and rapid prototyping.
- **Educators:** Leverage the clear, modular code and notebooks for teaching advanced NLP and deep learning concepts.

---

## Use Cases

- Building and understanding transformer-based LLMs from scratch
- Experimenting with pretraining and fine-tuning strategies
- Developing custom NLP models for research or production
- Learning and teaching the foundations of modern AI

---

## Getting Started

- **Quickstart:** Install dependencies with `pip install -r requirements.txt`
- **Setup Guides:** See the `setup/` directory for detailed instructions on Python, Docker, and cloud environments
- **Run Notebooks:** Open any chapter or appendix notebook in JupyterLab or your preferred editor

---

## License & Citation

- **License:** Apache 2.0 (permissive, suitable for educational and commercial use)
- **Citation:**  
  Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.  


---

## Acknowledgments

This repository is based on the book "Build a Large Language Model (From Scratch)" by Sebastian Raschka and is maintained with the support of the open-source and AI research communities.

---

**This project is a one-stop resource for anyone who wants to truly understand, build, and experiment with Large Language Models—empowering the next generation of AI practitioners and researchers.**
