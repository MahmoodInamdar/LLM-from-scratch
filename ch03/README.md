# Chapter 3: Coding Attention Mechanisms

&nbsp;
## Main Chapter Code

- [01_main-chapter-code](01_main-chapter-code) contains the main chapter code.

&nbsp;
## Bonus Materials

- [02_bonus_efficient-multihead-attention](02_bonus_efficient-multihead-attention) implements and compares different implementation variants of multihead-attention
- [03_understanding-buffers](03_understanding-buffers) explains the idea behind PyTorch buffers, which are used to implement the causal attention mechanism in chapter 3


In the video below, I provide a code-along session that covers some of the chapter contents as supplementary material.

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/-Ll8DtpNtvk/0.jpg)](https://www.youtube.com/watch?v=-Ll8DtpNtvk)



## Folder Structure & File Purposes

### 01_main-chapter-code/
- **ch03.ipynb**: Main notebook for Chapter 3. Covers theory and implementation of attention mechanisms, especially self-attention and multi-head attention, the core of transformer models. Includes:
  - Motivation for attention (why RNNs struggle with long sequences)
  - Mechanics of self-attention (with and without trainable weights)
  - Step-by-step code for building attention layers from scratch
  - Visualizations and explanations for each step
  - Practical PyTorch code for hands-on learning
- **multihead-attention.ipynb**: Minimal, focused notebook implementing multi-head attention and the data loading pipeline. Great for practicing the core ideas without extra distractions.
- **exercise-solutions.ipynb**: Worked solutions to the chapter's exercises. Excellent for self-testing and seeing how to implement and debug attention mechanisms.
- **small-text-sample.txt**: Small text file for quick experiments and data loading in the notebooks.
- **README.md**: Summarizes the folder and points to the main and optional code.

### 02_bonus_efficient-multihead-attention/
- **mha-implementations.ipynb**: Deep dive into different ways to implement multi-head attention, with performance benchmarks and comparisons. For advanced learners who want to optimize or understand trade-offs in different implementations.
- **README.md**: Explains the purpose of the notebook and summarizes benchmark results with helpful figures.

### 03_understanding-buffers/
- **understanding-buffers.ipynb**: Explains PyTorch buffers, which are used to store non-trainable tensors (like attention masks) that need to move with the model between CPU and GPU. Crucial for implementing causal (masked) attention efficiently.
- **README.md**: Summarizes the notebook and links to a video tutorial.

---

## Key Concepts & Learning Path

### 1. Why Attention?
- **Problem**: RNNs and LSTMs struggle with long-range dependencies in sequences (e.g., translating long sentences).
- **Solution**: Attention allows the model to "focus" on relevant parts of the input, regardless of their position.

### 2. Self-Attention (Core of Transformers)
- **Idea**: Each token in a sequence can "attend" to every other token, learning which are most relevant for the current prediction.
- **How**:  
  - Compute attention scores between all pairs of tokens.
  - Normalize scores (softmax) to get attention weights.
  - Use these weights to compute a weighted sum of the input vectors (context vectors).

**Example (PyTorch):**
```python
import torch
import torch.nn.functional as F

# Example input: 3 tokens, each with 4 features
x = torch.randn(3, 4)
# Compute attention scores (dot product)
scores = x @ x.T
# Normalize
weights = F.softmax(scores, dim=-1)
# Weighted sum
context = weights @ x
print(context)
```

### 3. Multi-Head Attention
- **Why**: Instead of a single attention calculation, use several in parallel ("heads") to capture different types of relationships.
- **How**:  
  - Project inputs into multiple sets of queries, keys, and values.
  - Compute attention for each head.
  - Concatenate and project the results.

**Example (PyTorch, simplified):**
```python
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)
        attn_scores = (q @ k.transpose(-2, -1)) / self.d_k**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(attn_output)
```

### 4. Causal (Masked) Attention
- **Why**: In language modeling, you want to prevent the model from "seeing the future" (i.e., attending to tokens that come after the current one).
- **How**: Use a mask (upper-triangular matrix) to block attention to future tokens.

**Example:**
```python
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

### 5. PyTorch Buffers
- **What**: Non-trainable tensors registered with a module (e.g., attention masks).
- **Why**: They move with the model between CPU/GPU, but aren't updated by optimizers.
- **How**:  
  ```python
  self.register_buffer('mask', mask_tensor)
  ```

---

## How to Practice and Learn

1. **Read and Run the Notebooks**  
   Start with `ch03.ipynb` for theory and step-by-step code.  
   Use `multihead-attention.ipynb` for focused practice.

2. **Try the Exercises**  
   Open `exercise-solutions.ipynb`, hide the solutions, and try to solve the exercises yourself.  
   Example: Implement a self-attention layer from scratch, then compare your code to the solution.

3. **Experiment**  
   - Change the input data (e.g., use your own sentences).
   - Modify the number of heads, embedding sizes, or mask shapes.
   - Visualize attention weights (e.g., using matplotlib).

4. **Deepen Your Understanding**  
   - Read `understanding-buffers.ipynb` to see how PyTorch handles non-trainable tensors.
   - Explore `mha-implementations.ipynb` for performance and implementation trade-offs.

5. **Watch the Linked Videos**  
   The README files link to video tutorials that walk through the code and concepts.

---

## Example Practice Task

**Task:**  
Implement a simple self-attention mechanism for a batch of sequences.

**Step-by-step:**
1. Create random input data: `inputs = torch.randn(batch_size, seq_len, d_model)`
2. Write code to compute queries, keys, values (linear projections).
3. Compute attention scores and apply softmax.
4. Use the attention weights to get context vectors.
5. Compare your output to the one from PyTorch's built-in `nn.MultiheadAttention`.


