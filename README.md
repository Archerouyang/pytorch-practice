# LLM Fundamentals from Scratch 🧠⚙️

**A self-education journey through core algorithms and architectures of large language models**

## 📖 Overview
This repository documents my hands-on learning journey through the fundamental components of modern language models. Each implementation is built from first principles **without using high-level ML frameworks** (PyTorch/TensorFlow) to deepen understanding.

**Core Philosophy**:  
"*What I cannot create, I do not understand*" - Richard Feynman

## 📚 Daily Log

### 2025-03-26 🚀
**Update** 
- *Repository initialization*
- *Basic functions in PyTorch*
- *Self-Attention from scratch*
  ```python
  # Self-Attention Initialization
  class SelfAttention(nn.Module):
      def __init__(self, embed_size, heads):
          super(SelfAttention, self).__init__()
          self.embed_size = embed_size
          self.heads = heads
          self.head_dim = embed_size // heads
