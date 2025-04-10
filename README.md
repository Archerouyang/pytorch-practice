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
  # starter code
  # Self-Attention Initialization
  class SelfAttention(nn.Module):
      def __init__(self, embed_size, heads):
          super(SelfAttention, self).__init__()
          self.embed_size = embed_size
          self.heads = heads
          self.head_dim = embed_size // heads

### 2025-03-26 🚀
**Update** 
- *Transformer Decoder*

### 2025-04-10 🚀
**Update** 
- *Hands on LoRA*
  ```python
  # starter code
  class LinearLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0, merge=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge = merge

