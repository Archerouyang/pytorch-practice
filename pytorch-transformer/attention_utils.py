import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    """
    Basic implementation of self-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert (self.head_dim * num_heads == embed_dim), "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        
        # Linear projections
        q = self.query(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attention_weights, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        
        # Final projection
        out = self.out_proj(out)
        
        return out, attention_weights

def create_padding_mask(seq, pad_idx=0):
    """Creates a mask for padding tokens."""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    """Creates a causal mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask

def visualize_attention(attention_weights, tokens=None, cmap='viridis'):
    """
    Visualize attention weights as a heatmap.
    
    Parameters:
    - attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
    - tokens: List of tokens corresponding to the sequence
    - cmap: Colormap for the heatmap
    """
    # Take the first batch
    attention = attention_weights[0].detach().cpu().numpy()
    
    fig, axes = plt.subplots(nrows=1, ncols=attention.shape[0], 
                             figsize=(attention.shape[0] * 4, 4))
    
    if attention.shape[0] == 1:
        axes = [axes]
    
    for i, head_attention in enumerate(attention):
        ax = axes[i]
        im = ax.imshow(head_attention, cmap=cmap)
        ax.set_title(f"Head {i+1}")
        
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
        
    plt.colorbar(im, ax=axes)
    plt.tight_layout()
    return fig 