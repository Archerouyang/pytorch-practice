import torch
from attention_utils import SelfAttention, visualize_attention
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Create pseudo data
batch_size = 1
seq_length = 6
embed_dim = 8
num_heads = 2

# Random sequence embeddings (could represent word embeddings)
pseudo_embeddings = torch.randn(batch_size, seq_length, embed_dim)

# Create self-attention module
self_attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)

# Process through self-attention
output, attention_weights = self_attention(pseudo_embeddings)

# Print shapes and sample values
print(f"Input shape: {pseudo_embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

print("\nSample of input embeddings:")
print(pseudo_embeddings[0, 0, :])

print("\nSample of output embeddings:")
print(output[0, 0, :])

print("\nSample of attention weights (Head 1, Token 1):")
print(attention_weights[0, 0, 0, :])

# Visualize attention weights with sample token names
token_names = ["Token1", "Token2", "Token3", "Token4", "Token5", "Token6"]
fig = visualize_attention(attention_weights, tokens=token_names)
plt.savefig("attention_visualization.png")
plt.show()

# Demonstrate how attention aggregates information
print("\nDemonstrating how attention works:")
for head_idx in range(num_heads):
    print(f"\nHead {head_idx+1} attention probabilities:")
    attn = attention_weights[0, head_idx].detach().numpy()
    
    for i in range(seq_length):
        max_idx = attn[i].argmax()
        print(f"  Token {i+1} ({token_names[i]}) attends most to: Token {max_idx+1} ({token_names[max_idx]}) with weight {attn[i, max_idx]:.4f}") 