import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0, merge=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge = merge
        
        self.linear = nn.Linear(in_features, out_features)
        # shape of linear weight: (out_features, in_features)
        # input x shape: (batch_size, seq_len, in_features)
        # calculation: x @ weight.T
        # weight shape: (out_features, in_features)

        if rank > 0:
            self.lora_a = nn.Parameter(torch.randn(out_features, rank))
            # gausian distribution
            nn.init.kaiming_uniform_(self.lora_a, a = math.sqrt(5))

            self.lora_b = nn.Parameter(torch.randn(rank, in_features))
            self.scale = alpha / rank
        # Initialize LoRA matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        # Apply dropout to input
        x = self.dropout_layer(x)
        
        # Compute LoRA adaptation
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return lora_output
        
    def merge_weights(self):
        if not self.merge:
            self.merge = True
            # Merge LoRA weights into a single matrix
            self.merged_weight = self.lora_B @ self.lora_A * self.scaling
            
    def unmerge_weights(self):
        if self.merge:
            self.merge = False
            # Remove merged weights
            del self.merged_weight 