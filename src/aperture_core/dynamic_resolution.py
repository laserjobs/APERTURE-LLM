# src/aperture_core/dynamic_resolution.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicResolutionAttention(nn.Module):
    """
    Implements an attention mechanism with dynamic resolution.
    'resolve_level' (0.0 to 1.0) controls the granularity of attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.model.embedding_dim % config.model.num_heads == 0
        self.head_dim = config.model.embedding_dim // config.model.num_heads
        self.num_heads = config.model.num_heads

        self.query = nn.Linear(config.model.embedding_dim, config.model.embedding_dim)
        self.key = nn.Linear(config.model.embedding_dim, config.model.embedding_dim)
        self.value = nn.Linear(config.model.embedding_dim, config.model.embedding_dim)
        self.proj = nn.Linear(config.model.embedding_dim, config.model.embedding_dim)
        
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x, resolve_level=1.0):
        # x: (B, T, embedding_dim)
        # resolve_level: float, 0.0 (low resolution) to 1.0 (high resolution)

        B, T, C = x.size()

        # Compute Q, K, V
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5)) # (B, nh, T, T)

        # Apply dynamic resolution to attention weights/dropout
        # Lower resolve_level -> higher effective dropout / blurrier attention
        # Higher resolve_level -> lower effective dropout / sharper attention
        
        # Option 1: Modulate dropout rate based on resolve_level
        # Min dropout 0.05 at max res, max dropout 0.5 at min res
        current_dropout_rate = (1.0 - resolve_level) * 0.45 + 0.05 
        if current_dropout_rate > 0 and self.training: # Only apply if training and rate > 0
            attn = F.dropout(attn, p=current_dropout_rate, training=self.training)

        # Option 2: Apply a "blurring" or "sharpening" effect to attention logits
        # This is a conceptual implementation. A real one might use kernel convolutions or learnable filters.
        # Here, we'll just scale the magnitude of attention values based on resolve_level,
        # effectively making attention distributions flatter at low resolution and sharper at high resolution.
        # The scaling factor ensures it's 1.0 at max resolve_level and smaller at lower levels.
        attn = attn * (0.5 + 0.5 * resolve_level) # More flat at low res, more peaked at high res

        attn = F.softmax(attn, dim=-1) # (B, nh, T, T)
        # Standard dropout after softmax is still applied
        attn = self.attn_dropout(attn) 

        # Weighted sum of values
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        return self.resid_dropout(self.proj(y))


class DRBlock(nn.Module):
    """
    A single APERTURE-LLM block incorporating Dynamic Resolution Attention.
    Represents an 'Iterative Processing Block'.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.model.embedding_dim)
        self.attn = DynamicResolutionAttention(config)
        self.ln2 = nn.LayerNorm(config.model.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.model.embedding_dim, 4 * config.model.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.model.embedding_dim, config.model.embedding_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, resolve_level=1.0):
        # 'resolve_level' is passed down to the attention mechanism
        x = x + self.attn(self.ln1(x), resolve_level)
        x = x + self.mlp(self.ln2(x))
        return x
