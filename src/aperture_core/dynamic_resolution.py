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
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

        # Apply dynamic resolution to attention weights/dropout
        current_dropout_rate = (1.0 - resolve_level) * 0.45 + 0.05
        if current_dropout_rate > 0 and self.training:  # Only apply if training and rate > 0
            attn = F.dropout(attn, p=current_dropout_rate, training=self.training)

        # Apply a "blurring" or "sharpening" effect to attention logits
        attn = attn * (0.5 + 0.5 * resolve_level)  # More flat at low res, more peaked at high res

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)

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
