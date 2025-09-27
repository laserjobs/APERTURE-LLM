# src/aperture_core/multi_modal_fusion.py
import torch
import torch.nn as nn

class MultiModalFusionModule(nn.Module):
    """
    Fuses multi-modal inputs (text, image, audio) using cross-modal attention.
    Maps features to a unified embedding space for further processing by DRBlocks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Trainable modality-specific weights for initial scaling
        self.text_weight = nn.Parameter(torch.tensor(1.0))
        # Initialized to 0.1 for faster learning from random inputs
        self.image_weight = nn.Parameter(torch.tensor(0.1)) if config.raw_encoder.image.enabled else None
        self.audio_weight = nn.Parameter(torch.tensor(0.1)) if config.raw_encoder.audio.enabled else None
        
        # Cross-modal attention to enable deep interaction between modalities
        # nn.MultiheadAttention expects embed_dim, num_heads, dropout
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.model.embedding_dim,
            num_heads=config.model.num_heads,  # Reuse num_heads from model config
            dropout=0.1
        )
        
        # Linear projection and LayerNorm after attention
        self.proj = nn.Linear(config.model.embedding_dim, config.model.embedding_dim)
        self.norm = nn.LayerNorm(config.model.embedding_dim)
        self.dropout_layer = nn.Dropout(0.1) # Added for consistency

    def forward(self, text_features, image_features, audio_features):
        """
        Fuses text, image, and audio features using cross-modal attention.
        Args:
            text_features: (B, T_text, embedding_dim) or None
            image_features: (B, T_image, embedding_dim) or None
            audio_features: (B, T_audio, embedding_dim) or None
        Returns:
            fused_features: (B, T_fused, embedding_dim)
        """
        # Collect non-None features and apply modality weights
        features = []
        if text_features is not None and text_features.numel() > 0:
            features.append(self.text_weight * text_features)
        
        # Check if image encoder is enabled AND features are provided and non-empty
        if self.image_weight is not None and image_features is not None and image_features.numel() > 0:
            features.append(self.image_weight * image_features)
        
        # Check if audio encoder is enabled AND features are provided and non-empty
        if self.audio_weight is not None and audio_features is not None and audio_features.numel() > 0:
            features.append(self.audio_weight * audio_features)

        # Handle edge case: no valid features (should ideally not happen if text_features is always present)
        if not features:
            # Determine device from available (or default to 'cpu')
            device = 'cpu'
            if text_features is not None: device = text_features.device
            elif image_features is not None: device = image_features.device
            elif audio_features is not None: device = audio_features.device
            
            # Return a dummy tensor for batch size 1, 1 sequence element, embedding_dim
            # This case indicates an error in multi-modal setup if text is expected.
            return torch.zeros(1, 1, self.config.model.embedding_dim, device=device)

        # Concatenate features along sequence dimension
        # (B, T_text + T_image + T_audio, embedding_dim)
        fused = torch.cat(features, dim=1) 

        # Apply cross-modal attention
        # nn.MultiheadAttention expects input as (sequence_length, batch_size, embed_dim)
        fused = fused.transpose(0, 1)  # (T_fused, B, embedding_dim)
        
        # Query, Key, Value are all the same for self-attention within the fused features
        attn_output, _ = self.cross_attn(fused, fused, fused)  # (T_fused, B, embedding_dim)
        
        attn_output = attn_output.transpose(0, 1)  # Transpose back to (B, T_fused, embedding_dim)

        # Project, normalize, and dropout
        fused_features = self.norm(self.proj(attn_output))
        fused_features = self.dropout_layer(fused_features)
        
        return fused_features  # (B, T_fused, embedding_dim)
