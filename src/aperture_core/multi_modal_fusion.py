# src/prometheus_core/multi_modal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusionModule(nn.Module):
    """
    Unifies aliased feature streams from different modalities.
    For this prototype, it primarily processes text embeddings,
    but is structured to conditionally handle image and audio.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text projection is always assumed for the current prototype
        self.text_proj = nn.Linear(config.raw_encoder.text.char_embed_dim * config.raw_encoder.text.multi_freq_components, 
                                   config.model.embedding_dim)
        
        # Conditionally initialize other modality projectors
        self.image_proj = None
        if hasattr(config.raw_encoder, 'image') and getattr(config.raw_encoder.image, 'enabled', False): # Check if 'image' section exists AND is enabled
            self.image_proj = nn.Linear(config.model.embedding_dim, config.model.embedding_dim) 
        
        self.audio_proj = None
        if hasattr(config.raw_encoder, 'audio') and getattr(config.raw_encoder.audio, 'enabled', False): # Check if 'audio' section exists AND is enabled
            self.audio_proj = nn.Linear(config.model.embedding_dim, config.model.embedding_dim)

        self.norm = nn.LayerNorm(config.model.embedding_dim)
        
        # A simple fusion MLP if multiple modalities are present.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.model.embedding_dim, config.model.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(config.model.embedding_dim * 2, config.model.embedding_dim)
        )

    def forward(self, text_features, image_features=None, audio_features=None):
        # text_features: (B, T_text, raw_encoder_output_dim) - always expected
        # image_features: (B, T_image, raw_encoder_output_dim_image) or (B, 1, embedding_dim) for dummy
        # audio_features: (B, T_audio, raw_encoder_output_dim_audio) or (B, 1, embedding_dim) for dummy

        all_modal_features_to_fuse = []

        # Process text features (always present for this prototype's main functionality)
        if text_features is not None and text_features.numel() > 0:
            all_modal_features_to_fuse.append(self.text_proj(text_features))
        else:
            raise ValueError("Text features cannot be empty in Prometheus prototype as it's the primary modality.")

        # Conditionally process image features
        if self.image_proj and image_features is not None and image_features.numel() > 0:
            all_modal_features_to_fuse.append(self.image_proj(image_features))
        
        # Conditionally process audio features
        if self.audio_proj and audio_features is not None and audio_features.numel() > 0:
            all_modal_features_to_fuse.append(self.audio_proj(audio_features))

        if len(all_modal_features_to_fuse) > 1:
            # For simplicity in prototype: Pad/trim all features to the max sequence length
            max_len = max(f.size(1) for f in all_modal_features_to_fuse)
            
            # Pad / trim features to match max_len
            padded_features = []
            for f in all_modal_features_to_fuse:
                if f.size(1) > max_len: # Trim if longer
                    padded_features.append(f[:, :max_len, :])
                elif f.size(1) < max_len: # Pad if shorter
                    pad_size = max_len - f.size(1)
                    # Pad (left, right, top, bottom) for last two dims
                    padded_features.append(F.pad(f, (0, 0, 0, pad_size))) 
                else:
                    padded_features.append(f)
            
            # Stack features and average them for a simple fusion in the prototype
            # (B, num_active_modalities, max_T, embedding_dim) -> (B, max_T, embedding_dim)
            fused_features = torch.stack(padded_features, dim=1).mean(dim=1)
            
            # Apply a simple fusion MLP
            fused_features = self.fusion_mlp(fused_features)
        else:
            # If only one modality (text) is active, just return its projected features
            fused_features = all_modal_features_to_fuse[0]

        return self.norm(fused_features) # (B, T_fused, embedding_dim)
