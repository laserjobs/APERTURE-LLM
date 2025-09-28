import torch
import torch.nn as nn


class MultiFrequencyCharEmbedding(nn.Module):
    """
    Multi-frequency embeddings for raw characters.
    Combines embeddings from different 'scales'.
    """
    def __init__(self, vocab_size, char_embed_dim, multi_freq_components):
        super().__init__()
        self.char_embed_dim = char_embed_dim
        self.total_embed_dim = char_embed_dim * multi_freq_components
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, char_embed_dim) for _ in range(multi_freq_components)
        ])

    def forward(self, idx):
        # idx: (B, T) tensor of character indices
        embeds = [emb(idx) for emb in self.embeddings]
        return torch.cat(embeds, dim=-1)  # (B, T, total_embed_dim)


class UniversalRawTextEncoder(nn.Module):
    """
    Encodes raw character streams directly.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Output dimension from MultiFrequencyCharEmbedding (e.g., 96 in your config)
        multi_freq_output_dim = config.raw_encoder.text.char_embed_dim * \
            config.raw_encoder.text.multi_freq_components

        # The final desired embedding dimension for the model (e.g., 128 from model_config.yaml)
        final_embedding_dim = config.model.embedding_dim

        self.multi_freq_embed = MultiFrequencyCharEmbedding(
            config.model.vocab_size,
            config.raw_encoder.text.char_embed_dim,
            config.raw_encoder.text.multi_freq_components
        )
        
        # Positional encoder should now match the *final* embedding dimension
        self.pos_encoder = nn.Embedding(config.model.block_size, final_embedding_dim)
        self.dropout = nn.Dropout(0.1)

        # Add a projection layer if the multi-frequency output doesn't match the final embedding dim
        if multi_freq_output_dim != final_embedding_dim:
            self.projection = nn.Linear(multi_freq_output_dim, final_embedding_dim)
        else:
            self.projection = nn.Identity()  # Use Identity if no projection needed (dimensions already match)

        # Update the encoder's advertised output dimension to the final model embedding dimension
        self.output_dim = final_embedding_dim  # This will now be 128

    def forward(self, raw_char_indices):
        # raw_char_indices: (B, T) tensor of character indices
        B, T = raw_char_indices.shape
        
        # Get multi-frequency character embeddings
        x = self.multi_freq_embed(raw_char_indices)  # (B, T, multi_freq_output_dim=96)
        
        # Project to the final embedding dimension if necessary
        x = self.projection(x)  # (B, T, final_embedding_dim=128)

        # Add positional embeddings (which also output final_embedding_dim=128)
        pos = torch.arange(0, T, dtype=torch.long, device=raw_char_indices.device)  # (T)
        x = self.dropout(x + self.pos_encoder(pos))  # (B, T, final_embedding_dim=128)
        return x


class UniversalRawImageEncoder(nn.Module):
    """
    Processes raw RGB pixels without external libraries (e.g., torchvision).
    Uses patch-based embedding to map (B, C, H, W) to (B, T_image, embedding_dim).
    Assumes fixed input image dimensions for simplicity in prototype.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Default/Expected input shape. Can be made configurable if needed.
        # For a 224x224 image with 16x16 patches, this means 14x14 = 196 patches.
        self.expected_C = 3
        self.expected_H = 224
        self.expected_W = 224
        
        self.patch_size = 16
        if self.expected_H % self.patch_size != 0 or self.expected_W % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size")

        self.num_patches = (self.expected_H // self.patch_size) * (self.expected_W // self.patch_size)
        # This correctly projects to config.model.embedding_dim (128)
        self.patch_embed = nn.Linear(self.expected_C * self.patch_size * self.patch_size, config.model.embedding_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, config.model.embedding_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, raw_pixels):
        # raw_pixels: (B, C, H, W)
        if raw_pixels.numel() == 0:  # Handle empty tensor if no image input
            # Return a tensor compatible with fusion: (B, T_image_dummy=1, embedding_dim)
            return torch.empty(raw_pixels.size(0), 0, self.config.model.embedding_dim, device=raw_pixels.device)
        
        B, C, H, W = raw_pixels.shape
        if C != self.expected_C or H != self.expected_H or W != self.expected_W:
            raise ValueError(f"UniversalRawImageEncoder expected input shape (B, {self.expected_C}, "
                             f"{self.expected_H}, {self.expected_W}), but got {raw_pixels.shape}. "
                             "Adjust raw_encoders.py or input data.")
        
        # Extract patches using unfold:
        # (B, C, H, W) -> (B, C, H/p, p, W/p, p) -> (B, C, H/p, W/p, p, p)
        patches = raw_pixels.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape to (B, num_patches, C*p*p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, self.num_patches, -1)
        
        x = self.patch_embed(patches)  # (B, num_patches, embedding_dim=128)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x  # (B, T_image=num_patches, embedding_dim=128)


class UniversalRawAudioEncoder(nn.Module):
    """
    Processes raw waveforms (PCM samples) without external libraries (e.g., torchaudio).
    Uses a simple FFT-based transform to map (B, Samples) to (B, T_audio, embedding_dim).
    Assumes fixed total samples for simplicity in prototype.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_size = 1024  # Number of samples per FFT window
        self.overlap = self.window_size // 4  # For overlapping windows, if needed; kept simple for now
        self.hop_length = self.window_size - self.overlap  # Stride for windowing; simple for now
        
        self.num_segments = 128  # Fixed number of time segments
        self.expected_samples = (self.window_size +
                                 (self.num_segments - 1) * self.hop_length)

        # This correctly projects to config.model.embedding_dim (128)
        self.proj = nn.Linear(self.window_size // 2 + 1,
                              config.model.embedding_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_segments, config.model.embedding_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, raw_waveform):
        # raw_waveform: (B, Samples)
        if raw_waveform.numel() == 0:  # Handle empty tensor if no audio input
            # Return a tensor compatible with fusion: (B, T_audio_dummy=1, embedding_dim)
            return torch.empty(raw_waveform.size(0), 0, self.config.model.embedding_dim, device=raw_waveform.device)
        
        B, S = raw_waveform.shape
        if S < self.expected_samples:
            # Pad if shorter than expected input for fixed segments
            raw_waveform = torch.nn.functional.pad(raw_waveform, (0, self.expected_samples - S))
        elif S > self.expected_samples:
            # Truncate if longer than expected input
            raw_waveform = raw_waveform[:, :self.expected_samples]
        
        # Create overlapping windows manually for the prototype
        # This is a simplified version of what torchaudio.transforms.Spectrogram would do
        windows = raw_waveform.unfold(dimension=-1, size=self.window_size, step=self.hop_length)  # (B, num_segments, window_size)
        
        # Apply FFT to each window
        # torch.fft.rfft returns complex numbers; .abs() takes magnitude
        fft_magnitude = torch.fft.rfft(windows, dim=-1).abs()  # (B, num_segments, window_size//2 + 1)
        
        x = self.proj(fft_magnitude)  # (B, num_segments, embedding_dim=128)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x  # (B, T_audio=num_segments, embedding_dim=128)
