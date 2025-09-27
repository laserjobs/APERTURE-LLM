# src/aperture_core/model.py
import torch
import torch.nn as nn
from src.aperture_core.raw_encoders import UniversalRawTextEncoder, UniversalRawImageEncoder, UniversalRawAudioEncoder
from src.aperture_core.multi_modal_fusion import MultiModalFusionModule
from src.aperture_core.dynamic_resolution import DRBlock
from src.aperture_core.output_convergence import NonLinearOutputConvergence

class APERTURE_LLM(nn.Module):
    """
    The APERTURE-LLM: An Adaptive Perception & Resolution LLM - The Ultimate Generative Model.
    Abolishes tokenization by processing raw digital inputs.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Universal Raw Digital Encoding (Replaces Tokenization)
        # Raw text encoder is always assumed for this prototype's main functionality
        self.raw_text_encoder = UniversalRawTextEncoder(config)
        
        # Conditionally initialize other raw encoders based on config
        self.raw_image_encoder = None
        # Check if 'image' attribute exists AND is truthy (e.g., config.raw_encoder.image.enabled is True)
        if hasattr(config.raw_encoder, 'image') and getattr(config.raw_encoder.image, 'enabled', False):
             self.raw_image_encoder = UniversalRawImageEncoder(config)
        
        self.raw_audio_encoder = None
        # Check if 'audio' attribute exists AND is truthy
        if hasattr(config.raw_encoder, 'audio') and getattr(config.raw_encoder.audio, 'enabled', False):
             self.raw_audio_encoder = UniversalRawAudioEncoder(config)

        # 2. Multi-Modal Fusion Module
        self.multi_modal_fusion = MultiModalFusionModule(config)
        
        # 3. Iterative Processing Blocks (Dynamic Resolution Attention)
        self.dr_blocks = nn.ModuleList([DRBlock(config) for _ in range(config.model.num_layers)])
        self.ln_f = nn.LayerNorm(config.model.embedding_dim) # Final LayerNorm

        # 4. Non-linear Output Convergence Head
        self.output_convergence = NonLinearOutputConvergence(config) # Renamed from self.output_head for consistency

        print(f"APERTURE-LLM Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")

    def forward(self, raw_text_input, raw_image_input=None, raw_audio_input=None, focus_strength=0.0):
        # raw_text_input: (B, T) tensor of raw character indices
        # raw_image_input: (B, C, H, W) for image encoder
        # raw_audio_input: (B, Samples) for audio encoder
        
        # 1. Encode Raw Digital Inputs
        text_features = self.raw_text_encoder(raw_text_input) if raw_text_input is not None else None
        
        image_features = None
        if self.raw_image_encoder is not None:
             image_features = self.raw_image_encoder(
                 raw_image_input if raw_image_input is not None else torch.empty(raw_text_input.size(0), 0, device=raw_text_input.device)
             )
        
        audio_features = None
        if self.raw_audio_encoder is not None:
            audio_features = self.raw_audio_encoder(
                raw_audio_input if raw_audio_input is not None else torch.empty(raw_text_input.size(0), 0, device=raw_text_input.device)
            )

        # 2. Multi-Modal Fusion
        fused_features = self.multi_modal_fusion(text_features, image_features, audio_features) # (B, T_fused, embedding_dim)

        # 3. Determine Dynamic Resolution Level based on 'focus_strength'
        # Higher focus_strength -> higher resolve_level -> sharper attention / less dropout
        resolve_level = (focus_strength * 
                         (self.config.dynamic_resolution.max_res_scale - self.config.dynamic_resolution.min_res_scale) + 
                         self.config.dynamic_resolution.min_res_scale)
        resolve_level = torch.clamp(torch.tensor(resolve_level), 
                                    self.config.dynamic_resolution.min_res_scale, 
                                    self.config.dynamic_resolution.max_res_scale).item()

        # 4. Iterative Processing with Dynamic Resolution
        for block in self.dr_blocks:
            fused_features = block(fused_features, resolve_level=resolve_level) # Pass resolve_level

        fused_features = self.ln_f(fused_features) # Final LayerNorm

        # 5. Output Convergence Head (Generates logits)
        logits = self.output_convergence(fused_features) # (B, T_fused, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, raw_text_input, max_new_tokens, focus_strength=0.0, raw_image_input=None, raw_audio_input=None):
        # raw_text_input: (B, T_initial)
        self.eval() # Set model to evaluation mode
        
        # Start generation from text_input (for now)
        generated_sequence = raw_text_input 

        for _ in range(max_new_tokens):
            # Crop input if it exceeds block_size (Transformer models have fixed context window)
            idx_cond = generated_sequence if generated_sequence.size(1) <= self.config.model.block_size else generated_sequence[:, -self.config.model.block_size:]

            # Get logits for the next token (pass through model with current sequence)
            # Only raw_text_input is updated for generation; other modalities are kept fixed if provided initially
            # Pass image/audio inputs as None if not explicitly provided during generation call,
            # so the model's forward method can handle them gracefully if their encoders are enabled.
            logits = self(idx_cond, raw_image_input=raw_image_input, raw_audio_input=raw_audio_input, focus_strength=focus_strength)
            logits = logits[:, -1, :] # Focus on the last token's logits (B, vocab_size)

            # Sample the next token using the output convergence head's strategy
            idx_next = self.output_convergence.generate(logits, focus_strength=focus_strength) # (B, 1)

            # Append sampled token to the running sequence
            generated_sequence = torch.cat((generated_sequence, idx_next), dim=1)
        
        return generated_sequence
