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
        self.raw_text_encoder = UniversalRawTextEncoder(config)
        
        self.raw_image_encoder = None
        if hasattr(config.raw_encoder, 'image') and getattr(config.raw_encoder.image, 'enabled', False):
             self.raw_image_encoder = UniversalRawImageEncoder(config)
        
        self.raw_audio_encoder = None
        if hasattr(config.raw_encoder, 'audio') and getattr(config.raw_encoder.audio, 'enabled', False):
             self.raw_audio_encoder = UniversalRawAudioEncoder(config)

        # 2. Multi-Modal Fusion Module
        self.multi_modal_fusion = MultiModalFusionModule(config)
        
        # 3. Iterative Processing Blocks (Dynamic Resolution Attention)
        self.dr_blocks = nn.ModuleList([DRBlock(config) for _ in range(config.model.num_layers)])
        self.ln_f = nn.LayerNorm(config.model.embedding_dim) # Final LayerNorm

        # 4. Non-linear Output Convergence Head
        self.output_convergence = NonLinearOutputConvergence(config) # Consistent naming

        # Calculate and print model parameters AFTER SRF_net is added
        print(f"APERTURE-LLM Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")

    def _get_fused_features(self, raw_text_input, raw_image_input=None, raw_audio_input=None, focus_strength=0.0):
        """Helper to get fused features for a given input, used by both forward and generate."""
        text_features = self.raw_text_encoder(raw_text_input) if raw_text_input is not None else None
        
        image_features = None
        if self.raw_image_encoder is not None:
            # Pass torch.empty if raw_image_input is None, so dummy encoder can handle it.
            # Use raw_text_input.size(0) for batch size reference if other inputs are None.
            img_batch_size = raw_image_input.size(0) if raw_image_input is not None else raw_text_input.size(0)
            image_features = self.raw_image_encoder(
                raw_image_input if raw_image_input is not None else torch.empty(img_batch_size, 0, device=raw_text_input.device)
            )
        
        audio_features = None
        if self.raw_audio_encoder is not None:
            audio_batch_size = raw_audio_input.size(0) if raw_audio_input is not None else raw_text_input.size(0)
            audio_features = self.raw_audio_encoder(
                raw_audio_input if raw_audio_input is not None else torch.empty(audio_batch_size, 0, device=raw_text_input.device)
            )

        fused_features = self.multi_modal_fusion(text_features, image_features, audio_features)

        resolve_level = (focus_strength * 
                         (self.config.dynamic_resolution.max_res_scale - self.config.dynamic_resolution.min_res_scale) + 
                         self.config.dynamic_resolution.min_res_scale)
        resolve_level = torch.clamp(torch.tensor(resolve_level, device=fused_features.device), 
                                    self.config.dynamic_resolution.min_res_scale, 
                                    self.config.dynamic_resolution.max_res_scale).item()

        for block in self.dr_blocks:
            fused_features = block(fused_features, resolve_level=resolve_level)
        
        fused_features = self.ln_f(fused_features) # Final LayerNorm
        return fused_features

    def forward(self, raw_text_input, raw_image_input=None, raw_audio_input=None, focus_strength=0.0):
        """
        Main forward pass for the model, outputs logits.
        """
        fused_features = self._get_fused_features(raw_text_input, raw_image_input, raw_audio_input, focus_strength)
        logits = self.output_convergence(fused_features) # (B, T_fused, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, raw_text_input, max_new_tokens, focus_strength=0.0, raw_image_input=None, raw_audio_input=None):
        """
        Generates a sequence of tokens autoregressively.
        """
        self.eval() # Set model to evaluation mode
        
        # Start generation from text_input
        generated_sequence = raw_text_input 

        # Initial context features (image/audio are fixed for the entire generation)
        # The text input for initial context should be current raw_text_input
        # Subsequent text context (idx_cond) will be updated in the loop
        initial_image_input = raw_image_input
        initial_audio_input = raw_audio_input


        for _ in range(max_new_tokens):
            # Crop input if it exceeds block_size (Transformer models have fixed context window)
            idx_cond = generated_sequence if generated_sequence.size(1) <= self.config.model.block_size else generated_sequence[:, -self.config.model.block_size:]

            # Get the full fused features (x_context) for the current sequence
            # This is passed to the output_convergence.generate for context-aware sampling
            current_fused_features = self._get_fused_features(
                raw_text_input=idx_cond, 
                raw_image_input=initial_image_input, 
                raw_audio_input=initial_audio_input, 
                focus_strength=focus_strength
            )
            
            # Get raw logits from the output convergence head for the last token
            logits = self.output_convergence(current_fused_features) # (B, T_fused, vocab_size)
            logits_for_sampling = logits[:, -1, :] # Focus on the last token's logits (B, vocab_size)

            # Sample the next token using the output convergence head's strategy,
            # passing the full fused features as context (x_context)
            idx_next = self.output_convergence.generate(
                logits_for_sampling, 
                x_context=current_fused_features, 
                focus_strength=focus_strength
            ) # (B, 1)

            # Append sampled token to the running sequence
            generated_sequence = torch.cat((generated_sequence, idx_next), dim=1)
        
        return generated_sequence
