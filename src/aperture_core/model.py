# src/aperture_core/model.py
import torch
import torch.nn as nn
from src.aperture_core.raw_encoders import UniversalRawTextEncoder, UniversalRawImageEncoder, UniversalRawAudioEncoder
from src.aperture_core.multi_modal_fusion import MultiModalFusionModule
from src.aperture_core.dynamic_resolution import DRBlock
from src.aperture_core.output_convergence import NonLinearOutputConvergence

class ComputationAllocator(nn.Module):
    """
    Predicts per-layer activation weights based on fused input features.
    Inspired by SRF's dynamic Planck Filter/Avalanche Collapse,
    it allows for adaptive computation paths during inference.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.complexity_scorer = nn.Sequential(
            nn.Linear(config.model.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.model.num_layers),
            nn.Sigmoid()  # Output per-layer activation probability [0, 1]
        )

    def forward(self, fused_features):
        # fused_features: (B, T_fused, embedding_dim)
        # Mean-pool over sequence length to get a single context vector per batch item
        context_vector = fused_features.mean(dim=1)  # (B, embedding_dim)
        layer_weights = self.complexity_scorer(context_vector)  # (B, num_layers)
        return layer_weights

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

        # 4. Computation Allocator for dynamic computation paths
        self.comp_allocator = ComputationAllocator(config)

        # 5. Non-linear Output Convergence Head
        self.output_convergence = NonLinearOutputConvergence(config)

        # Calculate and print model parameters AFTER all modules are added
        print(f"APERTURE-LLM Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")

    def _encode_and_fuse(self, raw_text_input, raw_image_input=None, raw_audio_input=None):
        """Helper to encode raw inputs and perform multi-modal fusion."""
        # Determine batch size from available input (raw_text_input is assumed primary for this prototype)
        batch_size_ref = raw_text_input.size(0) if raw_text_input is not None else 1
        device = raw_text_input.device if raw_text_input is not None else (raw_image_input.device if raw_image_input is not None else (raw_audio_input.device if raw_audio_input is not None else 'cpu'))

        text_features = self.raw_text_encoder(raw_text_input) if raw_text_input is not None else None
        
        image_features = None
        if self.raw_image_encoder is not None:
            image_features = self.raw_image_encoder(
                raw_image_input if raw_image_input is not None else torch.empty(batch_size_ref, 0, device=device)
            )
        
        audio_features = None
        if self.raw_audio_encoder is not None:
            audio_features = self.raw_audio_encoder(
                raw_audio_input if raw_audio_input is not None else torch.empty(batch_size_ref, 0, device=device)
            )

        fused_features = self.multi_modal_fusion(text_features, image_features, audio_features)
        return fused_features # (B, T_fused, embedding_dim)

    def _process_blocks_with_allocation(self, encoded_fused_features, focus_strength=0.0):
        """
        Helper to process features through DRBlocks, applying dynamic computation allocation
        during inference.
        """
        fused_features = encoded_fused_features # Start with fused features

        resolve_level = (focus_strength * 
                         (self.config.dynamic_resolution.max_res_scale - self.config.dynamic_resolution.min_res_scale) + 
                         self.config.dynamic_resolution.min_res_scale)
        resolve_level = torch.clamp(torch.tensor(resolve_level, device=fused_features.device), 
                                    self.config.dynamic_resolution.min_res_scale, 
                                    self.config.dynamic_resolution.max_res_scale).item()

        if self.training: # During training, always apply all blocks for gradient flow
            for block in self.dr_blocks:
                fused_features = block(fused_features, resolve_level=resolve_level)
        else: # During inference/evaluation, apply blocks based on ComputationAllocator
            layer_weights = self.comp_allocator(fused_features) # (B, num_layers)
            for i, block in enumerate(self.dr_blocks):
                # Create a binary mask for each block in the batch.
                # Threshold of 0.5 for Sigmoid output; if weight > 0.5, block is "active".
                mask = (layer_weights[:, i:i+1] > 0.5).float() # (B, 1)
                
                block_output = block(fused_features, resolve_level=resolve_level)
                
                # Selectively add block output based on mask.
                # fused_features = fused_features + (mask.unsqueeze(1) * block_output)
                # To prevent gradient issues with block_output when mask is 0
                fused_features = fused_features + block_output * mask.unsqueeze(1) # Apply block if mask is 1

        fused_features = self.ln_f(fused_features) # Final LayerNorm
        return fused_features


    def forward(self, raw_text_input, raw_image_input=None, raw_audio_input=None, focus_strength=0.0):
        """
        Main forward pass for the model, outputs logits.
        """
        encoded_fused_features = self._encode_and_fuse(raw_text_input, raw_image_input, raw_audio_input)
        processed_fused_features = self._process_blocks_with_allocation(encoded_fused_features, focus_strength)
        logits = self.output_convergence(processed_fused_features) # (B, T_fused, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, raw_text_input, max_new_tokens, focus_strength=0.0, raw_image_input=None, raw_audio_input=None):
        """
        Generates a sequence of tokens autoregressively.
        """
        self.eval() # Set model to evaluation mode (activates computation allocation)
        
        # Start generation from text_input
        generated_sequence = raw_text_input 

        # Initial context features (image/audio are fixed for the entire generation)
        initial_image_input = raw_image_input
        initial_audio_input = raw_audio_input

        for _ in range(max_new_tokens):
            # Crop input if it exceeds block_size (Transformer models have fixed context window)
            idx_cond = generated_sequence if generated_sequence.size(1) <= self.config.model.block_size else generated_sequence[:, -self.config.model.block_size:]

            # Get the full fused features (x_context) for the current sequence
            # This is passed to the output_convergence.generate for context-aware sampling
            encoded_fused_features_for_step = self._encode_and_fuse(
                raw_text_input=idx_cond, 
                raw_image_input=initial_image_input, 
                raw_audio_input=initial_audio_input
            )
            current_fused_features = self._process_blocks_with_allocation(
                encoded_fused_features_for_step, 
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
