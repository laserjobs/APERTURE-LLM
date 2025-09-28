import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MetaLearningModule(nn.Module):
    """
    Provides a context-aware scalar learning rate multiplier for online adaptation.
    Its own parameters (`base_online_lr`, `lr_multiplier_net`'s weights) are updated
    via standard backprop during main training if involved in the loss computation,
    or would be updated in an outer meta-training loop for true meta-learning.
    For this prototype, it enables adaptive online updates of the main model parameters.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr_multiplier_net = nn.Sequential(
            nn.Linear(config.model.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output a multiplier in [0, 1]
        )
        # A learnable base LR for online updates of the main model.
        # This parameter is part of the MetaLearningModule, so it will be included in self.parameters().
        self.base_online_lr = nn.Parameter(torch.tensor(0.001))

    def get_adaptive_online_lr(self, feedback_features):
        # feedback_features: (B, T_fused, embedding_dim) from current context
        context_vector = feedback_features.mean(dim=1)  # (B, embedding_dim)
        lr_multiplier = self.lr_multiplier_net(context_vector)  # (B, 1)
        # The adaptive online LR for each item in the batch
        return self.base_online_lr * lr_multiplier  # (B, 1)


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
        self.ln_f = nn.LayerNorm(config.model.embedding_dim)  # Final LayerNorm

        # 4. Computation Allocator for dynamic computation paths
        self.comp_allocator = ComputationAllocator(config)

        # 5. Non-linear Output Convergence Head (with Dynamic Sampling Rate)
        self.output_convergence = NonLinearOutputConvergence(config)

        # 6. Self-Improving Learning Algorithm (Online Adaptation)
        self.meta_learner = MetaLearningModule(config)

        # Calculate and print model parameters AFTER all modules are added
        print(f"APERTURE-LLM Model initialized with "
              f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")

    def _encode_and_fuse(self, raw_text_input, raw_image_input=None, raw_audio_input=None):
        """Helper to encode raw inputs and perform multi-modal fusion."""
        # Determine batch size from available input (raw_text_input is assumed primary for this prototype)
        batch_size_ref = (raw_text_input.size(0)
                          if raw_text_input is not None and raw_text_input.numel() > 0 else
                          (raw_image_input.size(0)
                           if raw_image_input is not None and raw_image_input.numel() > 0 else
                           (raw_audio_input.size(0)
                            if raw_audio_input is not None and raw_audio_input.numel() > 0 else 1)))

        device = (raw_text_input.device
                  if raw_text_input is not None and raw_text_input.numel() > 0 else
                  (raw_image_input.device
                   if raw_image_input is not None and raw_image_input.numel() > 0 else
                   (raw_audio_input.device
                    if raw_audio_input is not None and raw_audio_input.numel() > 0 else torch.device('cpu'))))

        text_features = (self.raw_text_encoder(raw_text_input)
                         if raw_text_input is not None and raw_text_input.numel() > 0 else None)

        image_features = None
        if self.raw_image_encoder is not None:
            image_features = self.raw_image_encoder(
                raw_image_input if raw_image_input is not None and raw_image_input.numel() > 0
                else torch.empty(batch_size_ref, 0, device=device)
            )

        audio_features = None
        if self.raw_audio_encoder is not None:
            audio_features = self.raw_audio_encoder(
                raw_audio_input if raw_audio_input is not None and raw_audio_input.numel() > 0
                else torch.empty(batch_size_ref, 0, device=device)
            )

        fused_features = self.multi_modal_fusion(text_features, image_features, audio_features)
        return fused_features  # (B, T_fused, embedding_dim)

    def _process_blocks_with_allocation(self, encoded_fused_features, focus_strength=0.0):
        """
        Helper to process features through DRBlocks, applying dynamic computation allocation
        during inference.
        """
        fused_features = encoded_fused_features  # Start with fused features

        resolve_level = (focus_strength *
                         (self.config.dynamic_resolution.max_res_scale -
                          self.config.dynamic_resolution.min_res_scale) +
                         self.config.dynamic_resolution.min_res_scale)
        resolve_level = torch.clamp(torch.tensor(resolve_level, device=fused_features.device),
                                    self.config.dynamic_resolution.min_res_scale,
                                    self.config.dynamic_resolution.max_res_scale).item()

        if self.training:  # During training, always apply all blocks for gradient flow
            for block in self.dr_blocks:
                fused_features = fused_features + block(fused_features, resolve_level=resolve_level)
        else:  # During inference/evaluation, apply blocks based on ComputationAllocator
            layer_weights = self.comp_allocator(fused_features)  # (B, num_layers)
            for i, block in enumerate(self.dr_blocks):
                # Create a binary mask for each block in the batch.
                # Threshold of 0.5 for Sigmoid output; if weight > 0.5, block is "active".
                mask = (layer_weights[:, i:i+1] > 0.5).float()  # (B, 1)

                block_output = block(fused_features, resolve_level=resolve_level)

                # Selectively add block output based on mask.
                fused_features = fused_features + block_output * mask.unsqueeze(1)  # Apply block if mask is 1

        fused_features = self.ln_f(fused_features)  # Final LayerNorm
        return fused_features


    def forward(self, raw_text_input, raw_image_input=None, raw_audio_input=None, focus_strength=0.0):
        """
        Main forward pass for the model, outputs logits.
        """
        encoded_fused_features = self._encode_and_fuse(raw_text_input, raw_image_input, raw_audio_input)
        processed_fused_features = self._process_blocks_with_allocation(encoded_fused_features, focus_strength)
        logits = self.output_convergence(processed_fused_features)  # (B, T_fused, vocab_size)
        return logits


    def generate(self, raw_text_input, max_new_tokens, focus_strength=0.0,
                 raw_image_input=None, raw_audio_input=None, targets=None,
                 adaptation_steps_limit=None):  # Added adaptation_steps_limit
        """
        Generates a sequence of tokens autoregressively, with optional online adaptation.
        If `targets` are provided, the model performs per-step gradient updates to its parameters.
        `targets` should be a (B, T_total_targets) tensor of desired continuation tokens.
        """
        # Store original training state and set to eval mode ( ComputationAllocator uses this)
        original_training_state = self.training
        self.eval()

        generated_sequence = raw_text_input
        initial_image_input = raw_image_input
        initial_audio_input = raw_audio_input

        targets_sequence = targets  # (B, T_total_targets)

        for step in range(max_new_tokens):
            # 1. Prepare current input for the model
            idx_cond = generated_sequence if generated_sequence.size(1) \
                <= self.config.model.block_size else generated_sequence[:, -self.config.model.block_size:]

            # 2. Forward pass for current step (gradients conditionally enabled)
            with torch.enable_grad() if targets_sequence is not None else torch.no_grad():
                encoded_fused_features_for_step = self._encode_and_fuse(
                    raw_text_input=idx_cond,
                    raw_image_input=initial_image_input,
                    raw_audio_input=initial_audio_input
                )
                current_fused_features = self._process_blocks_with_allocation(
                    encoded_fused_features_for_step,
                    focus_strength=focus_strength
                )
                logits = self.output_convergence(current_fused_features)  # (B, T_fused, vocab_size)

            logits_for_sampling = logits[:, -1, :]  # (B, vocab_size)

            # 3. Sample the next token
            idx_next = self.output_convergence.generate(
                logits_for_sampling,
                x_context=current_fused_features,
                focus_strength=focus_strength
            )  # (B, 1)

            # --- 4. Online Adaptation Step (if targets are provided and within limits) ---
            if targets_sequence is not None and \
               (adaptation_steps_limit is None or step < adaptation_steps_limit):

                # Determine the target token for this specific generation step
                current_target_idx_in_sequence = generated_sequence.size(1)

                # Ensure target_token_idx is within bounds of targets_sequence
                if current_target_idx_in_sequence < targets_sequence.size(1):
                    target_for_this_step = targets_sequence[:, current_target_idx_in_sequence]  # (B,)

                    # Compute adaptation loss
                    adaptation_loss = F.cross_entropy(logits_for_sampling, target_for_this_step)

                    # Compute gradients for adaptation
                    adaptation_loss.backward()

                    # --- MITIGATION: Apply gradient clipping ---
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    # Get adaptive online learning rate from MetaLearningModule
                    online_lr_per_batch = self.meta_learner.get_adaptive_online_lr(current_fused_features)
                    adaptive_lr = online_lr_per_batch.mean().item()  # Scalar LR for entire batch for all params

                    for param in self.parameters():
                        if param.grad is not None and param.requires_grad:
                            param.data -= adaptive_lr * param.grad

                    # Clear gradients for the next step before next forward pass
                    self.zero_grad()
                else:
                    # Ran out of targets sequence for adaptation
                    if step == 0:
                        print("Warning: Not enough targets provided for full online adaptation."
                              " Stopping online updates.")
                    targets_sequence = None  # Stop further adaptation
            # --- End Online Adaptation Step ---

            # 5. Append sampled token to the running sequence
            generated_sequence = torch.cat((generated_sequence, idx_next), dim=1)

        # Restore original training state
        self.train(original_training_state)
        return generated_sequence
