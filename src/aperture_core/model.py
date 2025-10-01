import torch
import torch.nn as nn
import torch.nn.functional as F

# CORRECTED: Use relative imports for modules within the same 'aperture_core' package
from .raw_encoders import UniversalRawTextEncoder, UniversalRawImageEncoder, UniversalRawAudioEncoder
from .multi_modal_fusion import MultiModalFusionModule
from .dynamic_resolution import DRBlock
from .output_convergence import NonLinearOutputConvergence


class ComputationAllocator(nn.Module):
    # ... (rest of the class code as before) ...
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
    # ... (rest of the class code as before) ...
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr_multiplier_net = nn.Sequential(
            nn.Linear(config.model.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output a multiplier in [0, 1]
        )
        self.base_online_lr = nn.Parameter(torch.tensor(0.001))

    def get_adaptive_online_lr(self, feedback_features):
        context_vector = feedback_features.mean(dim=1)  # (B, embedding_dim)
        lr_multiplier = self.lr_multiplier_net(context_vector)  # (B, 1)
        return self.base_online_lr * lr_multiplier  # (B, 1)


class APERTURE_LLM(nn.Module):
    # ... (rest of the class code as before) ...
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

        print(f"APERTURE-LLM Model initialized with "
              f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")

    def _encode_and_fuse(self, raw_text_input, raw_image_input=None, raw_audio_input=None):
        # ... (rest of the method code as before) ...
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
        return fused_features

    def _process_blocks_with_allocation(self, encoded_fused_features, focus_strength=0.0):
        # ... (rest of the method code as before) ...
        fused_features = encoded_fused_features

        resolve_level = (focus_strength *
                         (self.config.dynamic_resolution.max_res_scale -
                          self.config.dynamic_resolution.min_res_scale) +
                         self.config.dynamic_resolution.min_res_scale)
        resolve_level = torch.clamp(torch.tensor(resolve_level, device=fused_features.device),
                                    self.config.dynamic_resolution.min_res_scale,
                                    self.config.dynamic_resolution.max_res_scale).item()

        if self.training:
            for block in self.dr_blocks:
                fused_features = fused_features + block(fused_features, resolve_level=resolve_level)
        else:
            layer_weights = self.comp_allocator(fused_features)
            for i, block in enumerate(self.dr_blocks):
                mask = (layer_weights[:, i:i+1] > 0.5).float()

                block_output = block(fused_features, resolve_level=resolve_level)

                fused_features = fused_features + block_output * mask.unsqueeze(1)

        fused_features = self.ln_f(fused_features)
        return fused_features

    def forward(self, raw_text_input, raw_image_input=None, raw_audio_input=None, focus_strength=0.0):
        # ... (rest of the method code as before) ...
        encoded_fused_features = self._encode_and_fuse(raw_text_input, raw_image_input, raw_audio_input)
        processed_fused_features = self._process_blocks_with_allocation(encoded_fused_features, focus_strength)
        logits = self.output_convergence(processed_fused_features)
        return logits

    def generate(self, raw_text_input, max_new_tokens, focus_strength=0.0,
                 raw_image_input=None, raw_audio_input=None, targets=None,
                 adaptation_steps_limit=None):
        # ... (rest of the method code as before) ...
        original_training_state = self.training
        self.eval()

        generated_sequence = raw_text_input
        initial_image_input = raw_image_input
        initial_audio_input = raw_audio_input

        targets_sequence = targets

        for step in range(max_new_tokens):
            idx_cond = generated_sequence if generated_sequence.size(1) \
                <= self.config.model.block_size else generated_sequence[:, -self.config.model.block_size:]

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
                logits = self.output_convergence(current_fused_features)

            logits_for_sampling = logits[:, -1, :]

            idx_next = self.output_convergence.generate(
                logits_for_sampling,
                x_context=current_fused_features,
                focus_strength=focus_strength
            )

            if targets_sequence is not None and \
               (adaptation_steps_limit is None or step < adaptation_steps_limit):

                current_target_idx_in_sequence = generated_sequence.size(1)

                if current_target_idx_in_sequence < targets_sequence.size(1):
                    target_for_this_step = targets_sequence[:, current_target_idx_in_sequence]

                    adaptation_loss = F.cross_entropy(logits_for_sampling, target_for_this_step)

                    adaptation_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    online_lr_per_batch = self.meta_learner.get_adaptive_online_lr(current_fused_features)
                    adaptive_lr = online_lr_per_batch.mean().item()

                    for param in self.parameters():
                        if param.grad is not None and param.requires_grad:
                            param.data -= adaptive_lr * param.grad

                    self.zero_grad()
                else:
                    if step == 0:
                        print("Warning: Not enough targets provided for full online adaptation."
                              " Stopping online updates.")
                    targets_sequence = None
            generated_sequence = torch.cat((generated_sequence, idx_next), dim=1)

        self.train(original_training_state)
        return generated_sequence
