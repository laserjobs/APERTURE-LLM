import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class NonLinearOutputConvergence(nn.Module):
    """
    Non-linear output convergence head for APERTURE-LLM.
    Maps fused features to vocabulary logits and handles generation with dynamic sampling.
    Uses a context-aware neural network (SRF-inspired) to adjust temperature and top_p dynamically.
    """
    def __init__(self, config):
        super().__init__()
        # Debugging print statement:
        print(f"DEBUG: Initializing NonLinearOutputConvergence from: "
              f"{os.path.abspath(__file__)}")
        self.config = config
        self.linear_head = nn.Linear(config.model.embedding_dim, config.model.vocab_size)

        # SRF-inspired network to predict sampling parameters from context
        # Input: mean-pooled fused_features (embedding_dim)
        # Output: 2 scaling factors (for temperature and top_p) in [0, 1]
        self.srf_net = nn.Sequential(
            nn.Linear(config.model.embedding_dim, 64),  # Reduce dimensionality
            nn.ReLU(),
            nn.Linear(64, 2),   # Output 2 values: temp_scale, top_p_scale
            nn.Sigmoid()        # Normalize to [0, 1]
        )

    def forward(self, x):
        """
        Maps fused features to logits. This is used during training and as the first step in generation.
        Args:
            x: (B, T_fused, embedding_dim) tensor from DRBlocks/final LayerNorm
        Returns:
            logits: (B, T_fused, vocab_size)
        """
        return self.linear_head(x)

    def generate(self, logits, x_context, focus_strength=0.5):
        """
        Generates the next token using context-aware top-p sampling.
        Dynamically adjusts temperature and top_p based on context and focus_strength.
        Args:
            logits: (B, vocab_size) logits for the *last* token
            x_context: (B, T_fused, embedding_dim) context features from DRBlocks/final LayerNorm
                       for the current sequence
            focus_strength: Scalar in [0, 1] to modulate sampling behavior (user control)
        Returns:
            idx_next: (B, 1) sampled token indices
        """
        # Ensure focus_strength is a tensor for device compatibility and operations
        focus_strength_tensor = torch.tensor(focus_strength, device=logits.device, dtype=logits.dtype)

        # Compute context-aware sampling parameters (srf_net is currently unused in this simplified demo)
        # context_features = x_context.mean(dim=1)
        # srf_params = self.srf_net(context_features)
        # temp_scale, top_p_scale = srf_params[:, 0], srf_params[:, 1]

        # Retrieve min/max bounds from config
        temp_min = self.config.output_convergence.convergence_temp_min
        temp_max = self.config.output_convergence.convergence_temp_max
        top_p_min = self.config.output_convergence.convergence_top_p_min
        top_p_max = self.config.output_convergence.convergence_top_p_max

        # --- MODIFIED: Dynamic temperature calculation ---
        # High focus_strength -> lower temperature (more deterministic)
        # Low focus_strength -> higher temperature (more exploratory)
        # Use (1.0 - focus_strength_tensor) for 'exploration_strength' factor
        temperature = temp_min + (temp_max - temp_min) * (1.0 - focus_strength_tensor)
        temperature = torch.clamp(temperature, temp_min, temp_max)  # (B,) - ensure it's a batch of temps

        # --- MODIFIED: Dynamic top_p calculation ---
        # High focus_strength -> lower top_p (more deterministic)
        # Low focus_strength -> higher top_p (more exploratory)
        top_p = top_p_min + (top_p_max - top_p_min) * (1.0 - focus_strength_tensor)
        top_p = torch.clamp(top_p, top_p_min, top_p_max)  # (B,) - ensure it's a batch of top_p values

        # Apply temperature to logits (needs to be broadcasted per batch item)
        logits = logits / temperature.unsqueeze(-1)  # (B, vocab_size)

        # Top-p sampling (nucleus sampling)
        # Probabilities for sorting and cumulative sum
        probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create a boolean mask for tokens to remove: cumulative prob > top_p
        # top_p needs to be unsqueezed for broadcasting
        sorted_indices_to_remove = cumulative_probs > top_p.unsqueeze(-1)

        # Shift the indices to the right to keep at least one token (the highest prob token)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False  # Ensure the highest probability token is always kept

        # Create a mask to apply to the original `logits` tensor
        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        logits = logits.masked_fill(mask, float('-inf'))

        # Sample from the adjusted probabilities
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        return idx_next
