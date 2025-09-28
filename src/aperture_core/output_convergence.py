# src/aperture_core/output_convergence.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLinearOutputConvergence(nn.Module):
    # ... (rest of the class remains the same)

    def generate(self, logits, x_context, focus_strength=0.5):
        """
        Generates the next token using context-aware top-p sampling.
        Dynamically adjusts temperature and top_p based on context and focus_strength.
        Args:
            logits: (B, vocab_size) logits for the *last* token
            x_context: (B, T_fused, embedding_dim) context features from DRBlocks/final LayerNorm for the current sequence
            focus_strength: Scalar in [0, 1] to modulate sampling behavior (user control)
        Returns:
            idx_next: (B, 1) sampled token indices
        """
        # Ensure focus_strength is a tensor for device compatibility and operations
        focus_strength_tensor = torch.tensor(focus_strength, device=logits.device, dtype=logits.dtype)

        # Compute context-aware sampling parameters
        # Mean-pool over the sequence length to get a single context vector per batch item
        context_features = x_context.mean(dim=1)  # (B, embedding_dim)
        srf_params = self.srf_net(context_features)  # (B, 2)
        temp_scale, top_p_scale = srf_params[:, 0], srf_params[:, 1]  # (B,), (B,)

        # Retrieve min/max bounds from config
        temp_min = self.config.output_convergence.convergence_temp_min
        temp_max = self.config.output_convergence.convergence_temp_max
        top_p_min = self.config.output_convergence.convergence_top_p_min
        top_p_max = self.config.output_convergence.convergence_top_p_max

        # --- MODIFIED: Dynamic temperature calculation ---
        # High focus_strength -> lower temperature (more deterministic)
        # Low focus_strength -> higher temperature (more exploratory)
        # Use (1.0 - focus_strength_tensor) to get an 'exploration_strength'
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
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, vocab_size)
        
        # Create a boolean mask for tokens to remove: cumulative prob > top_p
        # top_p needs to be unsqueezed for broadcasting
        sorted_indices_to_remove = cumulative_probs > top_p.unsqueeze(-1)
        
        # Shift the indices to the right to keep at least one token (the highest prob token)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False # Ensure the highest probability token is always kept
        
        # Create a mask to apply to the original `logits` tensor
        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        logits = logits.masked_fill(mask, float('-inf'))

        # Sample from the adjusted probabilities
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        
        return idx_next
