# src/aperture_core/output_convergence.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLinearOutputConvergence(nn.Module):
    """
    APERTURE-LLM's output head, designed for non-linear output convergence.
    'focus_strength' (0.0 to 1.0) influences sampling strategy,
    simulating 'Conceptual Avalanche Collapse'.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear_head = nn.Linear(config.model.embedding_dim, config.model.vocab_size)

    def forward(self, x):
        # x: (B, T, embedding_dim) - output from the main model body
        return self.linear_head(x) # (B, T, vocab_size) - raw logits


    def generate(self, logits, focus_strength=0.0):
        # logits: (B, vocab_size) - raw logits for the next token from the linear head
        
        # Non-linear mapping of focus_strength to sampling parameters
        # Higher focus_strength -> lower temperature, lower top_p (more decisive)
        # Lower focus_strength -> higher temperature, higher top_p (more exploratory)
        
        # Sigmoid-like non-linear mapping for temperature and top_p
        # This simulates a 'conceptual avalanche' - a sharp transition in determinism
        # around a certain focus_strength threshold.
        normalized_focus = torch.tensor(focus_strength).to(logits.device)
        sigmoid_factor = 1 / (1 + torch.exp(-10 * (normalized_focus - 0.5))) # Sharp transition around 0.5
        
        temperature = self.config.output_convergence.convergence_temp_max - \
                      (self.config.output_convergence.convergence_temp_max - self.config.output_convergence.convergence_temp_min) * sigmoid_factor
        
        top_p = self.config.output_convergence.convergence_top_p_max - \
                (self.config.output_convergence.convergence_top_p_max - self.config.output_convergence.convergence_top_p_min) * sigmoid_factor

        # Clamp to ensure valid range
        temperature = torch.clamp(temperature, self.config.output_convergence.convergence_temp_min, self.config.output_convergence.convergence_temp_max)
        top_p = torch.clamp(top_p, self.config.output_convergence.convergence_top_p_min, self.config.output_convergence.convergence_top_p_max)

        # Apply temperature to logits
        logits = logits / temperature

        # Apply Top-P sampling (nucleus sampling)
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Create a mask to remove tokens with cumulative probability above the threshold
            # Shift the indices to the right to keep at least one token above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Ensure that the first token (highest probability) is never removed
            sorted_indices_to_remove[..., 0] = False 
            
            # Scatter the -inf values back to the original logits tensor
            logits = logits.scatter(dim=-1, index=indices_to_remove, value=float('-inf'))

        # Sample from the adjusted probabilities
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        return idx_next
