# src/aperture_core/token_bridge.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# Removed 'warnings' as it was imported but unused (F401)


class DummyExternalTokenizer:
    """
    A placeholder for an external, traditional token-based tokenizer (e.g., BPE, WordPiece).
    It operates on a pre-defined vocabulary of 'words' or 'subwords'.
    """

    def __init__(self, vocab_list=None):
        if vocab_list is None:
            # A very simple, fixed vocabulary for demonstration
            self.vocab = ["<unk>", "the", "a", "of", "and", "in", "is", "it", "that", "for",
                          "this", "to", "was", "with", "as", "on", "from", "by", "at", "not",
                          "he", "she", "they", "we", "you", "I", "an", "be", "have", "had",
                          "but", "or", "which", "are", "would", "could", "should", "will",
                          "can", "do", "said", "say", "went", "go", "come", "see", "make", "get",
                          "up", "down", "out", "in", "on", "off", "about", "into", "over", "under",
                          "then", "now", "when", "where", "why", "how", "what", "who", "whom",
                          ".", ",", "!", "?", "'", "-", " "]
        else:
            self.vocab = vocab_list

        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.unk_token_id = self.stoi.get("<unk>", 0)

    def encode(self, text: str) -> list[int]:
        """
        Mimics tokenization. Splits by common delimiters and falls back to <unk>.
        This is a VERY simplified example for demonstration.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be string, but got {type(text)}")

        # Simple splitting by spaces and common punctuation.
        # This is not robust like BPE but serves the purpose of mapping string to token IDs.
        processed_text = text.replace('.', ' . ').replace(',', ' , ').replace('!', ' ! ') \
                             .replace('?', ' ? ').replace("'", " ' ").replace('-', ' - ')
        words_and_punctuation = processed_text.lower().split()

        token_ids = []
        for item in words_and_punctuation:
            if item in self.stoi:
                token_ids.append(self.stoi[item])
            else:
                token_ids.append(self.unk_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Converts token IDs back to a string.
        """
        if not isinstance(token_ids, (list, torch.Tensor)):
            raise TypeError(f"Expected token_ids to be list or Tensor, but got {type(token_ids)}")

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        decoded_parts = []
        for i in token_ids:
            token_str = self.itos.get(i, self.itos[self.unk_token_id])
            if token_str == " ":  # Handle spaces explicitly
                decoded_parts.append(" ")
            # Handle punctuation with no preceding space
            elif token_str in [".", ",", "!", "?", "'", "-"]:
                if decoded_parts and decoded_parts[-1] == " ":  # Remove space if punctuation follows
                    decoded_parts.pop()
                decoded_parts.append(token_str)
            else:
                # Add a space before words unless it's the first word or
                # the previous character was punctuation that doesn't need a space.
                if decoded_parts and decoded_parts[-1] not in [" ", ".", ",", "!", "?", "'", "-"]:
                    decoded_parts.append(" ")
                decoded_parts.append(token_str)

        return "".join(decoded_parts).strip()


class TokenToRawCharAdapter(nn.Module):
    """
    Adapter to convert token IDs from a traditional tokenized AI into
    raw character indices suitable for APERTURE-LLM's UniversalRawTextEncoder.

    This effectively 'detokenizes' the input into characters.
    """

    def __init__(self, external_tokenizer, aperture_char_tokenizer):
        super().__init__()
        self.external_tokenizer = external_tokenizer
        self.aperture_char_tokenizer = aperture_char_tokenizer

        # Ensure that the external tokenizer has an .decode method
        if not hasattr(self.external_tokenizer, 'decode'):
            raise ValueError("Provided external_tokenizer must have a 'decode' method.")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of token_ids into a batch of raw character indices.

        Args:
            token_ids (torch.Tensor): A tensor of shape (B, T_tokens) containing token IDs
                                      from the external tokenized AI.

        Returns:
            torch.Tensor: A tensor of shape (B, T_chars) containing raw character indices
                          suitable for APERTURE-LLM.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)  # Add batch dimension if missing

        batch_raw_char_indices = []
        for i in range(token_ids.size(0)):
            # 1. Decode token IDs back to a human-readable string
            # This is the crucial step: tokens -> raw string
            decoded_text = self.external_tokenizer.decode(token_ids[i].tolist())

            # 2. Encode the raw string into APERTURE-LLM's character indices
            char_indices = self.aperture_char_tokenizer.encode(decoded_text)
            batch_raw_char_indices.append(torch.tensor(char_indices, dtype=torch.long,
                                                       device=token_ids.device))

        # Pad sequences to the maximum length in the batch
        max_len = max(len(seq) for seq in batch_raw_char_indices)
        # Assuming padding with the character for 'null' or space (index 0 for CharTokenizer)
        padded_batch = torch.stack([
            F.pad(seq, (0, max_len - len(seq)), 'constant', 0) for seq in batch_raw_char_indices
        ])

        return padded_batch


class RawCharToTokenAdapter(nn.Module):
    """
    Adapter to convert raw character indices from APERTURE-LLM's output
    into token IDs suitable for a traditional tokenized AI.

    This effectively 'tokenizes' APERTURE-LLM's raw output.
    """

    def __init__(self, external_tokenizer, aperture_char_tokenizer):
        super().__init__()
        self.external_tokenizer = external_tokenizer
        self.aperture_char_tokenizer = aperture_char_tokenizer

        # Ensure that the external tokenizer has an .encode method
        if not hasattr(self.external_tokenizer, 'encode'):
            raise ValueError("Provided external_tokenizer must have an 'encode' method.")

    def forward(self, char_indices: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of raw character indices into a batch of token IDs.

        Args:
            char_indices (torch.Tensor): A tensor of shape (B, T_chars) containing
                                         raw character indices from APERTURE-LLM's output.

        Returns:
            torch.Tensor: A tensor of shape (B, T_tokens) containing token IDs
                          suitable for the external tokenized AI.
        """
        if char_indices.ndim == 1:
            char_indices = char_indices.unsqueeze(0)  # Add batch dimension if missing

        batch_token_ids = []
        for i in range(char_indices.size(0)):
            # 1. Decode APERTURE-LLM's character indices into a raw string
            decoded_text = self.aperture_char_tokenizer.decode(char_indices[i].tolist())

            # 2. Encode the raw string into the external tokenizer's token IDs
            token_ids = self.external_tokenizer.encode(decoded_text)
            batch_token_ids.append(torch.tensor(token_ids, dtype=torch.long,
                                                device=char_indices.device))

        # Pad sequences to the maximum length in the batch
        max_len = max(len(seq) for seq in batch_token_ids)
        # Assuming padding with the external tokenizer's UNK token ID
        padded_batch = torch.stack([
            F.pad(seq, (0, max_len - len(seq)), 'constant', self.external_tokenizer.unk_token_id)
            for seq in batch_token_ids
        ])

        return padded_batch
