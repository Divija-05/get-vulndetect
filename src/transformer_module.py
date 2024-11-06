# src/transformer_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import Dict, Optional, Tuple

class SmartContractTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 1,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        
        # Initialize CodeBERT tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
        
        # Freeze most of the base model layers to prevent overfitting
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Only fine-tune the last 2 layers
        for param in self.base_model.encoder.layer[-2:].parameters():
            param.requires_grad = True
            
        self.max_length = max_length
        self.output_dim = hidden_dim
        
        # Project CodeBERT's hidden size (768) to our desired hidden dimension
        self.projection = nn.Linear(768, hidden_dim)
        
        # Additional transformer layers for Solidity-specific learning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling layer
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Vulnerability pattern attention
        self.vuln_patterns = nn.Parameter(torch.randn(5, hidden_dim))  # 5 vulnerability types
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def preprocess_code(self, code: str) -> str:
        """Preprocess Solidity code for tokenization."""
        # Remove comments
        code = self._remove_comments(code)
        # Normalize whitespace
        code = ' '.join(code.split())
        # Add special tokens for Solidity keywords
        code = self._highlight_solidity_keywords(code)
        return code
        
    def _remove_comments(self, code: str) -> str:
        """Remove single-line and multi-line comments from code."""
        # Remove single-line comments
        code = re.sub(r'//.*?\n', '\n', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code
        
    def _highlight_solidity_keywords(self, code: str) -> str:
        """Add special tokens around important Solidity keywords."""
        keywords = [
            'payable', 'view', 'pure', 'external', 'public', 'private',
            'internal', 'mapping', 'address', 'require', 'assert', 'revert'
        ]
        for keyword in keywords:
            code = re.sub(
                f'\\b{keyword}\\b',
                f'<SOL_{keyword.upper()}> {keyword} </SOL_{keyword.upper()}>',
                code
            )
        return code
        
    def tokenize(self, code: str) -> Dict[str, torch.Tensor]:
        """Tokenize preprocessed code."""
        # Preprocess code
        code = self.preprocess_code(code)
        
        # Tokenize with CodeBERT tokenizer
        tokens = self.tokenizer(
            code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return tokens
        
    def forward(
        self,
        code: str,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer.
        
        Args:
            code: Raw Solidity code string
            attention_mask: Optional attention mask
            
        Returns:
            tuple: (sequence_output, pooled_output)
        """
        # Tokenize input
        tokens = self.tokenize(code)
        
        # Get CodeBERT embeddings
        with torch.no_grad():
            base_outputs = self.base_model(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
                return_dict=True
            )
        
        # Project to our hidden dimension
        sequence_output = self.projection(base_outputs.last_hidden_state)
        
        # Pass through additional transformer layers
        if attention_mask is None:
            attention_mask = tokens.attention_mask
            
        # Create attention mask for transformer
        attn_mask = attention_mask.eq(0)
        
        sequence_output = self.transformer_encoder(
            sequence_output,
            src_key_padding_mask=attn_mask
        )
        
        # Apply vulnerability pattern attention
        pattern_attn_output, _ = self.pattern_attention(
            sequence_output,
            self.vuln_patterns.unsqueeze(0).expand(sequence_output.size(0), -1, -1),
            self.vuln_patterns.unsqueeze(0).expand(sequence_output.size(0), -1, -1)
        )
        
        # Combine with original output
        sequence_output = sequence_output + pattern_attn_output
        
        # Apply attention pooling
        attention_weights = self.attention_pool(sequence_output).squeeze(-1)
        attention_weights = F.softmax(attention_weights.masked_fill(attn_mask, -float('inf')), dim=1)
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1),
            sequence_output
        ).squeeze(1)
        
        return sequence_output, pooled_output
    
    def get_vulnerability_embeddings(self) -> torch.Tensor:
        """Get learned vulnerability pattern embeddings."""
        return self.vuln_patterns.detach()