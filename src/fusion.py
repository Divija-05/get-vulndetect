import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient cross-attention mechanism."""
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 64
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores in chunks to save memory
        outputs = []
        for i in range(0, q.size(2), self.chunk_size):
            chunk_q = q[:, :, i:i+self.chunk_size]
            
            # Scaled dot-product attention
            scores = torch.matmul(chunk_q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            chunk_output = torch.matmul(attn, v)
            outputs.append(chunk_output)
        
        # Combine chunks
        output = torch.cat(outputs, dim=2)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.o_proj(output)

class FusionLayer(nn.Module):
    def __init__(
        self,
        gnn_dim: int,
        transformer_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.transformer_dim = transformer_dim
        self.hidden_dim = hidden_dim
        self.use_gating = use_gating
        
        # Project inputs to same dimension
        self.gnn_projection = nn.Linear(gnn_dim, hidden_dim)
        self.transformer_projection = nn.Linear(transformer_dim, hidden_dim)
        
        # Cross-attention mechanisms
        self.gnn_to_transformer_attention = MemoryEfficientAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.transformer_to_gnn_attention = MemoryEfficientAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Gating mechanism
        if use_gating:
            self.gnn_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.transformer_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Vulnerability-specific attention
        self.vuln_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.vuln_patterns = nn.Parameter(torch.randn(5, hidden_dim))  # 5 vulnerability types
        
        # Memory-efficient output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(
        self,
        gnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
        gnn_mask: Optional[torch.Tensor] = None,
        transformer_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Project features to same dimension
        gnn_proj = self.gnn_projection(gnn_features)
        transformer_proj = self.transformer_projection(transformer_features)
        
        # Cross-attention
        gnn_attended = self.gnn_to_transformer_attention(
            gnn_proj, transformer_proj, transformer_proj, transformer_mask
        )
        transformer_attended = self.transformer_to_gnn_attention(
            transformer_proj, gnn_proj, gnn_proj, gnn_mask
        )
        
        # Apply gating if enabled
        if self.use_gating:
            gnn_gate = self.gnn_gate(torch.cat([gnn_proj, transformer_attended], dim=-1))
            transformer_gate = self.transformer_gate(torch.cat([transformer_proj, gnn_attended], dim=-1))
            
            gnn_features = gnn_gate * gnn_proj + (1 - gnn_gate) * transformer_attended
            transformer_features = transformer_gate * transformer_proj + (1 - transformer_gate) * gnn_attended
        else:
            gnn_features = gnn_proj + transformer_attended
            transformer_features = transformer_proj + gnn_attended
        
        # Combine features
        combined_features = (gnn_features + transformer_features) / 2
        combined_features = self.layer_norm(combined_features)
        
        # Apply vulnerability-specific attention
        vuln_patterns = self.vuln_patterns.unsqueeze(0).expand(
            combined_features.size(0), -1, -1
        )
        vuln_attended_features, _ = self.vuln_attention(
            combined_features, vuln_patterns, vuln_patterns
        )
        
        # Final projection with memory-efficient implementation
        output = self.output_projection(combined_features + vuln_attended_features)
        output = self.dropout(output)
        
        return output

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for interpretability."""
        return {
            'gnn_to_transformer': self.gnn_to_transformer_attention.o_proj.weight.detach(),
            'transformer_to_gnn': self.transformer_to_gnn_attention.o_proj.weight.detach(),
            'vulnerability_patterns': self.vuln_patterns.detach()
        }