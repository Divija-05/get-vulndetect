# src/fusion_layer.py
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class EnhancedFusionLayer(nn.Module):
    """
    Enhanced fusion layer that combines GNN and Transformer outputs 
    with vulnerability-specific attention and hierarchical feature integration.
    """
    def __init__(
            
        self,
        gnn_dim: int,
        transformer_dim: int,
        output_dim: int,
        num_vulnerability_types: int = 5,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_hierarchical: bool = True
    ):
        super().__init__()
        
        # Dimensions
        self.gnn_dim = gnn_dim
        self.transformer_dim = transformer_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_hierarchical = use_hierarchical
        
        # Project inputs to common dimension (output_dim)
        self.gnn_projection = nn.Linear(gnn_dim, output_dim)
        self.transformer_projection = nn.Linear(transformer_dim, output_dim)
        
        # Hierarchical integration layers
        if use_hierarchical:
            self.hierarchical_layers = nn.ModuleList([
                nn.Linear(output_dim * 2, output_dim),
                nn.Linear(output_dim, output_dim),
                nn.Linear(output_dim, output_dim)
            ])
            
            self.hierarchical_attention = nn.Parameter(
                torch.ones(len(self.hierarchical_layers)) / len(self.hierarchical_layers)
            )
        
        # Cross-Modal Attention
        self.gnn_to_transformer_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_to_gnn_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Vulnerability-specific attention
        self.vulnerability_patterns = nn.Parameter(
            torch.randn(num_vulnerability_types, output_dim)
        )
        self.vuln_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanisms
        self.gnn_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
        self.transformer_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
        
        # Output processing
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(
        self,
        gnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
        gnn_mask: Optional[torch.Tensor] = None,
        transformer_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Project features to common dimension
        gnn_proj = self.gnn_projection(gnn_features)
        transformer_proj = self.transformer_projection(transformer_features)
        
        attention_weights = {}
        
        # Cross-modal attention
        gnn_attended, gnn_attn_weights = self.gnn_to_transformer_attention(
            gnn_proj,
            transformer_proj,
            transformer_proj,
            key_padding_mask=transformer_mask,
            need_weights=True
        )
        
        transformer_attended, trans_attn_weights = self.transformer_to_gnn_attention(
            transformer_proj,
            gnn_proj,
            gnn_proj,
            key_padding_mask=gnn_mask,
            need_weights=True
        )
        
        attention_weights['gnn_to_transformer'] = gnn_attn_weights
        attention_weights['transformer_to_gnn'] = trans_attn_weights
        
        # Apply gating mechanisms
        gnn_gate = self.gnn_gate(torch.cat([gnn_proj, transformer_attended], dim=-1))
        transformer_gate = self.transformer_gate(torch.cat([transformer_proj, gnn_attended], dim=-1))
        
        gnn_features = gnn_gate * gnn_proj + (1 - gnn_gate) * transformer_attended
        transformer_features = transformer_gate * transformer_proj + (1 - transformer_gate) * gnn_attended
        
        # Hierarchical feature integration
        if self.use_hierarchical:
            hierarchical_features = []
            current_features = torch.cat([gnn_features, transformer_features], dim=-1)
            
            for i, layer in enumerate(self.hierarchical_layers):
                current_features = layer(current_features)
                current_features = F.relu(current_features)
                current_features = self.dropout(current_features)
                hierarchical_features.append(
                    current_features * self.hierarchical_attention[i]
                )
            
            combined_features = sum(hierarchical_features)
        else:
            combined_features = (gnn_features + transformer_features) / 2
        
        # Apply vulnerability-specific attention
        vuln_patterns = self.vulnerability_patterns.unsqueeze(0).expand(
            combined_features.size(0), -1, -1
        )
        
        vuln_attended, vuln_attn_weights = self.vuln_attention(
            combined_features,
            vuln_patterns,
            vuln_patterns,
            need_weights=True
        )
        
        attention_weights['vulnerability'] = vuln_attn_weights
        
        # Final processing
        output = self.layer_norm(combined_features + vuln_attended)
        output = self.output_projection(output)
        
        return output, attention_weights
    
    def get_interpretability_info(self) -> Dict[str, torch.Tensor]:
        """Get interpretable components of the fusion layer."""
        return {
            'vulnerability_patterns': self.vulnerability_patterns.detach(),
            'hierarchical_attention': (
                self.hierarchical_attention.detach() 
                if self.use_hierarchical else None
            ),
            'gnn_gate_weights': self.gnn_gate[0].weight.detach(),
            'transformer_gate_weights': self.transformer_gate[0].weight.detach()
        }
    
    @staticmethod
    def visualize_attention(
        attention_weights: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize attention weights using matplotlib.
        Args:
            attention_weights: Dictionary containing attention weights
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # GNN to Transformer attention
            sns.heatmap(
                attention_weights['gnn_to_transformer'].mean(0).cpu(),
                ax=axes[0],
                cmap='viridis'
            )
            axes[0].set_title('GNN → Transformer\nAttention')
            
            # Transformer to GNN attention
            sns.heatmap(
                attention_weights['transformer_to_gnn'].mean(0).cpu(),
                ax=axes[1],
                cmap='viridis'
            )
            axes[1].set_title('Transformer → GNN\nAttention')
            
            # Vulnerability attention
            sns.heatmap(
                attention_weights['vulnerability'].mean(0).cpu(),
                ax=axes[2],
                cmap='viridis'
            )
            axes[2].set_title('Vulnerability\nAttention')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except ImportError:
            print("Matplotlib and/or seaborn required for visualization")