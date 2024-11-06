# src/gnn_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import List, Optional, Tuple

class EnhancedSAGELayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 3,  # control flow, data flow, call graph
        aggr: str = 'mean'
    ):
        super().__init__()
        self.num_edge_types = num_edge_types
        
        # Separate convolutions for different edge types
        self.convs = nn.ModuleList([
            SAGEConv(
                in_channels=in_channels,
                out_channels=out_channels,
                normalize=True,
                aggr=aggr
            ) for _ in range(num_edge_types)
        ])
        
        # Edge type attention
        self.edge_attention = nn.Parameter(torch.ones(num_edge_types) / num_edge_types)
        
        # Layer for combining different edge type contributions
        self.combine = nn.Linear(out_channels * num_edge_types, out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: dict,  # Different edge indices for different types
        edge_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Process each edge type separately
        edge_type_outputs = []
        
        for i in range(self.num_edge_types):
            if i in edge_index_dict:
                # Apply edge-type specific convolution
                edge_type_out = self.convs[i](x, edge_index_dict[i])
                # Weight by learned attention
                edge_type_out = edge_type_out * self.edge_attention[i]
                edge_type_outputs.append(edge_type_out)
            else:
                # If edge type not present, add zeros
                edge_type_outputs.append(torch.zeros_like(x))
        
        # Combine outputs from different edge types
        combined = torch.cat(edge_type_outputs, dim=-1)
        out = self.combine(combined)
        
        return out

class SmartContractSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        pool_type: str = 'mean_max',
        num_edge_types: int = 3,
        sample_sizes: List[int] = [25, 10, 5]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_type = pool_type
        self.sample_sizes = sample_sizes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Enhanced SAGE layers
        self.layers = nn.ModuleList([
            EnhancedSAGELayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_edge_types=num_edge_types
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(
            hidden_dim * 2 if pool_type == 'mean_max' else hidden_dim,
            output_dim
        )
        
        # Hierarchical attention
        self.hierarchy_attention = nn.Parameter(torch.ones(num_layers) / num_layers)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: dict,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Initial projection
        x = self.input_proj(x)
        
        # Store intermediate representations
        intermediate = []
        
        # Process each layer
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            # Apply layer
            x = layer(x, edge_index_dict)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Store intermediate representation
            if batch is not None:
                if self.pool_type == 'mean':
                    pooled = global_mean_pool(x, batch)
                elif self.pool_type == 'max':
                    pooled = global_max_pool(x, batch)
                else:  # mean_max
                    pooled = torch.cat([
                        global_mean_pool(x, batch),
                        global_max_pool(x, batch)
                    ], dim=-1)
                intermediate.append(pooled * self.hierarchy_attention[i])
        
        # Combine intermediate representations
        if intermediate:
            x = torch.stack(intermediate).sum(0)
        
        # Final projection
        x = self.output_proj(x)
        
        return x

    def get_attention_weights(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get attention weights for interpretability."""
        edge_type_attention = [layer.edge_attention for layer in self.layers]
        return self.hierarchy_attention, edge_type_attention