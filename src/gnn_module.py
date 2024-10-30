# src/gnn_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import BatchNorm
from typing import Optional, Tuple

class ScalableGATLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = True
    ):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True
        )
        
        self.batch_norm = BatchNorm(out_channels) if batch_norm else None
        self.residual = residual
        self.dropout = dropout
        
        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, attention_weights = self.gat(x, edge_index, return_attention_weights=return_attention)
        
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        
        if self.residual:
            res = self.res_proj(x) if self.res_proj is not None else x
            out = out + res
            
        out = F.elu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        if return_attention:
            return out, attention_weights
        return out, None

class GNNModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim
            
            layer = ScalableGATLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual
            )
            self.gat_layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        # Project input features
        x = self.input_proj(x)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Apply GAT layers
        for layer in self.gat_layers:
            x, attn = layer(x, edge_index, batch, return_attention=return_attention)
            if attn is not None:
                attention_weights.append(attn)
        
        # Final projection
        x = self.output_proj(x)
        
        if return_attention:
            return x, attention_weights
        return x, None

# src/test_gnn_visualization.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gnn_module import GNNModule

def create_test_graph():
    """Create a small test graph"""
    num_nodes = 10
    num_features = 32
    
    # Create node features
    x = torch.randn(num_nodes, num_features)
    
    # Create a simple chain graph with some additional edges
    edge_index = []
    for i in range(num_nodes-1):
        edge_index.extend([[i, i+1], [i+1, i]])  # bidirectional edges
    
    # Add some random edges
    for _ in range(5):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            edge_index.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edge_index).t()
    
    # Create batch indices (assume all nodes are in same graph)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    return x, edge_index, batch

def visualize_attention(attention_weights, num_nodes):
    """Visualize attention weights as a heatmap"""
    # Get attention weights from last layer
    last_layer_attention = attention_weights[-1][1].mean(dim=1)  # Average over heads
    attention_matrix = torch.zeros((num_nodes, num_nodes))
    
    edge_index = attention_weights[-1][0]
    for idx, (i, j) in enumerate(edge_index.t()):
        attention_matrix[i, j] = last_layer_attention[idx]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix.detach().cpu().numpy(), 
                cmap='YlOrRd', 
                xticklabels=range(num_nodes),
                yticklabels=range(num_nodes))
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Target Node')
    plt.ylabel('Source Node')
    plt.show()

def main():
    # Create test data
    x, edge_index, batch = create_test_graph()
    print(f"Created test graph with {x.shape[0]} nodes and {edge_index.shape[1]} edges")
    
    # Initialize model
    model = GNNModule(
        input_dim=x.shape[1],
        hidden_dim=64,
        output_dim=32,
        num_layers=3,
        heads=4,
        dropout=0.1
    )
    
    # Set model to eval mode
    model.eval()
    
    # Forward pass
    print("\nPerforming forward pass...")
    with torch.no_grad():
        output, attention = model(x, edge_index, batch, return_attention=True)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Number of attention layers: {len(attention)}")
    
    # Visualize attention weights
    print("\nVisualizing attention weights...")
    visualize_attention(attention, x.shape[0])
    
    # Print some statistics about the output
    print("\nOutput statistics:")
    print(f"Mean: {output.mean().item():.4f}")
    print(f"Std: {output.std().item():.4f}")
    print(f"Min: {output.min().item():.4f}")
    print(f"Max: {output.max().item():.4f}")

if __name__ == "__main__":
    main()