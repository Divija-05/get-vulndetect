# src/get_vulndetect_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .gnn_module import SmartContractSAGE
from .transformer_module import SmartContractTransformer
from .fusion_layer import EnhancedFusionLayer

class GETVulnDetect(nn.Module):
    """
    Graph-Enhanced Transformer for Smart Contract Vulnerability Detection (GET-VulnDetect).
    Combines GNN and Transformer approaches with an enhanced fusion mechanism.
    """
    def __init__(
        self,
        input_dim: int,
        gnn_hidden_dim: int = 256,
        transformer_hidden_dim: int = 256,
        output_dim: int = 5,  # Number of vulnerability types
        gnn_num_layers: int = 3,
        transformer_num_layers: int = 4,
        num_heads: int = 1,
        dropout: float = 0.1,
        max_length: int = 512,
        num_edge_types: int = 3,
        use_hierarchical: bool = True,
        gnn_sample_sizes: List[int] = [25, 10, 5]
    ):
        #print(f"output_dim: {output_dim}, num_heads: {num_heads}")
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        super().__init__()
        
        self.input_dim = input_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.output_dim = output_dim
        
        # GNN Module
        self.gnn = SmartContractSAGE(
            input_dim=input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
            num_edge_types=num_edge_types,
            sample_sizes=gnn_sample_sizes
        )
        
        # Transformer Module
        self.transformer = SmartContractTransformer(
            hidden_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_length
        )
        
        # Fusion Layer
        self.fusion = EnhancedFusionLayer(
            gnn_dim=gnn_hidden_dim,
            transformer_dim=transformer_hidden_dim,
            output_dim=output_dim,
            num_vulnerability_types=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_hierarchical=use_hierarchical
        )
        
        # Additional components for enhanced learning
        self.vulnerability_embeddings = nn.Parameter(
            torch.randn(output_dim, transformer_hidden_dim)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()  # For multi-label classification
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[int, torch.Tensor],
        code: Union[str, List[str]],
        batch: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of the GET-VulnDetect model.
        
        Args:
            x: Node features tensor
            edge_index_dict: Dictionary of edge indices for different edge types
            code: Raw Solidity code string or list of strings
            batch: Batch indices for graph data
            attention_mask: Attention mask for transformer
            return_attention: Whether to return attention weights
            
        Returns:
            torch.Tensor: Vulnerability predictions
            Dict[str, torch.Tensor]: Attention weights (if return_attention=True)
        """
        # Process through GNN
        gnn_output = self.gnn(x, edge_index_dict, batch)
        
        # Process through Transformer
        if isinstance(code, list):
            # Batch processing
            transformer_sequence, transformer_pooled = zip(*[
                self.transformer(c, attention_mask) for c in code
            ])
            transformer_sequence = torch.stack(transformer_sequence)
            transformer_pooled = torch.stack(transformer_pooled)
        else:
            # Single instance
            transformer_sequence, transformer_pooled = self.transformer(
                code,
                attention_mask
            )
        
        # Combine through fusion layer
        fused_output, attention_weights = self.fusion(
            gnn_output,
            transformer_pooled,
            batch=batch
        )
        
        # Final prediction
        predictions = self.output_layer(fused_output)
        
        if return_attention:
            return predictions, attention_weights
        return predictions
    
    def predict_vulnerabilities(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[int, torch.Tensor],
        code: Union[str, List[str]],
        batch: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        return_probabilities: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict vulnerabilities with thresholding.
        
        Args:
            x: Node features tensor
            edge_index_dict: Dictionary of edge indices
            code: Raw Solidity code
            batch: Batch indices for graph data
            threshold: Probability threshold for positive prediction
            return_probabilities: Whether to return raw probabilities
            
        Returns:
            torch.Tensor: Binary predictions
            torch.Tensor: Probability scores (if return_probabilities=True)
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(x, edge_index_dict, code, batch)
            predictions = (probabilities > threshold).float()
            
        if return_probabilities:
            return predictions, probabilities
        return predictions
    
    def get_interpretability_info(
        self
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get interpretable components of the model.
        
        Returns:
            Dict containing:
            - Vulnerability embeddings
            - GNN attention weights
            - Transformer attention patterns
            - Fusion layer information
        """
        return {
            'vulnerability_embeddings': self.vulnerability_embeddings.detach(),
            'gnn_interpretability': self.gnn.get_attention_weights(),
            'transformer_patterns': self.transformer.get_vulnerability_embeddings(),
            'fusion_info': self.fusion.get_interpretability_info()
        }
    
    def get_vulnerability_explanations(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[int, torch.Tensor],
        code: str,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate explanations for vulnerability predictions.
        
        Args:
            x: Node features tensor
            edge_index_dict: Dictionary of edge indices
            code: Raw Solidity code
            batch: Batch indices for graph data
            
        Returns:
            Dict containing attention weights and importance scores
        """
        predictions, attention_weights = self.forward(
            x,
            edge_index_dict,
            code,
            batch,
            return_attention=True
        )
        
        # Get token-level contributions
        token_contributions = attention_weights['transformer_to_gnn'].mean(1)
        
        # Get node-level contributions
        node_contributions = attention_weights['gnn_to_transformer'].mean(1)
        
        # Get vulnerability-specific attention
        vuln_attention = attention_weights['vulnerability']
        
        return {
            'predictions': predictions,
            'token_contributions': token_contributions,
            'node_contributions': node_contributions,
            'vulnerability_attention': vuln_attention
        }