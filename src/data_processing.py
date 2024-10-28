# data_processing.py

import torch
from torch_geometric.data import Dataset, Data
import os
import json
import requests
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
from solidity_parser import parser  # We'll need to install this

class SmartContractDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        print(f"Initializing dataset with root directory: {root}")
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return ['vulnerability_dataset.json']
        
    @property
    def processed_file_names(self):
        return ['processed_data.pt']
        
    def download(self):
        print("Starting download process...")
        # Using SolidiFI dataset
        url = "https://raw.githubusercontent.com/smartbugs/SolidiFI/master/Dataset/Vulnerable.json"
        
        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"Downloading dataset from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            save_path = os.path.join(self.raw_dir, self.raw_file_names[0])
            with open(save_path, 'w') as f:
                json.dump(response.json(), f)
            print(f"Dataset downloaded successfully to {save_path}")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
            
    def _extract_features(self, source_code):
        """Extract meaningful features from smart contract code"""
        features = []
        try:
            # Parse the Solidity code
            ast = parser.parse(source_code)
            
            # Extract basic features
            features = self._extract_ast_features(ast)
            
            # Add security pattern features
            features.extend(self._extract_security_features(source_code))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return default features if parsing fails
            return np.zeros((10, 32))
            
    def _extract_ast_features(self, ast):
        """Extract features from AST"""
        features = []
        
        # Count contract components
        contract_features = {
            'state_variables': 0,
            'functions': 0,
            'modifiers': 0,
            'events': 0,
            'mappings': 0
        }
        
        # Traverse AST and count components
        # This is a simplified version - we'll expand this
        if 'children' in ast:
            for node in ast['children']:
                if node['type'] == 'ContractDefinition':
                    for child in node.get('subNodes', []):
                        if child['type'] == 'StateVariableDeclaration':
                            contract_features['state_variables'] += 1
                        elif child['type'] == 'FunctionDefinition':
                            contract_features['functions'] += 1
                        elif child['type'] == 'ModifierDefinition':
                            contract_features['modifiers'] += 1
                        elif child['type'] == 'EventDefinition':
                            contract_features['events'] += 1
                            
        return list(contract_features.values())
        
    def _extract_security_features(self, source_code):
        """Extract security-related features"""
        security_patterns = {
            'reentrancy': len(re.findall(r'\.call{value:', source_code)),
            'tx_origin': len(re.findall(r'tx\.origin', source_code)),
            'assembly': len(re.findall(r'assembly', source_code)),
            'selfdestruct': len(re.findall(r'selfdestruct|suicide', source_code)),
            'unchecked_call': len(re.findall(r'\.call\(', source_code))
        }
        
        return list(security_patterns.values())
        
    def _create_graph(self, source_code):
        """Create contract graph from source code"""
        try:
            # Parse the Solidity code
            ast = parser.parse(source_code)
            
            # Create nodes and edges from AST
            nodes, edges = self._process_ast_to_graph(ast)
            
            return np.array(edges)
            
        except Exception as e:
            print(f"Error in graph creation: {e}")
            # Return default graph if parsing fails
            return np.array([[0, 1], [1, 0]])
            
    def _process_ast_to_graph(self, ast):
        """Process AST to create graph structure"""
        nodes = []
        edges = []
        node_counter = 0
        
        def traverse_ast(node, parent_id=None):
            nonlocal node_counter
            current_id = node_counter
            node_counter += 1
            
            # Add node
            nodes.append({
                'id': current_id,
                'type': node.get('type', 'unknown')
            })
            
            # Add edge from parent if exists
            if parent_id is not None:
                edges.append([parent_id, current_id])
                edges.append([current_id, parent_id])  # Make it bidirectional
                
            # Traverse children
            for key, value in node.items():
                if isinstance(value, dict):
                    traverse_ast(value, current_id)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            traverse_ast(item, current_id)
                            
        # Start traversal from root
        traverse_ast(ast)
        
        return nodes, edges
        
    def process(self):
        print("Starting data processing...")
        
        # Read the raw data
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)
            
        data_list = []
        print(f"Processing {len(raw_data)} contracts...")
        
        for contract in tqdm(raw_data):
            try:
                # Extract features and create graph
                x = self._extract_features(contract['source_code'])
                edge_index = self._create_graph(contract['source_code'])
                y = 1 if contract.get('vulnerable', False) else 0
                
                data = Data(
                    x=torch.tensor(x, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    y=torch.tensor([y], dtype=torch.float),
                    code=contract['source_code']
                )
                
                data_list.append(data)
                
            except Exception as e:
                print(f"Error processing contract: {e}")
                continue
                
        # Save processed data
        processed_path = os.path.join(self.processed_dir, 'processed_data.pt')
        torch.save(data_list, processed_path)
        print(f"Successfully processed {len(data_list)} contracts")

if __name__ == "__main__":
    # Install required package if not already installed
    try:
        import solidity_parser
    except ImportError:
        print("Installing solidity-parser...")
        os.system('pip install solidity-parser-antlr')
        
    # Create and test dataset
    dataset = SmartContractDataset(root='../data')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample data properties:")
        print(f"- Features shape: {sample.x.shape}")
        print(f"- Edge index shape: {sample.edge_index.shape}")
        print(f"- Label: {sample.y}")