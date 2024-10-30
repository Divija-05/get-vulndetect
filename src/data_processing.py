import torch
from torch_geometric.data import Dataset, Data
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
from solidity_parser import parser

class SmartContractDataset(Dataset):
    def __init__(self, root='data', transform=None, pre_transform=None):
        """
        Initialize the dataset
        Args:
            root: Root directory where the data is stored (default: 'data')
            transform: Any graph transforms to be applied
            pre_transform: Any pre-transforms to be applied
        """
        self.vulnerability_categories = [
            'access_control',
            'arithmetic',
            'denial_of_service',
            'reentrancy',
            'unchecked_low_level_calls'
        ]
        self.vuln_to_idx = {cat: idx for idx, cat in enumerate(self.vulnerability_categories)}
        print(f"Initializing dataset with root directory: {root}")
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """Define the raw file names"""
        return ['ethereum_datasets/vulnerabilities.json']

    @property
    def raw_dir(self):
        """Override raw_dir to match your structure"""
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        """Override processed_dir to match your structure"""
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        """Define the processed file names"""
        return ['train.pt', 'val.pt', 'test.pt', 'statistics.json']

    def load_vulnerabilities(self):
        """Load vulnerability data from the single JSON file"""
        vuln_path = os.path.join(self.raw_dir, 'ethereum_datasets', 'vulnerabilities.json')
        all_contracts = {}

        try:
            with open(vuln_path, 'r') as f:
                contracts = json.load(f)
                print(f"Loading {len(contracts)} contracts from vulnerabilities.json")
                
                for contract in contracts:
                    contract_id = contract['path']
                    
                    # Group vulnerabilities by category
                    vulns_by_category = {}
                    for vuln in contract['vulnerabilities']:
                        category = vuln['category']
                        if category not in vulns_by_category:
                            vulns_by_category[category] = []
                        vulns_by_category[category].extend(vuln['lines'])

                    all_contracts[contract_id] = {
                        'name': contract['name'],
                        'path': contract_id,
                        'pragma': contract.get('pragma', ''),
                        'source': contract.get('source', ''),
                        'vulnerabilities': vulns_by_category,
                        'code': None
                    }

            # Load contract code
            for contract_id, contract_data in all_contracts.items():
                try:
                    # Adjust path to match your structure
                    relative_path = contract_data['path'].replace('dataset/', '')
                    code_path = os.path.join(self.raw_dir, 'ethereum_datasets', relative_path)
                    
                    with open(code_path, 'r', encoding='utf-8') as f:
                        contract_data['code'] = f.read()
                except Exception as e:
                    print(f"Error loading code for {contract_id}: {e}")
                    print(f"Attempted path: {code_path}")

        except Exception as e:
            print(f"Error loading vulnerabilities file: {e}")
            return {}

        return all_contracts

    def _extract_features(self, source_code):
        """Extract features from source code"""
        try:
            # Parse the contract
            ast = parser.parse(source_code)
            
            # Get AST features
            ast_features = self._extract_ast_features(ast)
            
            # Get security pattern features
            security_features = self._extract_security_features(source_code)
            
            # Combine features
            combined_features = np.concatenate([ast_features, security_features])
            
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return a default feature vector
            return np.zeros(32)  # Adjust size based on your feature dimensions

    def _extract_ast_features(self, ast):
        """Extract features from AST"""
        # Initialize feature counters
        features = {
            'StateVariableDeclaration': 0,
            'FunctionDefinition': 0,
            'Mapping': 0,
            'ModifierDefinition': 0,
            'IfStatement': 0,
            'WhileStatement': 0,
            'ForStatement': 0,
            'RequireStatement': 0,
            'AssertStatement': 0,
            'RevertStatement': 0
        }
        
        def traverse_ast(node):
            if isinstance(node, dict):
                node_type = node.get('type', '')
                if node_type in features:
                    features[node_type] += 1
                    
                # Recursively traverse children
                for value in node.values():
                    traverse_ast(value)
            elif isinstance(node, list):
                for item in node:
                    traverse_ast(item)
        
        traverse_ast(ast)
        return np.array(list(features.values()))

    def _extract_security_features(self, source_code):
        """Extract security-related features from source code"""
        security_patterns = {
            'selfdestruct': r'selfdestruct|suicide',
            'delegatecall': r'delegatecall',
            'send': r'\.send\(',
            'call_value': r'\.call\.value\(',
            'tx_origin': r'tx\.origin',
            'block_timestamp': r'block\.timestamp|now',
            'assembly': r'assembly',
            'unchecked_math': r'\+\+|\-\-|\+=|\-='
        }
        
        features = []
        for pattern in security_patterns.values():
            count = len(re.findall(pattern, source_code))
            features.append(count)
            
        return np.array(features)

    def _create_graph(self, source_code):
        """Create a graph representation of the contract"""
        try:
            ast = parser.parse(source_code)
            nodes, edges = self._process_ast_to_graph(ast)
            
            if not nodes:  # If no nodes were created
                return torch.tensor([[0], [0]], dtype=torch.long)
                
            return torch.tensor(edges, dtype=torch.long)
            
        except Exception as e:
            print(f"Error creating graph: {e}")
            return torch.tensor([[0], [0]], dtype=torch.long)

    def _process_ast_to_graph(self, ast):
        """Process AST to create graph structure"""
        nodes = []
        edges = []
        node_to_idx = {}
        current_idx = 0

        def traverse(node, parent_idx=None):
            nonlocal current_idx
            
            if isinstance(node, dict):
                node_type = node.get('type', '')
                if node_type:
                    node_idx = current_idx
                    nodes.append(node_type)
                    node_to_idx[id(node)] = node_idx
                    current_idx += 1
                    
                    if parent_idx is not None:
                        edges.append([parent_idx, node_idx])
                        edges.append([node_idx, parent_idx])  # Add bidirectional edge
                    
                    # Process children
                    for value in node.values():
                        if isinstance(value, (dict, list)):
                            traverse(value, node_idx)
                            
            elif isinstance(node, list):
                for item in node:
                    traverse(item, parent_idx)

        traverse(ast)
        
        if not edges:
            return [], [[0], [0]]
            
        return nodes, list(map(list, zip(*edges)))  # Transpose edges for PyG format

    def process(self):
        """Process the raw data into the format needed for training"""
        print("Starting data processing...")
        
        # Load all vulnerability data
        contracts = self.load_vulnerabilities()
        
        data_list = []
        print(f"Processing {len(contracts)} contracts...")
        
        for contract_id, contract_data in tqdm(contracts.items()):
            try:
                if not contract_data['code']:
                    print(f"Skipping {contract_id} - no code found")
                    continue
                    
                # Extract features and create graph
                x = self._extract_features(contract_data['code'])
                edge_index = self._create_graph(contract_data['code'])
                
                # Create multi-label target
                y = torch.zeros(len(self.vulnerability_categories))
                for category in contract_data['vulnerabilities'].keys():
                    if category in self.vuln_to_idx:
                        y[self.vuln_to_idx[category]] = 1
                
                data = Data(
                    x=torch.tensor(x, dtype=torch.float),
                    edge_index=edge_index,
                    y=y,
                    contract_id=contract_id,
                    name=contract_data['name'],
                    pragma=contract_data['pragma'],
                    source=contract_data['source'],
                    vulnerable_lines=contract_data['vulnerabilities']
                )
                
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data_list.append(data)
                
            except Exception as e:
                print(f"Error processing contract {contract_id}: {e}")
                continue
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Split data into train/val/test
        num_samples = len(data_list)
        indices = torch.randperm(num_samples)
        
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        
        # Save splits
        torch.save(
            [data_list[i] for i in indices[:train_size]],
            os.path.join(self.processed_dir, 'train.pt')
        )
        torch.save(
            [data_list[i] for i in indices[train_size:train_size+val_size]],
            os.path.join(self.processed_dir, 'val.pt')
        )
        torch.save(
            [data_list[i] for i in indices[train_size+val_size:]],
            os.path.join(self.processed_dir, 'test.pt')
        )
        
        # Save statistics
        self._save_statistics(data_list)
        print("Processing completed successfully!")

    def _save_statistics(self, data_list):
        """Save dataset statistics"""
        stats = {
            'total_contracts': len(data_list),
            'vulnerability_distribution': {
                category: sum(1 for data in data_list if data.y[idx] == 1)
                for idx, category in enumerate(self.vulnerability_categories)
            },
            'multi_label_distribution': {
                str(num_vulns): sum(1 for data in data_list 
                                  if sum(data.y) == num_vulns)
                for num_vulns in range(len(self.vulnerability_categories) + 1)
            }
        }
        
        with open(os.path.join(self.processed_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def len(self):
        """Get the number of samples in the dataset"""
        if not os.path.exists(os.path.join(self.processed_dir, 'train.pt')):
            return 0
        return len(torch.load(os.path.join(self.processed_dir, 'train.pt')))

    def get(self, idx):
        """Get a sample from the dataset"""
        data = torch.load(os.path.join(self.processed_dir, 'train.pt'))[idx]
        return data

if __name__ == "__main__":
    """
    Test the dataset processing
    """
    # Initialize and process dataset
    dataset = SmartContractDataset()
    dataset.process()
    
    # Check if processing was successful
    processed_dir = Path('data/processed')
    if processed_dir.exists():
        # Load and display statistics
        with open(processed_dir / 'statistics.json', 'r') as f:
            stats = json.load(f)
        print("\nDataset statistics:")
        print(f"Total processed contracts: {stats['total_contracts']}")
        print("\nVulnerability distribution:")
        for category, count in stats['vulnerability_distribution'].items():
            print(f"- {category}: {count}")
        print("\nMulti-label distribution (number of vulnerabilities per contract):")
        for num_vulns, count in stats['multi_label_distribution'].items():
            print(f"- Contracts with {num_vulns} vulnerability types: {count}")
    else:
        print("Processing failed - no processed directory found")