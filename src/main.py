gnn = SmartContractSAGE(
    input_dim=Config.GNN_INPUT_DIM,
    hidden_dim=Config.GNN_HIDDEN_DIM,
    output_dim=Config.GNN_OUTPUT_DIM,
    num_edge_types=Config.NUM_EDGE_TYPES,
    sample_sizes=Config.SAMPLE_SIZES
)