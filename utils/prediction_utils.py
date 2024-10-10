# api/utils/prediction_utils.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle
from models.data_loader import Custom_Dataset
from models.lstm_imf import Model
import logging

logger = logging.getLogger(__name__)

def preprocess_input(data: List[List[float]], feature_names: List[str], config: dict) -> torch.Tensor:
    """
    Convert input data to the format expected by the model.
    """
    df = pd.DataFrame(data, columns=feature_names)
    
    # Initialize a temporary dataset to leverage scaling and decomposition
    temp_dataset = Custom_Dataset(
        root_path=config['root_path'],
        flag='test',  # Using 'test' for inference
        size=(config['input_seq_len'], config['output_seq_len'], config['lag']),
        data_path=config['data_path'],
        features='S',  # Adjust based on your use case
        max_imfs=config['num_imfs'],
        lag=config['lag'],
        scale=True,
        inverse=False,
        cols=None
    )
    
    # Manually set the data_x based on input
    temp_dataset.scaler.fit(df.values)  # Ideally, load the scaler fitted during training
    temp_dataset.data_x = temp_dataset.scaler.transform(df.values).reshape(1, -1, df.shape[1])
    
    # Apply CEEMDAN decomposition
    if os.path.exists(temp_dataset.decomposition_path):
        with open(temp_dataset.decomposition_path, 'rb') as f:
            decomposition = pickle.load(f)
        data_x_imfs = decomposition['imfs']
        data_x_residue = decomposition['residue']
    else:
        # Perform decomposition if not available
        data_x_imfs, data_x_residue = temp_dataset.apply_ceemdan(temp_dataset.data_x)
        with open(temp_dataset.decomposition_path, 'wb') as f:
            pickle.dump({'imfs': data_x_imfs, 'residue': data_x_residue}, f)
    
    # Convert to tensors
    data_x_imfs_tensor = data_x_imfs.to(config['device'])
    data_x_residue_tensor = data_x_residue.to(config['device'])
    
    return data_x_imfs_tensor, data_x_residue_tensor
