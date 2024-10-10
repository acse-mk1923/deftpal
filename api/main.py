# api/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle
from models.data_loader import Custom_Dataset
from models.lstm_imf import Model
from models.train_evaluate import aggregate_predictions, visualize_multi_step_forecast
from models.train_evaluate import Config  # Ensure Config is importable
import logging

app = FastAPI(title="IMF LSTM Forecasting API", version="1.0")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define configuration (this should match your training configuration)
CONFIG = {
    'input_seq_len': 6,
    'output_seq_len': 1,
    'n_features': 41,
    'hidden_dim': 8,
    'dropout_rate': 0.05,
    'num_epochs': 1500,
    'num_imfs': 3,
    'learning_rate': 0.01,
    'batch_size': 4,
    'seg_len': 1,
    'dec_way': 'pmf',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'root_path': '/Users/mk1923/deftpal/datasets/long',
    'data_path': 'feature_engineered_data.csv',
    'model_save_dir': '/Users/mk1923/deftpal/train_models',
    'lag': 0
}


# Define input data model
class ForecastRequest(BaseModel):
    data: List[List[float]]  # 2D list representing the time series data
    feature_names: List[str]  # Names of the features

class ForecastResponse(BaseModel):
    predictions: List[List[float]]
    metrics: dict

# Load trained models on startup
@app.on_event("startup")
def load_models():
    global models, model_configs
    logger.info("Loading trained models...")
    
    # Ensure the model_save_dir exists
    if not os.path.exists(CONFIG['model_save_dir']):
        raise FileNotFoundError(f"Model directory {CONFIG['model_save_dir']} does not exist.")
    
    # Initialize model configurations
    model_configs = Config(
        input_seq_len=CONFIG['input_seq_len'],
        output_seq_len=CONFIG['output_seq_len'],
        n_features=CONFIG['n_features'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout_rate=CONFIG['dropout_rate'],
        seg_len=CONFIG['seg_len'],
        dec_way=CONFIG['dec_way'],
        revin=False,  # Set according to your training
        channel_id=False  # Set according to your training
    )
    
    # Load each IMF model and residual model
    models = []
    for imf_idx in range(CONFIG['num_imfs']):
        model = Model(model_configs)
        model_path = os.path.join(CONFIG['model_save_dir'], f'imf_model_{imf_idx}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        models.append(model)
    
    # Load residual model
    residual_model = Model(model_configs)
    residual_model_path = os.path.join(CONFIG['model_save_dir'], 'residual_model.pth')
    if not os.path.exists(residual_model_path):
        raise FileNotFoundError(f"Model file {residual_model_path} not found.")
    residual_model.load_state_dict(torch.load(residual_model_path, map_location=CONFIG['device']))
    residual_model.to(CONFIG['device'])
    residual_model.eval()
    models.append(residual_model)
    
    logger.info("All models loaded successfully.")

# Utility function to process input data
def preprocess_input(data: List[List[float]], feature_names: List[str]) -> torch.Tensor:
    """
    Convert input data to the format expected by the model.
    """
    df = pd.DataFrame(data, columns=feature_names)
    
    # Assume 'Date' column is not needed or already handled
    # If 'Date' is needed, adjust accordingly
    
    # Initialize a temporary dataset to leverage scaling and decomposition
    temp_dataset = Custom_Dataset(
        root_path=CONFIG['root_path'],
        flag='test',  # Using 'test' for inference
        size=(CONFIG['input_seq_len'], CONFIG['output_seq_len'], CONFIG['lag']),
        data_path=CONFIG['data_path'],
        features='S',  # Adjust based on your use case
        max_imfs=CONFIG['num_imfs'],
        lag=CONFIG['lag'],
        scale=True,
        inverse=False,
        cols=None
    )
    
    # Manually set the data_x and data_y based on input
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
    data_x_imfs_tensor = data_x_imfs.to(CONFIG['device'])
    data_x_residue_tensor = data_x_residue.to(CONFIG['device'])
    
    return data_x_imfs_tensor, data_x_residue_tensor

# Prediction endpoint
@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    """
    Predict future values based on the input time series data.
    """
    try:
        # Preprocess input data
        data_x_imfs, data_x_residue = preprocess_input(request.data, request.feature_names)
        
        # Initialize lists to collect predictions
        batch_predictions = []
        
        # Process each IMF
        for imf_idx in range(CONFIG['num_imfs']):
            model = models[imf_idx]
            with torch.no_grad():
                inputs = data_x_imfs[:, :, imf_idx, :]
                outputs = model(inputs)
                batch_predictions.append(outputs.cpu().numpy())
        
        # Process residual
        residual_model = models[-1]
        with torch.no_grad():
            residual_output = residual_model(data_x_residue.squeeze(2))
            batch_predictions.append(residual_output.cpu().numpy())
        
        # Combine predictions
        combined_prediction = np.stack(batch_predictions, axis=2)  # Shape: [batch, pred_len, num_imfs+1, features]
        
        # Sum IMFs and residual to get the final prediction
        final_prediction = combined_prediction.sum(axis=2)  # Shape: [batch, pred_len, features]
        
        # Inverse transform if necessary
        # Assuming the scaler was fitted on training data, apply inverse transform
        inverse_predictions = temp_dataset.scaler.inverse_transform(final_prediction.reshape(-1, final_prediction.shape[-1]))
        inverse_predictions = inverse_predictions.reshape(final_prediction.shape)
        
        # Prepare response
        predictions = inverse_predictions.tolist()
        
        # Optionally, compute metrics here if ground truth is available
        
        return ForecastResponse(predictions=predictions, metrics={})
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Optionally, add an endpoint for health checks
@app.get("/health")
def health_check():
    return {"status": "API is running"}

