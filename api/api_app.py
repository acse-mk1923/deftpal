from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os
import torch
import numpy as np
import logging
from models.lstm_imf import Model
from models.train_evaluate import train_imf_model
from models.data_loader import Custom_Dataset
from main import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup your configuration as a global variable or load dynamically
config = {
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
    'root_path': '//Users/mk1923/deftpal/datasets/long',
    'data_path': 'feature_engineered_data.csv',
    'model_save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models'),
    'lag': 0
}

# Define Pydantic Models for request body
class TrainRequest(BaseModel):
    epochs: int
    learning_rate: float
    batch_size: int
    
class EvaluateRequest(BaseModel):
    model_path: str

    class Config:
        protected_namespaces = ()  # Disable the protected namespaces check


@app.get("/")
def read_root():
    return {"message": "Welcome to the LSTM IMF Model API"}

@app.post("/train")
def train_model(train_request: TrainRequest, background_tasks: BackgroundTasks):
    """
    API endpoint to start training the models asynchronously
    """
    background_tasks.add_task(run_training, train_request)
    return {"status": "Training started"}

def run_training(train_request: TrainRequest):
    # Initialize Dataset and DataLoader
    train_dataset = Custom_Dataset(
        root_path=config['root_path'],
        flag='train',
        size=(config['input_seq_len'], config['output_seq_len'], 1),
        data_path=config['data_path'],
        features='M',
        max_imfs=config['num_imfs'],
        lag=config['lag']
    )

    val_dataset = Custom_Dataset(
        root_path=config['root_path'],
        flag='val',
        size=(config['input_seq_len'], config['output_seq_len'], 1),
        data_path=config['data_path'],
        features='M',
        max_imfs=config['num_imfs'],
        lag=config['lag']
    )

    train_loader = DataLoader(train_dataset, batch_size=train_request.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_request.batch_size, shuffle=False)

    # Model Initialization
    models = [Model(config) for _ in range(config['num_imfs'])]
    models.append(Model(config))  # Residual model

    optimizers = [optim.Adam(model.parameters(), lr=train_request.learning_rate) for model in models]
    criterion = nn.MSELoss()

    # Training loop for each IMF model and residual
    for imf_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
        model_desc = f'IMF {imf_idx + 1}' if imf_idx < config['num_imfs'] else 'Residual'
        logger.info(f'Training model for {model_desc}')
        train_imf_model(
            model, train_loader, val_loader, criterion, optimizer, imf_idx, config['num_imfs'],
            num_epochs=train_request.epochs, device=config['device']
        )

    # Save the trained models
    os.makedirs(config['model_save_dir'], exist_ok=True)
    for idx, model in enumerate(models):
        model_name = f'imf_model_{idx}.pth' if idx < config['num_imfs'] else 'residual_model.pth'
        model_path = os.path.join(config['model_save_dir'], model_name)
        torch.save(model.state_dict(), model_path)
    logger.info(f"Trained models saved in: {config['model_save_dir']}")

@app.post("/evaluate")
def evaluate_model(evaluate_request: EvaluateRequest):
    """
    API endpoint to evaluate a model
    """
    model_path = evaluate_request.model_path
    test_dataset = Custom_Dataset(
        root_path=config['root_path'], 
        flag='test', 
        size=(config['input_seq_len'], config['output_seq_len'], 1), 
        data_path=config['data_path'], 
        features='M', 
        max_imfs=config['num_imfs'],
        lag=config['lag']
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    models = [Model(config) for _ in range(config['num_imfs'])]
    models.append(Model(config))  # Residual model
    
    # Load trained model
    for idx, model in enumerate(models):
        model_name = f'imf_model_{idx}.pth' if idx < config['num_imfs'] else 'residual_model.pth'
        model_path = os.path.join(model_path, model_name)
        model.load_state_dict(torch.load(model_path))
    
    criterion = nn.MSELoss()
    test_results = aggregate_predictions(models, test_loader, criterion, config['num_imfs'], config['device'])
    
    return {
        "mse_original": test_results["mse_original"],
        "mae_original": test_results["mae_original"],
        "mape_original": test_results["mape_original"]
    }

