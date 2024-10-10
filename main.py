# api/main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.data_loader import *
import logging
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from models.lstm_imf import Model  # Import your model definitions
from models.train_evaluate import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


import os
import numpy as np
import matplotlib.pyplot as plt

def aggregate_predictions(models, test_loader, criterion, num_imfs, device, imfs_to_use=None):
    all_predictions = []
    all_targets_decomposed = []
    all_targets_original = []
    mse_decomposed = 0
    mse_original = 0
    mae_decomposed = 0
    mae_original = 0
    mape_decomposed = 0
    mape_original = 0
    num_batches = 0

    # If imfs_to_use is not specified, use all IMFs
    if imfs_to_use is None:
        imfs_to_use = list(range(num_imfs))
    
    total_components = len(imfs_to_use) + 1  # +1 for the residual

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            seq_x_imfs, seq_y_imfs, seq_x_residue, seq_y_residue, original_x, original_y = batch

            # Move data to device
            seq_x_imfs = seq_x_imfs.float().to(device)
            seq_y_imfs = seq_y_imfs.float().to(device)
            seq_x_residue = seq_x_residue.float().to(device)
            seq_y_residue = seq_y_residue.float().to(device)
            original_y = original_y.float().to(device)

            batch_predictions = []

            # Process selected IMFs
            for i in imfs_to_use:
                inputs = seq_x_imfs[:, :, i, :]
                outputs = models[i](inputs)
                batch_predictions.append(outputs)

            # Process residual
            residual_outputs = models[-1](seq_x_residue.squeeze(2))
            batch_predictions.append(residual_outputs)

            # Combine predictions
            combined_prediction = torch.stack(batch_predictions, dim=2)

            # Prepare targets
            selected_imfs = seq_y_imfs[:, :, imfs_to_use, :]
            targets_decomposed = torch.cat((selected_imfs, seq_y_residue), dim=2)

            # Ensure predictions and targets have the same shape
            min_length = min(combined_prediction.shape[1], targets_decomposed.shape[1])
            combined_prediction = combined_prediction[:, :min_length, :, :]
            targets_decomposed = targets_decomposed[:, :min_length, :, :]
            original_y = original_y[:, :min_length, :]

            # Calculate losses
            loss_decomposed = criterion(combined_prediction, targets_decomposed)
            loss_original = criterion(combined_prediction.sum(dim=2), original_y)
            
            # Calculate MAE
            mae_decomposed_batch = torch.mean(torch.abs(combined_prediction - targets_decomposed))
            mae_original_batch = torch.mean(torch.abs(combined_prediction.sum(dim=2) - original_y))
            
            # Calculate MAPE with handling for zero/near-zero values
            epsilon = 1e-8  # Small constant to avoid division by zero
            
            # For decomposed predictions
            abs_percentage_error_decomposed = torch.abs((targets_decomposed - combined_prediction) / (targets_decomposed + epsilon))
            mape_decomposed_batch = torch.mean(torch.where(targets_decomposed != 0, abs_percentage_error_decomposed, torch.zeros_like(abs_percentage_error_decomposed))) * 100
            
            # For original predictions
            abs_percentage_error_original = torch.abs((original_y - combined_prediction.sum(dim=2)) / (original_y + epsilon))
            mape_original_batch = torch.mean(torch.where(original_y != 0, abs_percentage_error_original, torch.zeros_like(abs_percentage_error_original))) * 100

            mse_decomposed += loss_decomposed.item()
            mse_original += loss_original.item()
            mae_decomposed += mae_decomposed_batch.item()
            mae_original += mae_original_batch.item()
            mape_decomposed += mape_decomposed_batch.item()
            mape_original += mape_original_batch.item()
            num_batches += 1

            all_predictions.append(combined_prediction.sum(dim=2).cpu().numpy())  # Summing IMFs and residual
            all_targets_decomposed.append(targets_decomposed.cpu().numpy())
            all_targets_original.append(original_y.cpu().numpy())

    mse_decomposed /= num_batches
    mse_original /= num_batches
    mae_decomposed /= num_batches
    mae_original /= num_batches
    mape_decomposed /= num_batches
    mape_original /= num_batches

    # print(f"\nFinal Results:")
    # print(f"Number of batches processed: {num_batches}")
    # print(f"Number of IMFs used: {len(imfs_to_use)}")
    # print(f"Total components (IMFs + residual): {total_components}")
    # print(f"MSE (decomposed): {mse_decomposed:.4f}")
    # print(f"MSE (original): {mse_original:.4f}")
    # print(f"MAE (decomposed): {mae_decomposed:.4f}")
    # print(f"MAE (original): {mae_original:.4f}")
    # print(f"MAPE (decomposed): {mape_decomposed:.2f}%")
    # print(f"MAPE (original): {mape_original:.2f}%")

    return {
        "mse_decomposed": mse_decomposed,
        "mse_original": mse_original,
        "mae_decomposed": mae_decomposed,
        "mae_original": mae_original,
        "mape_decomposed": mape_decomposed,
        "mape_original": mape_original,
        "predictions": all_predictions,
        "targets_decomposed": all_targets_decomposed,
        "targets_original": all_targets_original
    }

import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_multi_step_forecast(predictions, targets, feature_names, base_dir=None, feature_indices=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    save_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert lists to numpy arrays if they're not already
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions, axis=0)
    if isinstance(targets, list):
        targets = np.concatenate(targets, axis=0)
    
    # If feature_indices is provided, select only those features
    if feature_indices is not None:
        predictions = predictions[:, :, feature_indices]
        targets = targets[:, :, feature_indices]
        feature_names = [feature_names[i] for i in feature_indices]
    
    num_samples, forecast_length, num_features = predictions.shape
    print(f"Data shape: {num_samples} samples, {forecast_length}-step forecast, {num_features} features")
    
    assert num_features == len(feature_names), f"Number of features ({num_features}) doesn't match the number of feature names ({len(feature_names)})"
    
    # Create a plot for each feature
    for feature_idx, feature_name in enumerate(feature_names):
        plt.figure(figsize=(15, 6))
        plt.title(f'Complete Test Data Comparison: {feature_name}')
        
        # Flatten the data for plotting
        target_flat = targets[:, :, feature_idx].flatten()
        pred_flat = predictions[:, :, feature_idx].flatten()
        
        # Create x-axis values
        x = np.arange(len(target_flat))
        
        plt.plot(x, target_flat, label='Ground Truth', alpha=0.7)
        plt.plot(x, pred_flat, label='Forecast', alpha=0.7)
        
        plt.xlabel('Time Steps')
        plt.ylabel(feature_name)
        plt.legend()
        
        # Add vertical lines to separate different samples
        for i in range(1, num_samples):
            plt.axvline(x=i*forecast_length, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'complete_test_comparison_{feature_name}.png')
        plt.savefig(save_path)
        plt.close()

    print(f"Saved complete test data comparison visualizations in {save_dir}")
    return save_dir

# Configure the logging to ignore less severe messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Config:
    def __init__(self, input_seq_len, output_seq_len, n_features, hidden_dim, dropout_rate, seg_len, dec_way, revin=False, channel_id=False):
        self.seq_len = input_seq_len
        self.pred_len = output_seq_len
        self.enc_in = n_features
        self.d_model = hidden_dim
        self.dropout = dropout_rate
        self.seg_len = seg_len
        self.dec_way = dec_way
        self.revin = revin
        self.channel_id = channel_id

def warmup_lambda(epoch):
    if epoch < 5:
        return 0.2 * (epoch + 1)
    return 1.0
def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)


def main():
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
        'lag': 0  # Add this line to include the lag parameter
    }

    # Initialize Dataset and DataLoader
    logger.info("Initializing datasets and dataloaders...")

    train_dataset = Custom_Dataset(
        root_path=config['root_path'], 
        flag='train', 
        size=(config['input_seq_len'], config['output_seq_len'], 1), 
        data_path=config['data_path'], 
        features='M', 
        max_imfs=config['num_imfs'],
        lag=config['lag']  # Add this line
    )

    val_dataset = Custom_Dataset(
        root_path=config['root_path'], 
        flag='val', 
        size=(config['input_seq_len'], config['output_seq_len'], 1), 
        data_path=config['data_path'], 
        features='M', 
        max_imfs=config['num_imfs'],
        lag=config['lag']  # Add this line
    )

    test_dataset = Custom_Dataset(
        root_path=config['root_path'], 
        flag='test', 
        size=(config['input_seq_len'], config['output_seq_len'], 1), 
        data_path=config['data_path'], 
        features='M', 
        max_imfs=config['num_imfs'],
        lag=config['lag']  # Add this line
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    logger.info("Initializing models...")
    Model_configs = Config(
        input_seq_len=config['input_seq_len'],
        output_seq_len=config['output_seq_len'],
        n_features=config['n_features'],
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate'],
        seg_len=config['seg_len'],
        dec_way=config['dec_way']
    )

    models = [Model(Model_configs) for _ in range(config['num_imfs'])]
    models.append(Model(Model_configs))  # Residual model

    # Initialize weights for each model individually
    for model in models:
        initialize_weights(model)

    optimizers = [optim.Adam(model.parameters(), lr=config['learning_rate']) for model in models]
    criterion = nn.MSELoss()

    # Training loop for each IMF model and the residual model
    for imf_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
        model_desc = f'IMF {imf_idx + 1}' if imf_idx < config['num_imfs'] else 'Residual'
        logger.info(f'Training model for {model_desc}')
        model = train_imf_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            imf_idx, 
            config['num_imfs'], 
            num_epochs=config['num_epochs'], 
            device=config['device'],
            patience=2,
            max_grad_norm=1.0,
            warm_reset_threshold=10,
            reversion_epochs=5
        )
        models[imf_idx] = model  # Update the model in the list

    # Evaluate on the validation set
    logger.info("Evaluating models on validation set...")
    val_results = aggregate_predictions(models, val_loader, criterion, config['num_imfs'], config['device'])
    logger.info(f'Validation MSE: {val_results["mse_decomposed"]:.4f}')
    logger.info(f'Validation MAE: {val_results["mae_decomposed"]:.4f}')

    # Final evaluation on the test set
    logger.info("Performing final evaluation on test set...")
    test_results = aggregate_predictions(models, test_loader, criterion, config['num_imfs'], config['device'])

    # Reshape and inverse transform the predictions and targets
    predictions = np.concatenate(test_results["predictions"], axis=0)
    targets = np.concatenate(test_results["targets_original"], axis=0)
    
    num_samples, forecast_length, num_features = predictions.shape
    
    # Reshape to 2D
    predictions_2d = predictions.reshape(-1, num_features)
    targets_2d = targets.reshape(-1, num_features)
    
    # Inverse transform
    inverse_predictions = test_dataset.inverse_transform(predictions_2d)
    inverse_targets = test_dataset.inverse_transform(targets_2d)
    
    # Reshape back to 3D
    inverse_predictions = inverse_predictions.reshape(num_samples, forecast_length, num_features)
    inverse_targets = inverse_targets.reshape(num_samples, forecast_length, num_features)

    # Visualize predictions for all test data
    vis_save_dir = visualize_multi_step_forecast(
        inverse_predictions,
        inverse_targets,
        feature_names=test_dataset.feature_names,
        feature_indices=[test_dataset.feature_names.index('Monthly quantity')] if 'Monthly quantity' in test_dataset.feature_names else None
    )
    logger.info(f"Visualizations saved in: {vis_save_dir}")

    # Save the trained models
    logger.info("Saving trained models...")
    os.makedirs(config['model_save_dir'], exist_ok=True)
    for idx, model in enumerate(models):
        model_name = f'imf_model_{idx}.pth' if idx < config['num_imfs'] else 'residual_model.pth'
        model_path = os.path.join(config['model_save_dir'], model_name)
        torch.save(model.state_dict(), model_path)
    logger.info(f"Trained models saved in: {config['model_save_dir']}")

    logger.info("Training and evaluation completed.")

if __name__ == "__main__":
    main()