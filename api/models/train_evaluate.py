import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import *
import logging
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import copy
#set seed
torch.manual_seed(42)
np.random.seed(42)


def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            print(f'Gradient range for {name}: min={grad_min}, max={grad_max}')

def process_batch(batch, model, imf_index, num_imfs, device):
    seq_x_imfs, seq_y_imfs, seq_x_residue, seq_y_residue, original_x, original_y = batch

    seq_x_imfs = seq_x_imfs.float().to(device)
    seq_y_imfs = seq_y_imfs.float().to(device)
    seq_x_residue = seq_x_residue.float().to(device)
    seq_y_residue = seq_y_residue.float().to(device)

    if imf_index < num_imfs:
        inputs = seq_x_imfs[:, :, imf_index, :]
        targets = seq_y_imfs[:, 1:, imf_index, :]
    else:
        inputs = seq_x_residue
        targets = seq_y_residue[:, 1:, :]

    if inputs.dim() == 4:
        inputs = inputs.squeeze(2)
    if targets.dim() == 4:
        targets = targets.squeeze(2)

    outputs = model(inputs)

    if outputs.shape != targets.shape:
        outputs = outputs[:, :targets.shape[1], :]

    return outputs, targets

def warm_reset(model, optimizer, lr_factor=0.8):
    current_lr = optimizer.param_groups[0]['lr']
    
    for param in model.parameters():
        param.data = param.data + torch.randn_like(param.data) * 0.01
    
    optimizer = type(optimizer)(model.parameters(), lr=current_lr * lr_factor)
    
    return model, optimizer

def train_imf_model(model, train_loader, val_loader, criterion, optimizer, imf_index, num_imfs, num_epochs=50, device='mps', patience=50, max_grad_norm=1.0, window_size=100, spike_threshold=2.0, warm_reset_threshold=10, reversion_epochs=5):
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    warm_reset_counter = 0
    epochs_since_reset = 0

    print(f"Training IMF model for index: {imf_index}, num_imfs: {num_imfs}")
    print(f"Device: {device}, Max grad norm: {max_grad_norm}")
    print(f"Window size: {window_size}, Spike threshold: {spike_threshold}")
    print(f"Warm reset threshold: {warm_reset_threshold}, Reversion epochs: {reversion_epochs}")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        losses = []
        moving_avg = None

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            outputs, targets = process_batch(batch, model, imf_index, num_imfs, device)

            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print("NaN detected in loss. Skipping this batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            current_loss = loss.item()
            total_train_loss += current_loss
            losses.append(current_loss)

            if moving_avg is None:
                moving_avg = current_loss
            else:
                moving_avg = 0.99 * moving_avg + 0.01 * current_loss

            if len(losses) >= window_size:
                recent_avg = sum(losses[-window_size:]) / window_size
                if recent_avg > spike_threshold * moving_avg:
                    break

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {current_loss:.4f}, Moving Avg: {moving_avg:.4f}")

        model.eval()
        total_val_loss = 0
        print("Validation phase:")
        with torch.no_grad():
            for batch in val_loader:
                outputs, targets = process_batch(batch, model, imf_index, num_imfs, device)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            warm_reset_counter = 0
            epochs_since_reset = 0
        else:
            patience_counter += 1
            warm_reset_counter += 1
            epochs_since_reset += 1
            
            if warm_reset_counter >= warm_reset_threshold:
                print(f"Warm reset triggered after {warm_reset_counter} epochs without improvement")
                model, optimizer = warm_reset(model, optimizer)
                warm_reset_counter = 0
                epochs_since_reset = 0
            elif patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Check if we should revert to the best model
        if epochs_since_reset >= reversion_epochs:
            print(f"Reverting to best model after {epochs_since_reset} epochs since last improvement")
            model.load_state_dict(best_model_state)
            epochs_since_reset = 0

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Ensure the final model is the best one
    model.load_state_dict(best_model_state)
    return model

# def train_imf_model(model, train_loader, criterion, optimizer, imf_index, num_imfs, *, num_epochs=50, device='mps', patience=50, max_grad_norm=1.0, window_size=100, spike_threshold=2.0, warm_reset_threshold=10, reversion_epochs=5):
#     model.to(device)
#     best_train_loss = float('inf')
#     best_model_state = None
#     warm_reset_counter = 0

#     print(f"Training IMF model on full dataset for index: {imf_index}, num_imfs: {num_imfs}")
#     print(f"Device: {device}, Max grad norm: {max_grad_norm}")
#     print(f"Window size: {window_size}, Spike threshold: {spike_threshold}")
#     print(f"Warm reset threshold: {warm_reset_threshold}")

#     for epoch in range(num_epochs):
#         model.train()
#         total_train_loss = 0
#         losses = []
#         moving_avg = None

#         for i, batch in enumerate(train_loader):
#             optimizer.zero_grad()

#             outputs, targets = process_batch(batch, model, imf_index, num_imfs, device)

#             loss = criterion(outputs, targets)

#             if torch.isnan(loss):
#                 print("NaN detected in loss. Skipping this batch.")
#                 continue

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
#             optimizer.step()

#             current_loss = loss.item()
#             total_train_loss += current_loss
#             losses.append(current_loss)

#             if moving_avg is None:
#                 moving_avg = current_loss
#             else:
#                 moving_avg = 0.99 * moving_avg + 0.01 * current_loss

#             if len(losses) >= window_size:
#                 recent_avg = sum(losses[-window_size:]) / window_size
#                 if recent_avg > spike_threshold * moving_avg:
#                     print(f"Training spike detected. Recent Avg: {recent_avg}, Moving Avg: {moving_avg}")
#                     break

#             if i % 100 == 0:
#                 print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {current_loss:.4f}, Moving Avg: {moving_avg:.4f}")

#         avg_train_loss = total_train_loss / len(train_loader)

#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

#         # Save the best model state if the training loss improves
#         if avg_train_loss < best_train_loss:
#             best_train_loss = avg_train_loss
#             best_model_state = copy.deepcopy(model.state_dict())
#             warm_reset_counter = 0
#         else:
#             warm_reset_counter += 1

#             if warm_reset_counter >= warm_reset_threshold:
#                 print(f"Warm reset triggered after {warm_reset_counter} epochs without improvement")
#                 model, optimizer = warm_reset(model, optimizer)
#                 warm_reset_counter = 0

#     print(f"Training completed. Best training loss: {best_train_loss:.4f}")

#     # Ensure the final model is the best one
#     model.load_state_dict(best_model_state)
#     return model
