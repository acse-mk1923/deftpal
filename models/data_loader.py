#api/models/data_loader.py

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import logging
from joblib import Parallel, delayed
import torch
import pickle
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('PyEMD.EMD').setLevel(logging.WARNING)

from PyEMD import EMD, CEEMDAN

# Updated patch for EMD
def patched_common_dtype(self, x, y):
    common_type = np.result_type(x, y)
    return np.array(x, dtype=common_type), np.array(y, dtype=common_type)

EMD._common_dtype = patched_common_dtype
class Custom_Dataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='feature_engineered_data.csv', 
                 target='Monthly quantity', scale=True, inverse=False, cols=None, max_imfs=8, lag=0):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.lag = lag  # New parameter for lag
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.max_imfs = max_imfs
        
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        
        self.decomposition_path = os.path.join(root_path, f'{flag}_ceemdan_decomposition.pkl')
        
        # Create directories for saving plots
        self.plot_dir = os.path.join(root_path, 'imf_plots', flag)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.__read_data__()
        
        print(f"DEBUG: After __read_data__(), self.data_x shape: {self.data_x.shape}")
        print(f"DEBUG: self.seq_len: {self.seq_len}, self.pred_len: {self.pred_len}, self.lag: {self.lag}")

        # Add these checks after __read_data__()
        if not hasattr(self, 'data_x') or not hasattr(self, 'data_y'):
            raise AttributeError("data_x or data_y not set. Check __read_data__ method.")
        if not hasattr(self, 'data_x_imfs') or not hasattr(self, 'data_x_residue'):
            raise AttributeError("data_x_imfs or data_x_residue not set. Check CEEMDAN decomposition.")

        logger.debug(f"Dataset initialized. data_x shape: {self.data_x.shape}, "
                     f"data_x_imfs shape: {self.data_x_imfs.shape}")

    def __read_data__(self):
        logger.debug("Starting data reading process")
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        logger.debug(f"Raw data shape: {df_raw.shape}")

        # Convert Date to datetime
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        
        # Sort by date to ensure chronological order
        df_raw = df_raw.sort_values('Date')

        # Calculate total length and split points
        total_length = len(df_raw)
        train_length = int(total_length * 0.333)
        val_length = int(total_length * 0.333)
        test_length = total_length - train_length - val_length

        # Adjust the borders so that there is no overlap between the sets
        border1s = [0, train_length, train_length + val_length]
        border2s = [train_length, train_length + val_length, total_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[df_raw.columns.get_loc('Unit Price in Cedis'):]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        self.feature_names = df_data.columns.tolist()
        logger.debug(f"Features: {self.feature_names}")

        if self.scale:
            logger.debug("Scaling data")
            train_data = df_data[:train_length]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        logger.debug(f"Processed data shape: {data.shape}")
        
        # Set data_x and data_y attributes
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        
        # Store the original data before decomposition
        self.original_data = torch.tensor(self.data_x).float()
        
        # Only decompose the relevant subset
        subset_data = self.data_x

        if os.path.exists(self.decomposition_path):
            logger.debug(f"Loading existing decomposition for {self.flag}")
            with open(self.decomposition_path, 'rb') as f:
                decomposition = pickle.load(f)
            self.data_x_imfs = decomposition['imfs']
            self.data_x_residue = decomposition['residue']
        else:
            logger.debug(f"Performing CEEMDAN decomposition for {self.flag}")
            self.data_x_imfs, self.data_x_residue = self.apply_ceemdan(subset_data)
            with open(self.decomposition_path, 'wb') as f:
                pickle.dump({'imfs': self.data_x_imfs, 'residue': self.data_x_residue}, f)
        
        self.data_y_imfs = self.data_x_imfs
        self.data_y_residue = self.data_x_residue

        # Generate and save plots
        self.plot_imfs(self.data_x_imfs, self.data_x_residue, 
                       save_path=os.path.join(self.plot_dir, 'all_features_imfs.png'), 
                       feature_names=self.feature_names)

        logger.debug("Data reading process completed")

        
    def plot_imfs(self, data_imfs, residue, save_path=None, feature_names=None):
        n_features = data_imfs.shape[2]
        n_imfs = data_imfs.shape[1]

        if feature_names is None or len(feature_names) != n_features:
            logger.warning("Feature names are missing or mismatched. Using default names.")
            feature_names = [f'Feature {i + 1}' for i in range(n_features)]

        fig, axs = plt.subplots(n_features, n_imfs + 1, figsize=(4*(n_imfs + 1), 3*n_features))
        fig.suptitle(f"IMFs for All Features ({self.flag})", fontsize=16)

        # Convert PyTorch tensors to numpy arrays
        data_imfs_np = data_imfs.cpu().numpy()
        residue_np = residue.cpu().numpy()

        for feature in range(n_features):
            feature_name = feature_names[feature]
            # Plot IMFs
            for imf in range(n_imfs):
                ax = axs[feature, imf] if n_features > 1 else axs[imf]
                ax.plot(data_imfs_np[:, imf, feature])
                ax.set_title(f'IMF {imf + 1}')
                if feature == n_features - 1:
                    ax.set_xlabel('Time Steps')
                if imf == 0:
                    ax.set_ylabel(feature_name)

            # Plot residue
            ax = axs[feature, -1] if n_features > 1 else axs[-1]
            ax.plot(residue_np[:, feature])
            ax.set_title('Residue')
            if feature == n_features - 1:
                ax.set_xlabel('Time Steps')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"IMF plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_decomposition(self):
        with open(self.decomposition_path, 'rb') as f:
            return pickle.load(f)
    def apply_ceemdan(self, data):
        logger.debug("Starting CEEMDAN decomposition")
        ceemdan = CEEMDAN()
        
        data = data.astype(np.float32)

        def process_feature(feature):
            feature_imfs = ceemdan(feature, max_imf=self.max_imfs)
            
            if len(feature_imfs) < self.max_imfs:
                feature_imfs = np.pad(feature_imfs, ((0, self.max_imfs - len(feature_imfs)), (0, 0)), mode='constant')
            elif len(feature_imfs) > self.max_imfs:
                feature_imfs = feature_imfs[:self.max_imfs]
            
            residue = feature - np.sum(feature_imfs, axis=0)
            return feature_imfs, residue

        # Ensure we're using parallel processing
        results = Parallel(n_jobs=-1, verbose=10)(delayed(process_feature)(feature) for feature in data.T)
        imfs, residues = zip(*results)

        logger.debug("CEEMDAN decomposition completed")
        
        # Ensure imfs and residues are numpy arrays
        imfs = np.array(imfs, dtype=np.float32)
        residues = np.array(residues, dtype=np.float32)
        
        # Reshape the IMFs to (seq_len, num_imfs, channels)
        imfs = imfs.transpose(2, 1, 0)
        residues = residues.T
        
        logger.debug(f"IMFs shape: {imfs.shape}, Residues shape: {residues.shape}")
        
        # Convert to PyTorch tensors manually
        imfs_tensor = torch.tensor(imfs.tolist(), dtype=torch.float32)
        residues_tensor = torch.tensor(residues.tolist(), dtype=torch.float32)
        
        logger.debug(f"IMFs tensor shape: {imfs_tensor.shape}, Residues tensor shape: {residues_tensor.shape}")
        
        return imfs_tensor, residues_tensor


    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len 
    #     r_end = r_begin + self.label_len + self.pred_len

    #     seq_x_imfs = self.data_x_imfs[s_begin:s_end]
    #     seq_y_imfs = self.data_y_imfs[r_begin:r_end]
    #     seq_x_residue = self.data_x_residue[s_begin:s_end].unsqueeze(1)
    #     seq_y_residue = self.data_y_residue[r_begin:r_end].unsqueeze(1)
    #     original_x = self.original_data[s_begin:s_end]
    #     original_y = self.original_data[r_begin:r_end]

    #     return seq_x_imfs, seq_y_imfs, seq_x_residue, seq_y_residue, original_x, original_y

    # def __len__(self):
    #     return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len + self.lag  # Adjust for lag
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_imfs = self.data_x_imfs[s_begin:s_end]
        seq_y_imfs = self.data_y_imfs[r_begin:r_end]
        seq_x_residue = self.data_x_residue[s_begin:s_end].unsqueeze(1)
        seq_y_residue = self.data_y_residue[r_begin:r_end].unsqueeze(1)
        original_x = self.original_data[s_begin:s_end]
        original_y = self.original_data[r_begin:r_end]

        return seq_x_imfs, seq_y_imfs, seq_x_residue, seq_y_residue, original_x, original_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len - self.lag + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def save_all_imf_plots(self):
        # Generate and save plots for each subset
        self.plot_imfs(data_imfs=self.train_imfs, residue=self.train_residue, save_path=os.path.join(self.train_plot_dir, 'all_features_imfs.png'))
        self.plot_imfs(data_imfs=self.val_imfs, residue=self.val_residue, save_path=os.path.join(self.val_plot_dir, 'all_features_imfs.png'))
        self.plot_imfs(data_imfs=self.test_imfs, residue=self.test_residue, save_path=os.path.join(self.test_plot_dir, 'all_features_imfs.png'))
