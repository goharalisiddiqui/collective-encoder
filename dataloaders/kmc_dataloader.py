import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import argparse

class KmcData(Dataset):
    """KMC dataset"""

    def __init__(self, colvar_list):
        self.nstates = len(colvar_list)
        self.colvar = colvar_list

    def __len__(self):
        return len(self.colvar[0])

    def __getitem__(self, idx):
        x = ()
        for i in range(self.nstates):
            x += (self.colvar[i][idx],)
        return x


def kmcdatset_args():
    desc = "KMC Dataset Arguments"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--framefile', type=str, required=True, help='KMC frames file')
    parser.add_argument('--labelfile', type=str, default=None, help='Label file')
    parser.add_argument('--labelnames', type=str, nargs='+', default=None, help='Names for lables in the labelfile')


    args, _ = parser.parse_known_args()

    return args


KMC_args = kmcdatset_args()

class KmcDataset(pl.LightningDataModule):
    def __init__(self,
                 framefile : str,
                 labelfile : str,
                 labelnames : list,
                 train_prop : int = 0.8,
                 batch_size : int = -1,
                 batch_prop : float = 0.1,
                 standardize_inputs : bool = True):
        super().__init__()
        print("\n\n[Initializing KmcDataset Module]")
        print("==========================================")
        if framefile is None:
            raise ValueError("Frame file not provided")
        print(f"Loading frames data from file: {framefile}")
        if not os.path.exists(framefile):
            raise FileNotFoundError(f"File {framefile} not found")
        frames = np.load(framefile)
        print(f"Loaded {frames.shape[0]} frames data with shape: {frames.shape[1:]}")
        if labelfile is not None:
            print(f"Loading label data from file: {labelfile}  ...")
            if not os.path.exists(labelfile):
                raise FileNotFoundError(f"File {labelfile} not found")
            label = np.load(labelfile)
            assert label.shape[0] == frames.shape[0], f"Label data shape {label.shape} does not match frames data shape {frames.shape}"
            assert len(label.shape) == 2, "Label data must be 1D"
            print(f"Loaded label data from file: {labelfile}")
        self.label_list = []
        no_of_labels = label.shape[1]
        for i in range(no_of_labels):
            if labelnames is not None and len(labelnames) > i:
                self.label_list.append(labelnames[i])
            else:
                self.label_list.append(f"label-{i+1}")

        ## Dividing data into training and validation data
        self.alldata = frames.reshape(frames.shape[0], -1)
        self.alllabel = label
        self.num_inputs = self.alldata.shape[1]
        FRAMES = frames.shape[0]
        self.n_train_data = int(FRAMES * train_prop)
        self.n_valid_data = int(FRAMES * (1.0 - train_prop))
        if batch_size == -1:
            self.train_batchsize = int(self.n_train_data * batch_prop)
        else:
            self.train_batchsize = batch_size

        # Printing
        print("[Imported data]")
        print("- data.shape:", self.alldata.shape)
        print("- label.shape:", self.alllabel.shape)
        print("- labels:", self.label_list)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(self.alldata)

        print(f"Total frames: {FRAMES}, Train size: {self.n_train_data}, Batch size: {self.train_batchsize}, Validation size: {self.n_valid_data}")
        print("==========================================")
        self.save_hyperparameters()
        p = np.random.permutation(len(self.alldata))
        self.all_dataset = KmcData([self.alldata[p], self.alllabel[p]])





    # def prepare_data(self): # only called on 1 GPU/TPU in distributed


    def setup(self, stage): # Called on every GPU/TPU in distributed
        # Assign train/val datasets for use in dataloaders
        self.training_data, self.validation_data = train_test_split(self.all_dataset, train_size=self.hparams.train_prop, random_state=1868)


        # called on every process in DDP
    def train_dataloader(self):
        train_data = self.target_scaler.transform(self.training_data) if self.hparams.standardize_inputs else self.training_data
        return DataLoader(train_data, batch_size=self.train_batchsize, shuffle=True, drop_last=True, num_workers=1)

    def val_dataloader(self):
        valid_data = self.target_scaler.transform(self.validation_data) if self.hparams.standardize_inputs else self.validation_data
        return DataLoader(valid_data, batch_size=len(self.validation_data), shuffle=False, drop_last=True, num_workers=1)

    def test_dataloader(self):
        all_dataset = KmcData([self.alldata, self.alllabel])
        return DataLoader(all_dataset, batch_size=len(all_dataset), shuffle=False, drop_last=False)

    def target_scaler(self, X):
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        return self.target_scaler.mean_

    def get_scaler_var(self):
        return self.target_scaler.var_

    def get_scaler_scale(self):
        return self.target_scaler.scale_