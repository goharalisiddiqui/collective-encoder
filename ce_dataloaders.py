import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class ColvarDataset(Dataset):
    """COLVAR dataset"""

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



class LITColvarData(pl.LightningDataModule):
    def __init__(self, colvar_file, train_prop : int = 0.6, batch_size : int = -1,
                 batch_prop : float = 0.1,
                 label_list : list = [],
                 standardize_inputs : bool = True):
        super().__init__()
        print("\n\n[Initializing LITColvarData Module]")
        print("==========================================")
        print(f"Loading data from file: {colvar_file}")
        with open(colvar_file) as f:
            first_line = f.readline().split()
        assert all([x in first_line for x in label_list]) # Asserts that all the print labels are present in the data file

        ignore_list = ["#!", "FIELDS", "time"]
        header = [x for x in first_line if x not in ignore_list and x not in label_list]
        first_col = len(ignore_list) + len(label_list) - 2  # 0-indexed
        self.header_string = ",".join(header)
        self.num_inputs = len(header)



        col_range = range(first_col, first_col + self.num_inputs)

        header_dist = [x for x in header if "distvec" in x]
        header_solv = [x for x in header if "VV" in x]

        if False: # Use only distance vectors
            print("[WARNING: Using only distvec from data file. Ignoring solvation descriptors]")
            self.num_inputs = len(header_dist)
            col_range = range(first_col, first_col + self.num_inputs)
        if False: # Use only solvation
            print("[WARNING: Using only VV from data file. Ignoring distances]")
            self.num_inputs = len(header_solv)
            col_range = range(first_col + len(header_dist), first_col + len(header_dist) + self.num_inputs)

        alldata = np.loadtxt(colvar_file, usecols=col_range)
        alllabel = np.loadtxt(colvar_file, usecols=[a for a in range(1,len(label_list) + 1)], ndmin = 2)




        p = np.random.permutation(len(alldata))
        alldata, alllabel = alldata[p], alllabel[p]

        ## Dividing data into training and validation data
        FRAMES = alldata.shape[0]
        self.n_train_data = int(FRAMES * train_prop)
        self.n_valid_data = int(FRAMES * (1.0 - train_prop))
        if batch_size == -1:
            self.train_batchsize = int(self.n_train_data * batch_prop)
        else:
            self.train_batchsize = batch_size

        # Printing
        print("[Imported data]")
        print("- data.shape:", alldata.shape)
        print("- label.shape:", alllabel.shape)
        assert alldata.shape[0] == alllabel.shape[0]
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(alldata)

        print(f"Total frames: {FRAMES}, Train size: {self.n_train_data}, Batch size: {self.train_batchsize}, Validation size: {self.n_valid_data}")
        print("==========================================")
        self.save_hyperparameters()
        self.all_dataset = ColvarDataset([alldata, alllabel])

        # print(f"\nshape={alllabel.shape}\n")
        # print(f"\nmax={np.amax(alllabel)}\n")
        # print(f"\nmin={np.amin(alllabel)}\n")
        # exit()




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
        test_data = self.target_scaler.transform(self.all_dataset) if self.hparams.standardize_inputs else self.all_dataset
        return DataLoader(test_data, batch_size=len(self.all_dataset), shuffle=False, drop_last=True)

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