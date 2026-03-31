import os
import argparse
import torch
import torch.nn as nn
import torch.jit
from torch_geometric import datasets
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict

import ase
import ase.build
from sklearn.utils import shuffle

import numpy as np

import pytorch_lightning as pl

DOUBLE_PRECISION = False


def md17_args():
    desc = "Colvar Dataset Arguments"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--datasets', nargs='+', required=True, help='Input file for training')

    args, _ = parser.parse_known_args()

    return args


MD17_args = md17_args

class MDDataset(Dataset):
    def __init__(
        self,
        structures: List[ase.Atoms],
        energies,
        offset_energy : bool = False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dists = [torch.tensor(self.get_dist(s.positions)) for s in structures]
        self.num_inputs = len(self.dists[0])
        self.base_energies = torch.tensor(energies)
        if offset_energy:
            self.offset = torch.mean(self.base_energies)
        else:
            self.offset = 0.0
        self.energies = (self.base_energies - self.offset).to(dtype)
        self.energies = self.energies.view(-1, 1)

    def get_dist(self, positions):
        _ , distmat = ase.geometry.get_distances(positions)
        distmat = np.triu(distmat)
        dist = distmat[distmat != 0]
        return dist

    def __getitem__(self, index):
        x = ()
        x += (self.dists[index],)
        x += (self.energies[index],)
        return x

    def __len__(self):
        return len(self.energies)


class MD17Data(pl.LightningDataModule):
    def __init__(self, data_dir : str ='./md17_data',
                 datasets : list = ["revised benzene"],
                 train_size : int = 1000,
                 validation_size : int = 500,
                 test_size : int = 3000,
                 batch_size : int = 32,
                 val_batch_size : int = 64,
                 test_batch_size : int = 3000,
                 predict_batch_size : int = 1,
                 num_workers : int = 1,
                 offset_energy : bool = False,
                 ):
        super().__init__()
        self.num_inputs = 1
        self.label_list = ["energies"]
        self.save_hyperparameters()
    def prepare_data(self):
        for dataset_name in self.hparams.datasets:
            datasets.MD17(self.hparams.data_dir, dataset_name)
        # only called on 1 GPU/TPU in distributed
    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            iend = self.hparams.train_size + self.hparams.validation_size + self.hparams.test_size + 1
            all_structures = []
            all_energies = []
            for i, name in enumerate(self.hparams.datasets):
                filename = datasets.MD17(self.hparams.data_dir, name).raw_file_names
                md17_base = np.load(os.path.join(self.hparams.data_dir, f"raw/{filename}"))
                structures = [
                    ase.Atoms(numbers=md17_base["nuclear_charges"], positions=cs)
                    for cs in md17_base["coords"][:iend]
                ]
                all_structures.extend(structures)
                all_energies.extend(md17_base["energies"][:iend])
                md17_base.close()
            all_structures, all_energies = shuffle(all_structures, all_energies, random_state=42)

            mddata_full = MDDataset(
                structures=all_structures,
                energies=all_energies,
                offset_energy=self.hparams.offset_energy,
                dtype=torch.float64 if DOUBLE_PRECISION else torch.float32)
            self.num_inputs = mddata_full.num_inputs

            self.mddata_train, self.mddata_val, self.mddata_test, _ = \
                torch.utils.data.random_split(
                    mddata_full,
                    [
                        self.hparams.train_size,
                        self.hparams.validation_size,
                        self.hparams.test_size,
                        len(mddata_full) - self.hparams.train_size - self.hparams.validation_size - self.hparams.test_size
                    ])

        # called on every process in DDP
    def train_dataloader(self):
        return DataLoader(
            self.mddata_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.mddata_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            self.mddata_test,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(
            self.mddata_test,
            batch_size=self.hparams.predict_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def get_sample_point(self):
        data_point = self.mddata_test[0]
        return data_point

