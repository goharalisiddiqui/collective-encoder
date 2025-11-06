import os
import numpy as np
import torch
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

class Ala2DataAnalyser():
    def __init__(self, output_dir, data_args = {}):
        self.output_dir = output_dir
        self.data_args = data_args
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_dihedrals(self, data):
        idx_phi = 6 # [1,3,4,5]
        idx_psi = 10 # [3,4,6,8]

        sequence_len = self.data_args.get("sequence_len", 1)
        sin_phi, cos_phi, sin_psi, cos_psi = [], [], [], []
        for d in tqdm(data, desc="Processing dihedrals"):
            sin_phi.append(d.y_torsions_sin[idx_phi].cpu().numpy())
            cos_phi.append(d.y_torsions_cos[idx_phi].cpu().numpy())
            sin_psi.append(d.y_torsions_sin[idx_psi].cpu().numpy())
            cos_psi.append(d.y_torsions_cos[idx_psi].cpu().numpy())

        phi = np.arctan2(sin_phi, cos_phi)
        psi = np.arctan2(sin_psi, cos_psi)

        # Make sequence length alternate color
        colors = ['red'] * sequence_len + ['blue'] * sequence_len
        colors = colors * (len(phi) // (2 * sequence_len) + 1)
        colors = colors[:len(phi)]

        fig, ax = plt.subplots(2, 1, figsize=(7,5))
        ax[0].scatter(range(len(phi)), phi, marker='+', s=5, color=colors)
        ax[0].set_ylabel(r"$\Phi$")
        ax[0].set_ylim([-np.pi, np.pi])
        ax[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax[0].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])


        ax[1].scatter(range(len(psi)), psi, marker='+', s=5, color=colors)
        ax[1].set_ylabel(r"$\Psi$")
        ax[1].set_xlabel("Frame")
        ax[1].set_ylim([-np.pi, np.pi])
        ax[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        fig.savefig(self.output_dir + "/dihedrals.png", dpi=300)
        print(f"\n[{type(self).__name__}]: Saved dihedral plot to {self.output_dir}/dihedrals.png")

    def write_data(self, data):
        print(f"\n[{type(self).__name__}]: Writing data analysis to {self.output_dir}")
        print("="*80)
        self.plot_dihedrals(data)
        print("="*80)
