import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .base import BaseDataAnalyser

class Ala2DataAnalyser(BaseDataAnalyser):
    def plot_dihedrals(self, data, label = ""):
        idx_phi = 6 # [1,3,4,5]
        idx_psi = 10 # [3,4,6,8]
        
        sin_phi, cos_phi, sin_psi, cos_psi = [], [], [], []
        for d in tqdm(data, desc="Processing dihedrals"):
            sin_phi.append(d.y_torsions_sin[idx_phi].cpu().numpy())
            cos_phi.append(d.y_torsions_cos[idx_phi].cpu().numpy())
            sin_psi.append(d.y_torsions_sin[idx_psi].cpu().numpy())
            cos_psi.append(d.y_torsions_cos[idx_psi].cpu().numpy())

        phi = np.arctan2(sin_phi, cos_phi)
        psi = np.arctan2(sin_psi, cos_psi)
        
        # Make sequence length alternate color
        if all(key in self.data_args for key in ['input_chunk_length', 'output_chunk_length']):
            input_chunk_length = self.data_args['input_chunk_length']
            output_chunk_length = self.data_args['output_chunk_length']
            n_samples = self.data_args.get('n_seq_per_sample', 1)
            sequence_len = input_chunk_length + n_samples * output_chunk_length

            colors = ['red'] * sequence_len + ['blue'] * sequence_len
            colors = colors * (len(phi) // (2 * sequence_len) + 1)
            colors = colors[:len(phi)]
        else:
            colors = 'blue'

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
        fig.savefig(self.output_dir + f"/dihedrals_{label}.png", dpi=300)
        self.log_msg(f"Saved dihedral plot to {self.output_dir}/dihedrals_{label}.png")

        plt.close(fig)
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.scatter(phi, psi, marker='+', s=5, color=colors)
        ax.set_xlabel(r"$\Phi$")
        ax.set_ylabel(r"$\Psi$")
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        fig.savefig(self.output_dir + f"/ramachandran_{label}.png", dpi=300)
        self.log_msg(f"Saved Ramachandran plot to {self.output_dir}/ramachandran_{label}.png")
        plt.close(fig)
        

    def write_data(self, data, label = ""):
        self.log_msg(f"Writing data analysis to {self.output_dir}")
        self.plot_dihedrals(data, label)
        
