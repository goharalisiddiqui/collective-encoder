import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .base import BaseDataAnalyser

class LabelsAnalyser(BaseDataAnalyser):
    """
    Data analyser for extracting and plotting dihedral angles.
    """

    _IDENTIFIER = "LABELS"
    _COMPATIBLE_DATASET_TYPES = ["DISTANCES", "GRAPH"]
    _REQUIRED_ARGS = BaseDataAnalyser._REQUIRED_ARGS + [
        'labels_list',
    ]
    _OPTIONAL_ARGS = {
        'extra_2d': [],
        'correlation': [],
    }

    def write_data(self, data, label = ""):
        self.log_msg(f"Writing data analysis to {self.output_dir}")
        self._plot_labels(data, label)
    
    def _plot_axes_modifier(self, ax, yonly=True):
        pass
    
    def _get_label(self, datapoint):
        if self.ds_type == "GRAPH":
            return datapoint.y
        elif self.ds_type == "DISTANCES":
            return datapoint[1]
        else:
            self.raise_error(f"Unknown dataset type '{self.ds_type}'. ")
        
    def _extract_labels(self, data):
        labels = {}
        max_l = self._get_label(data[0]).shape[0]
        for label, idx in self.labels_list.items():
            if idx < max_l:
                labels[label] = []
            else:
                self.log_warn(f"Label '{label}' index {idx} exceeds data length {max_l}. Skipping.")
        if len(labels) == 0:
            self.raise_error("No valid labels found in labels_list. "
                "Check that the dataset is correctly configured for label extraction.")

        for d in tqdm(data, desc="Extracting labels"):
            for key in labels.keys():
                idx = self.labels_list[key]
                labels[key].append(self._get_label(d)[idx].cpu().numpy())
        for key in labels.keys():
            labels[key] = np.array(labels[key])
        return labels

    def _plot_labels(self, data, label = ""):
        if len(data) == 0:
            self.log_warn("No data to plot dihedrals.")
            return
        labels = self._extract_labels(data)
        
        # Make sequence length alternate color
        if 'input_chunk_length' in self.datamodule_args and \
                                'output_chunk_length' in self.datamodule_args:
            input_chunk_length = self.datamodule_args['input_chunk_length']
            output_chunk_length = self.datamodule_args['output_chunk_length']
            n_seq_per_sample = self.datamodule_args.get('n_seq_per_sample', 1)
            sequence_len = input_chunk_length + \
                                n_seq_per_sample * output_chunk_length

            colors = ['red'] * sequence_len + ['blue'] * sequence_len
            colors = colors * (len(labels[list(labels.keys())[0]]) // (2 * sequence_len) + 1)
            colors = colors[:len(labels[list(labels.keys())[0]])]
        else:
            colors = 'blue'
        
        fig, ax = plt.subplots(len(labels), 1, figsize=(4 + 2*(len(labels)),5))
        for i, k in enumerate(labels.keys()):
            ax[i].scatter(range(len(labels[k])), labels[k], marker='+', s=5, c=colors)
            ax[i].set_ylabel(f"{k}")
            self._plot_axes_modifier(ax[i], k)
        fig.savefig(self.output_dir + f"/dihedral_{label}.png", dpi=300)
        plt.close(fig)
        
        self._plot_extra_2d(labels, label, colors)
        self._plot_correlation(labels, label)
    
    def _plot_extra_2d(self, labels, label = "", colors = 'blue'):
        
        for sel in self.extra_2d:
            label_x, label_y = sel.split(':')
            if label_x not in labels:
                self.log_warn(f"Label '{label_x}' not found in labels. Skipping 2D plot for '{sel}'.")
                continue
            if label_y not in labels:
                self.log_warn(f"Label '{label_y}' not found in labels. Skipping 2D plot for '{sel}'.")
                continue

            fig, ax = plt.subplots(1, 1, figsize=(5,5))
            ax.scatter(labels[label_x], 
                       labels[label_y], marker='+', s=5, c=colors)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            self._plot_axes_modifier(ax, yonly=False)
            fig.savefig(self.output_dir + f"/2d_{sel.replace(':', '_')}_{label}.png", dpi=300)
            plt.close(fig)
    
    def _plot_correlation(self, labels, label = ""):
        
        for sel in self.correlation:
            fields = sel.split(':')
            if len(fields) < 2:
                self.log_warn(f"Correlation selection '{sel}' does not have two fields. Skipping.")
                continue
            for field in fields:
                if field not in labels:
                    self.log_warn(f"Label '{field}' not found in labels. Skipping correlation plot for '{sel}'.")
                    continue
            l = len(labels[fields[0]])
            for f in fields[1:]:
                if len(labels[f]) != l:
                    self.log_warn(f"Label '{f}' has a different length than '{fields[0]}'. Skipping correlation plot for '{sel}'.")
                    continue
            

            fig, ax = plt.subplots(1, 1, figsize=(5,5))
            correlation = np.corrcoef([labels[f] for f in fields])
            ax.matshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
            for (i, j), val in np.ndenumerate(correlation):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if abs(val) > 0.5 else 'black')
            ax.set_xticks(range(len(fields)))
            ax.set_xticklabels(fields, rotation=45)
            ax.set_yticks(range(len(fields)))
            ax.set_yticklabels(fields)
            fig.savefig(self.output_dir + f"/correlation_{sel.replace(':', '_')}_{label}.png", dpi=300)
            plt.close(fig)
