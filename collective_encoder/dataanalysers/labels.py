import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .base import BaseDataAnalyser

class LabelsAnalyser(BaseDataAnalyser):
    """
    Data analyser for extracting and plotting label data.

    This analyser fetches properties designated as labels in datasets
    and can create 1D time-series style plots, 2D scatter plots,
    and correlation matrices based on provided configurations.
    """

    _IDENTIFIER = "LABELS"
    _COMPATIBLE_DATASET_TYPES = ["DISTANCES", "GRAPH"]
    _OPTIONAL_ARGS = {
        'labels_list': None,
        'extra_2d': [],
        'correlation': [],
    }
    
    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        if self.labels_list is None:
            self.log_warn("No labels_list provided. "
                          "Taking all labels from dataset. "
                          "This may result in a large number of plots.")
            self.labels_list = {f"{name}": i for i, name in enumerate(self.datamodule_labels_list)}

    def write_data(self, data, label=""):
        """
        Extract labels from data and write the visual analysis to disk.

        Parameters
        ----------
        data : list
            Data points from the dataset.
        label : str, optional
            Suffix added to output file names (e.g. 'train', 'val'). Default is empty string.
        """
        self.log_msg(f"Writing data analysis to {self.output_dir}")
        self._plot_labels(data, label)
    
    def _plot_axes_modifier(self, ax, yonly=True):
        """
        Applies modifiers to matplotlib axes. Meant to be overridden by subclasses.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        yonly : bool, optional
            Whether to apply modifications only to the y-axis, by default True.
        """
        pass
    
    def _get_label(self, datapoint):
        """
        Retrieve the label part of a datapoint based on dataset type.

        Parameters
        ----------
        datapoint : Any
            The input data point.

        Returns
        -------
        Any
            The extracted label object.
        """
        if self.ds_type == "GRAPH":
            return datapoint.y
        elif self.ds_type == "DISTANCES":
            return datapoint[1]
        else:
            self.raise_error(f"Unknown dataset type '{self.ds_type}'. ")
        
    def _extract_labels(self, data):
        """
        Extract specific labeled data into a dictionary structure.

        Parameters
        ----------
        data : list
            Data points from the dataset.

        Returns
        -------
        dict
            Dictionary mapping label names to numpy arrays of their values.
        """
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

    def _plot_labels(self, data, label=""):
        """
        Plots standard 1D scatter sequences for the extracted labels.

        Parameters
        ----------
        data : list
            Data points from the dataset.
        label : str, optional
            Suffix added to output file names. Default is empty string.
        """
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
        
        fig, ax = plt.subplots(len(labels), 1, figsize=(5, 4 + 2*(len(labels))))
        for i, k in enumerate(labels.keys()):
            ax[i].scatter(range(len(labels[k])), labels[k], marker='+', s=5, c=colors)
            ax[i].set_ylabel(f"{k}")
            self._plot_axes_modifier(ax[i], k)
        fig.savefig(self.output_dir + f"/dihedral_{label}.png", dpi=300)
        plt.close(fig)
        
        self._plot_extra_2d(labels, label, colors)
        self._plot_correlation(labels, label)
    
    def _plot_extra_2d(self, labels, label="", colors='blue'):
        """
        Creates 2D scatter plots for configured pairs of labels.
        
        Parameters
        ----------
        labels : dict
            Dictionary mapping label names to their extracted values.
        label : str, optional
            Suffix added to output file names.
        colors : str or list, optional
            Colors for the scatter points.
        """
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
    
    def _plot_correlation(self, labels, label=""):
        """
        Generates and saves correlation matrices for configured label combinations.
        
        Parameters
        ----------
        labels : dict
            Dictionary mapping label names to their extracted values.
        label : str, optional
            Suffix added to output file names.
        """
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
