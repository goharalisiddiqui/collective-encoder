import torch

import numpy as np
import pandas as pd

import scipy
from scipy.special import comb

import matplotlib.pyplot as plt

def combinations(n, r):
    # Generate all combinations of n items taken r at a time
    pool = np.arange(n)
    indices = np.arange(r)
    yield tuple(int(pool[i]) for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(int(pool[i]) for i in indices)
'''
def print_fve(self, datamodule):
        dl = datamodule.test_dataloader()
        flag = self.training
        self.training = False
        with torch.no_grad():
            data = next(iter(dl))[0].float()
            output, _ = self(data)
            target = self.normalize(data)
            sub = torch.sub(target, output)
            ss_err = torch.sum(torch.pow(sub, 2), dim=0)
            meann = torch.mean(target, dim=0, keepdim=True)
            sub_meann = torch.sub(target, meann)
            ss_tot = torch.sum(torch.pow(sub_meann, 2), dim=0)
            fve = 1 - torch.div(ss_err, ss_tot)
            fve_mean = torch.mean(fve).detach().cpu().numpy() # This calculates FVE for each input dim and mean it

            ss_err = torch.sum(torch.pow(sub, 2))
            ss_tot = torch.sum(torch.pow(sub_meann, 2))
            fve_sum = 1 - torch.div(ss_err, ss_tot).detach().cpu().numpy() # This calculates one FVE by taking inner product of vectors instead of square
        print("\n\n=======================================")
        print("Fraction of Variation Explined (FVE)")
        print("=======================================")
        # print("FVE_mean = ", fve_mean)
        print("FVE = ", fve_sum)
        print("=======================================\n\n")
        self.training = flag
        return fve_mean

def print_labels_latent_correlations(self, latent, labels = None):
    # Calculates and prints correlation between labels+latent_space
    if labels is None:
        data_df = pd.DataFrame(latent, columns=["Latent Dimension %d"%i for i in range(latent.shape[1])])
    else:
        all_data = np.concatenate((latent, labels), axis=1)
        all_column_headers = ["Latent Dimension %d"%i for i in range(latent.shape[1])] 
        all_column_headers = all_column_headers + self.trainer.datamodule.label_list
        data_df = pd.DataFrame(all_data, columns=all_column_headers)
    print("\n\n=======================================")
    print("Correlation of latent space and labels (if present)")
    print("=======================================")
    print(data_df.corr())
    print("=======================================\n\n")

    fig = plt.figure(figsize=(19, 15))
    plt.matshow(data_df.corr().abs(), fignum=fig.number)
    plt.xticks(range(data_df.columns.shape[0]), data_df.columns.tolist(), fontsize=14, rotation=45)
    plt.yticks(range(data_df.columns.shape[0]), data_df.columns.tolist(), fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    self.log_tbimage("Correlation", fig)

'''



@torch.no_grad()
def plot_2dscatter(x, y, labels = None, tag = None):
    assert x.shape == y.shape, "x, y must have the same shape"
    assert len(x.shape) == 1, "x, y must be 1D arrays"
    if labels is None or len(labels) == 0:
        raise ValueError("Labels must be provided for 2D scatter plot")
    ncols = len(labels)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8 * ncols, 6))
    if ncols == 1:
        axes = [axes]
    for ind, ax in enumerate(axes):
        label_name = list(labels.keys())[ind]
        label = labels[label_name]
        if len(label.shape) != 1 or label.shape[0] != x.shape[0]:
            raise ValueError(f"Label {label_name} must be a 1D array with the same length as x and y")
        scatter = ax.scatter(x, y, c=label, cmap='viridis', alpha=0.7)
        ax.set_xlabel(f"$LD_{tag.split('_')[0]}$")
        ax.set_ylabel(f"$LD_{tag.split('_')[1]}$")
        fig.colorbar(scatter, ax=ax, label=label_name)
    plt.tight_layout()
    return fig

@torch.no_grad()
def plot_2dline(x, labels = None):
    assert len(x.shape) == 1, "x must be a 1D array"
    if labels is None or len(labels) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, marker='o', linestyle='-', markersize=4)
        ax.set_xlabel("index")
        ax.set_ylabel("LD")
    else:
        ncols = len(labels)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8, 6))
        if ncols == 1:
            axes = [axes]
        for ind, ax in enumerate(axes):
            label_name = list(labels.keys())[ind]
            label = labels[label_name]
            if len(label.shape) != 1 or label.shape[0] != x.shape[0]:
                raise ValueError(f"Label {label_name} must be a 1D array with the same length as x")
            ax.scatter(x, label)
            ax.set_xlabel(f"$LD$")
            ax.set_ylabel(label)
    plt.tight_layout()
    return fig



def LDplotter(data, latent, pred, labels, meta, logger, outstem="./untitled_"):
    # print("[Plotting latent space with LDplotter]")
    # print("Latent shape: ", latent.shape)
    # print("Data shape: ", data.shape)
    # print("Pred shape: ", pred.shape)
    # for k, v in labels.items():
    #     print(f"Label {k}: {v.shape}")
    # for k, v in meta.items():
    #     print(f"{k}: {v.shape}")
    mu_latent = meta.get('mu_latent', None)
    if mu_latent is not None:
        nld = mu_latent.shape[1]
        if nld == 1:
            fig = plot_2dline(mu_latent[:, 0], labels=labels, tag="LDplotter")
            # logger.add_figure("LDplotter/mu_latent", fig)
            fig.savefig(f"{outstem}LDplotter_mu_latent.png", dpi=150)
        elif nld == 2:
            fig = plot_2dscatter(mu_latent[:, 0], mu_latent[:, 1], labels=labels, tag="0_1")
            # logger.add_figure("LDplotter/mu_latent", fig)
            fig.savefig(f"{outstem}LDplotter_mu_latent.png", dpi=150)
        else:
            combs = combinations(nld, 2)
            for (i, j) in combs:
                fig = plot_2dscatter(mu_latent[:, i], mu_latent[:, j], labels=labels, tag=f"{i}_{j}")
                # logger.add_figure(f"LDplotter/mu_latent_{i}_{j}", fig)
                fig.savefig(f"{outstem}LDplotter_mu_latent_{i}_{j}.png", dpi=150)

    # plot_latent(data, latent, labels = meta.get('labels', None), plot_func = None, tag = "LDplotter")
    return  