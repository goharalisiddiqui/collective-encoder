##################################
# Plotting functions
##################################

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rc
#plt.rcParams["font.family"] = "Times New Roman"
outname = "plot"
device = "cpu"

##################################
# Importing Encoding Functions
##################################
import AE_encode
from AE_encode import encode_latent
#from AE_encode import encode_cv
#from AE_encode import encode_cv_all


def plot_results(epochs, model, loss_list, train_all_loader, valid_loader, label_header, save=False):
    epoch = epochs[-1]
    n_hidden = AE_encode.n_hidden

    train_x, train_y = next(iter(train_all_loader))
    val_x, val_y = next(iter(valid_loader))

    train_x, train_y, val_x, val_y = train_x.float(), train_y.float(), val_x.float(), val_y.float() 

    n_labels = train_y.shape[-1]
    # print(f"nlabels={n_labels}")
    
    fig, axes = plt.subplots(n_hidden if n_hidden > 2 else 1, n_labels + 1, squeeze=False,figsize=(13, 5))
    plot_training(axes[0][0], epochs, loss_list)
    for i in range(0, axes.shape[0]):
        for j in range(n_labels):
            plot_H(fig, axes[i][j+1], train_x, train_y[:,j], val_x, val_y[:,j], model, label_header[j], i)
    plt.tight_layout()
    if save:
        fig.savefig(f"{outname}{epoch}_training.png", dpi=150)
        plt.close()


def plot_training(ax, epochs, loss_list):
    ax.set_title("Network Loss minimization")
    ax.set_yscale("log")
    ax.plot(
        np.asarray(epochs),
        np.asarray([x.cpu().detach().numpy() for x in loss_list]),
        ".-",
        c="tab:green",
        label="loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.legend()

def plot_H(fig, ax, train_x, train_y, val_x, val_y, model, label, i):
    ax.set_title("AE Latent-space-"+str(i))
    
    latent_train = model.encode(train_x.to(device)).cpu().detach().numpy()
    latent_val = model.encode(val_x.to(device)).cpu().detach().numpy()
    
    cm = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=min(np.concatenate((train_y,val_y))), vmax=max(np.concatenate((train_y,val_y))))
    
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    yaxis = (i+1) if (i+1) < latent_train.shape[1] else 0
    ax.scatter(latent_train[:, i], latent_train[:, yaxis], c=scalarMap.to_rgba(train_y), label="train set", alpha=0.3)
    ax.scatter(latent_val[:, i], latent_val[:, yaxis], c=scalarMap.to_rgba(val_y), label="valid set", marker="+", alpha=0.5)
    
    #mIN = np.min([np.min(trA[:, 0]), np.min(trB[:, 0]), np.min(ttA[:, 0]), np.min(ttB[:, 0])])
    #mAX = np.max([np.max(trA[:, 0]), np.max(trB[:, 0]), np.max(ttA[:, 0]), np.max(ttB[:, 0])])

    ax.set_xlabel("h_{}".format(i))
    ax.set_ylabel("h_{}".format(yaxis))

    #x = np.linspace(mIN,mAX,100)
    #y = -eigen[0]/eigen[1] * x
    #ax.plot(x,y, linewidth=2, label='Separation')
    scalarMap.set_array(np.concatenate((train_y,val_y)))
    cb = fig.colorbar(scalarMap, ax=ax)
    cb.set_label(label)
    ax.legend()
