import numpy as np

import torch
import torch.nn as nn
# from torch_geometric.nn.conv import NNConv
# from torch_geometric.nn import global_mean_pool, global_max_pool
# from torch_geometric.data import Batch

import torch.nn.functional as F
import pytorch_lightning as pl


from torch.autograd import Variable
torch.manual_seed(0)




import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc


class LITcollAE(pl.LightningModule):
    def __init__(self, l:list, lr : float = 0.01, l2_reg : float = 1e-7, 
                 outname : str = './LITcollAE_untitled/LITcollAE_'):
        super().__init__()
        assert len(l) >= 2
        print("[Initializing LITcollAE Module]")
        print("- hidden layers:", l)
        print("")
        print("========= NN =========")
        modules = []
        for i in range(len(l) - 1):
            print(l[i], " --> ", l[i + 1], end=" ")
            if i < len(l) - 2:
                modules.append(nn.Linear(l[i], l[i + 1]))
                modules.append(nn.ReLU(True))
                modules.append(nn.BatchNorm1d(l[i + 1]))
                print("(relu)")
            else:
                modules.append(nn.Linear(l[i], l[i + 1]))
                print("")
        modules.append(nn.Sigmoid())
        print("(sigmoid)")
        self.encoder = nn.Sequential(*modules)
        modules = []
        a = len(l) - 1
        for i in range(len(l) - 1):
            print(l[a - i], " --> ", l[a - i - 1], end=" ")
            if i < len(l) - 2:
                modules.append(nn.Linear(l[a - i], l[a - i - 1]))
                modules.append(nn.ReLU(True))
                modules.append(nn.BatchNorm1d(l[a - i - 1]))
                print("(relu)")
            else:
                modules.append(nn.Linear(l[a - i], l[a - i - 1]))
                print("")
        self.decoder = nn.Sequential(*modules)
        print("======================")
        
        # Model meta info
        self.normIn = False
        self.metaD = False
        self.save_hyperparameters()
        
        self.train_loss_list = []
        self.val_loss_list = []
        self.print_loss = 1
        
    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        self.normIn = True
        self.register_buffer('Mean', Mean)
        self.register_buffer('Range', Range)

    def normalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        # print(f"\n\nmean shape={self.Mean.shape}\n\n")
        # print(f"\n\nmean shape={x.shape}\n\n")
        
        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean).div(Range)
    
    def denormalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.mul(Range).add(Mean)

    def encode(self, x: Variable) -> (Variable):
        if self.normIn:
            x = self.normalize(x)
        z = self.encoder(x)
        return z

    def decode(self, x: Variable) -> (Variable):
        z = self.decoder(x)
        return z

    def forward(self, x: Variable) -> (Variable):
        z = self.encode(x)
        if self.metaD:
            return z
        y = self.decode(z)
        
        return y

    def loss_fn(self, output, target):
        return F.mse_loss(output, target)
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                    factor=0.7, patience=10,
    #                                                    min_lr=0.0000001)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_error",
    #             "frequency": 1,
    #         }
    #     }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay= self.hparams.l2_reg)
        return optimizer
    
    def on_train_start(self):
        print("\n\n==================================")
        print("Starting training LITcollAE module")
        print("==================================")
        print("[Optimization Settings]")
        print("- Learning rate \t=", self.hparams.lr)
        print("- l2 regularization \t=", self.hparams.l2_reg)
        print("==================================\n\n")
        # print("[{:>3}/{:>3}] {:>10}".format("ep", "tot", "train_loss",  "val_loss"))
    
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch[0].float()
        result = self(data)
        target = self.normalize(data)
        
        loss = self.loss_fn(result, target)
        self.train_loss_list.append(loss)
        # self.log('train_loss', loss.item(), prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch[0].float()
        result = self(data)
        target = self.normalize(data)
        loss = self.loss_fn(result, target)
        self.val_loss_list.append(loss)
        self.log('val_loss', loss.item(), prog_bar=True)
        return loss

    # def on_validation_epoch_end(self):
    #     if (len(train_loss_list)) % self.print_loss == 0:
    #         print("[{:3d}/{:3d}] {:10.3f} {:10.3f}".format(len(self.train_loss_list),self.max_epochs, train_loss_list[-1], val_loss_list[-1]))
        

    def test_step(self, test_batch, batch_idx):
        epoch = self.current_epoch
        n_hidden = self.hparams.l[-1]

        train_x, train_y = next(iter(test_batch))
        train_x, train_y = train_x.float(), train_y.float()
        n_labels = train_y.shape[-1]
        fig, axes = plt.subplots(n_hidden if n_hidden > 2 else 1, n_labels + 1, squeeze=False,figsize=(13, 5))
        
        self.plot_training(axes[0][0], range(1, len(self.train_loss_list)), self.train_loss_list)
        for i in range(0, axes.shape[0]):
            for j in range(n_labels):
                self.plot_latent(fig, axes[i][j+1], train_x, train_y[:,j], i)
        
        plt.tight_layout()
        fig.savefig(f"{self.hparams.outname}{epoch}_training.png", dpi=150)
        plt.close()
    
        return None


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

    def plot_latent(fig, ax, train_x, train_y, val_x, val_y, model, i):
        ax.set_title("LITcollAE Latent-space-"+str(i))
        
        latent_train = model.encode(train_x).cpu().detach().numpy()
        
        cm = plt.get_cmap('jet')
        cNorm = matplotlib.colors.Normalize(vmin=min(latent_train), vmax=max(latent_train))
        
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        yaxis = (i+1) if (i+1) < latent_train.shape[1] else 0
        ax.scatter(latent_train[:, i], latent_train[:, yaxis], c=scalarMap.to_rgba(train_y), label="Test Set", alpha=0.3)
        
        ax.set_xlabel("h_{}".format(i))
        ax.set_ylabel("h_{}".format(yaxis))

        scalarMap.set_array(train_y)
        cb = fig.colorbar(scalarMap, ax=ax)
        # cb.set_label(label)
        ax.legend()