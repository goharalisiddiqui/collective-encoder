import numpy as np
from typing import Any, Tuple, Dict

import torch

from collective_encoder.nets.ae_base import AEBase
from collective_encoder.nets.modules.variational_nn import VariationalNN
from collective_encoder.losses.nll import CELossNLL
from collective_encoder.losses.kld_resolver import create_kld_loss
from collective_encoder.losses.bond_deviation import CELossBondDeviation
from collective_encoder.losses.steric import CELossSteric

EPSILON = 1e-7


class VAE(AEBase):
    _IDENTIFIER = "VAE"
    _COMPATIBLE_DATASETS = ["DEFAULT", "DISTANCES", "SOAP", "SOAP_PS"]
    _OPTIONAL_ARGS = AEBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "beta": 1.0,  # Weight for the KL divergence term in the loss function
        "kld_type": "gaussian",  # Prior distribution for the latent space (supports 'gaussian', ')
        "kld_args": None,  # Additional arguments for the KLD loss (e.g., number of components for Gaussian Mixture)
        "use_bond_deviation_loss": False,  # Whether to include a bond deviation loss based on bonded atom pairs
        "use_steric_loss": False,
        "use_bond_deviation_loss": False,
    })
    
    @staticmethod
    def extract_args_from_datamodule(datamodule, args) -> dict:
        args = AEBase.extract_args_from_datamodule(datamodule, args)
        if args.get("use_bond_deviation_loss", False) or args.get("use_steric_loss", False):
            args['atomic_numbers'] = datamodule.get_atns()
            if args.get("use_bond_deviation_loss", False):
                args['bond_indices'] = datamodule.get_bond_indices()
        return args

    def __init__(self,
                args: Dict[str, Any] = None,
                **kwargs
                ):
        self.save_hyperparameters()
        super().__init__(args=args, **kwargs)
        
        if self.kld_type == 'nflow':
            if self.kld_args is None:
                self.kld_args = {}
            self.kld_args['latent_dim'] = self.latent_dim

        self.losses = {
            "rec_loss": CELossNLL({}, **kwargs),
            "reg_loss": create_kld_loss(self.kld_type, self.kld_args, **kwargs),
        }
        if self.use_bond_deviation_loss:
            self.losses["bond_deviation_loss"] = CELossBondDeviation({
                "atomic_numbers": self.atomic_numbers,
                "bond_indices": self.bond_indices,
            }, **kwargs)
        if self.use_steric_loss:
            self.losses["steric_loss"] = CELossSteric({
                "atomic_numbers": self.atomic_numbers,
            }, **kwargs)

    def get_metad_output(self, latent: Tuple[torch.Tensor, torch.Tensor], meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        # For metaD we use only use the mean of the latent distribution
        mean, logvar = latent
        return mean

    def aggregate_losses(self, losses):
        loss = losses['rec_loss'] + self.beta * losses['reg_loss']
        return loss

    def print_hparams(self):
        super().print_hparams()
        self.ce_log_dict("VAE hparams:", self.args)

    def init_network(self):
        self.encoder_net = VariationalNN(layers=self.encoder_network, 
                                        batch_norm=self.batch_norm,
                                        activation=self.activation,
                                        activation_args=self.activation_args)
        self.decoder_net = VariationalNN(layers=self.decoder_network, 
                                        batch_norm=self.batch_norm,
                                        activation=self.activation,
                                        activation_args=self.activation_args)

    def encoder(self, x):
        mu, logvar = self.encoder_net(x)
        return (mu, logvar), {}

    def decoder(self, z):
        mu_x, logvar_x = self.decoder_net(z)
        x_out = self.reparametrize_multivariate(mu_x, logvar_x)
        return x_out, {"mu_x": mu_x, "logvar_x": logvar_x}

    def latent_to_decoder_input(self, latent: Tuple[torch.Tensor, torch.Tensor]):
        mu_latent, logvar_latent = latent
        z = self.reparametrize_multivariate(mu_latent, logvar_latent)
        return z, {"mu_latent": mu_latent, "logvar_latent": logvar_latent, "z_sample": z}

    def on_validation_epoch_end(self):
        self.losses['reg_loss'].on_validation_epoch_end(self) # FIXME: Find better way to call this, The KLDScheduler "Auto" need to know when the validation epoch ends
    
    def plot_avg_sigma(self, latent_logvar):
        ld_mean = np.mean(np.exp(0.5 * latent_logvar), axis=0)
        lines = ["Avg. Sigma per LD:"] + [f"  LD {i}: {ld_mean[i]}" for i in range(len(ld_mean))]
        self.log_msg("\n".join(lines))

    def get_latent(self, data_x):
        data_x = self.normalize(data_x)
        (latent_mu, latent_logvar), _ = self.encoder(data_x)
        return latent_mu.detach().cpu().numpy(), latent_logvar.detach().cpu().numpy()

    def get_latent_mean(self, data_x):
        return self.get_latent(data_x)[0]

    def get_latent_names(self):
        return "mu_latent", "logvar_latent"

    def get_metatomic_model(self):
        model = MetatomicModelVAE(
            encoder=self.encoder_net,
            normIn=self.normIn,
            dmean=self.Mean,
            drange=self.Range,
        )
        return model

# ------------------------------------------------------------------
# Symmetric Variational Autoencoder (sVAE)
# ------------------------------------------------------------------

class sVAE(VAE):
    _IDENTIFIER = "sVAE"
    
    """
    Symmetric Variational Autoencoder (sVAE) with symmetric encoder and decoder architectures.
    The encoder and decoder architectures are determined by the provided network
    """

    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs
                 ):
        self.save_hyperparameters()
        network = args.pop('network', None)
        if network is None:
            raise ValueError("Argument 'network' is required for sVAE")
        args['encoder_network'] = network
        args['decoder_network'] = network[:-1][::-1]
        super().__init__(args=args, **kwargs)

# ------------------------------------------------------------------
# Metatomic Interface
# ------------------------------------------------------------------

try:
    from metatomic.torch import ModelOutput

    class MetatomicModelVAE(torch.nn.Module):
        def __init__(self, 
                    encoder: torch.nn.Module,
                    normIn: bool = False,
                    dmean: torch.Tensor = torch.zeros(1), 
                    drange: torch.Tensor = torch.ones(1),
                    ):
            super().__init__()
            self.encoder = encoder

            self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
            self.register_buffer('Mean', dmean)
            self.register_buffer('Range', drange)
        
        def get_metatomic_outputs(self):
            return {"features": ModelOutput(quantity="", unit="none", per_atom=False),}

        def forward(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:

            if self.normIn:
                # TorchScript-compatible broadcasting
                # Reshape Mean and Range to match x dimensions for broadcasting
                mean_expanded = self.Mean.view(1, -1).expand_as(x)
                range_expanded = self.Range.view(1, -1).expand_as(x)
                
                x = (x - mean_expanded) / range_expanded
            latent = self.encoder(x)
            mean, logvar = latent
            return mean

except ImportError:
    pass

