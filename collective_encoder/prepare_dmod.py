import os
import yaml

import torch

import collective_encoder.ce_run_base as crb

_SETTINGS = {
    'module': 'dmod',
}

def prepare_dmod():
    """Prepare the datamodule for training a collective encoder model based on the provided configuration."""

    config, metargs = crb.prepare(_SETTINGS)
    dm = crb.load_datamodule(config, metargs)
    run_dir = metargs['run_dir']
    
    ##################################
    # Saving datamodule and config
    ##################################
    if 'data_analysers' in config:
        dm.ext_analyse_data(data_analysers=config['data_analysers'])
    else:
        torch.save(dm, os.path.join(run_dir, "datamodule.pth"))
        yaml.dump({**config, **metargs}, open(os.path.join(run_dir, "config.yaml"), 'w'))


def main():
    """Main entry point for preparing the datamodule."""
    prepare_dmod()

if __name__ == "__main__":
    main()