def get_net(model_name: str):
    """Returns the neural network class corresponding to the given model name."""
    if model_name == "VAE":
        from collective_encoder.nets.vae_net import VAE
        return VAE
    elif model_name == "AE":
        from collective_encoder.nets.ae_net import AE
        return AE
    elif model_name == "DVAE":
        from collective_encoder.nets.dvae_net import DVAE
        return DVAE
    elif model_name == "EDVAE":
        from collective_encoder.nets.edvae_net import EDVAE
        return EDVAE
    elif model_name == "BGE":
        from collective_encoder.nets.bge import BondGraphNetEncoderDecoder
        return BondGraphNetEncoderDecoder
    else:
        raise ValueError(f"Unknown model name: {model_name}")