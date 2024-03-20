import torch.nn as nn
from diffusion.ddpm import DDPM
from models.d2c_vae.autoencoder_unet import Autoencoder
from models.d2c_vae.mlp import MLP
from huggingface_hub import PyTorchModelHubMixin



class DDMI(nn.Module, PyTorchModelHubMixin):
    def __init__(self, ldm, vae, mlp):
        super().__init__()
        self.ldm = ldm
        self.vae = vae
        self.mlp = mlp