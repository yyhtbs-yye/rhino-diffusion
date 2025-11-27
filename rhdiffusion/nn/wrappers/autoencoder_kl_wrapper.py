from diffusers.models.autoencoders import AutoencoderKL
import torch.nn as nn

class AutoencoderKLWrapper(nn.Module):

    def __init__(self, model):
        super().__init__() 

        self.model = model
        self.config = self.model.config
        self.eval()

    def encode(self, x):
        z = self.model.encode(x).latent_dist.mean
        z = z * self.config.scaling_factor
        return z

    def decode(self, z):
        z = 1 / self.config.scaling_factor * z
        x = self.model.decode(z).sample
        return x
    
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path):
        model = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        return AutoencoderKLWrapper(model)

    @staticmethod
    def from_config(config):
        model = AutoencoderKL.from_config(config)
        return AutoencoderKLWrapper(model)
