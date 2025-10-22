import torch

from rhdiffusion.boat.base_diffusion_boat import BaseDiffusionBoat
from rhtrain.utils.ddp_utils import move_to_device

class LatentDiffusionBoat(BaseDiffusionBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        for param in self.models['latent_encoder'].parameters():
            param.requires_grad = False
        self.models['latent_encoder'].eval()

    def predict(self, noise):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        z0_hat = self.models['solver'].solve(network_in_use, noise)

        x0_hat = self.decode_latents(z0_hat)
        
        result = torch.clamp(x0_hat, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch):

        gt = batch['gt']

        batch_size = gt.size(0)
        
        with torch.no_grad():
            latents = self.encode_images(gt)

        # Initialize random noise in latent space
        noise = torch.randn_like(latents)

        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        zt = self.models['scheduler'].perturb(latents, noise, timesteps)

        targets = self.models['scheduler'].get_targets(latents, noise, timesteps)
        
        z0_hat = self.models['net'](zt, timesteps)['sample']
        
        weights = self.models['scheduler'].get_loss_weights(timesteps)

        train_output = {
            'preds': z0_hat,
            'targets': targets,
            'weights': weights,
            **batch
        }
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses
        
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']

        with torch.no_grad():

            latents = self.encode_images(gt)

            noise = torch.randn_like(latents)

            x0_hat = self.predict(noise)

            valid_output = {'preds': x0_hat, 'targets': gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'gt': gt, 'gen': x0_hat,}

        return metrics, named_imgs
    
    def decode_latents(self, z):
        # Scale latents according to VAE configuration
        x = self.models['latent_encoder'].decode(z)
        return x
    
    def encode_images(self, x):
        # Encode images to latent space
        z = self.models['latent_encoder'].encode(x)
        return z
