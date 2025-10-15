import torch

from rhcore.boats.base_boat import BaseBoat
from rhtrain.utils.ddp_utils import move_to_device

class BaseDiffusionBoat(BaseBoat):

    def predict(self, noise):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        x0_hat = self.models['solver'].solve(network_in_use, noise)

        result = torch.clamp(x0_hat, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch):

        gt = batch['gt']

        batch_size = gt.size(0)
        
        # Initialize random noise in latent space
        noise = torch.randn_like(gt)

        # Sample random timesteps
        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        # Add noise to latents according to noise schedule
        xt = self.models['scheduler'].perturb(gt, noise, timesteps)

        # Get targets for the denoising process
        targets = self.models['scheduler'].get_targets(gt, noise, timesteps)
        
        x0_hat = self.models['net'](xt, timesteps)['sample']
        
        weights = self.models['scheduler'].get_loss_weights(timesteps)

        train_output = {'preds': x0_hat,
                        'targets': targets,
                        'weights': weights,
                        **batch}
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses
        
    def validation_step(self, batch, batch_idx):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']

        with torch.no_grad():

            noise = torch.randn_like(gt)

            x0_hat = self.predict(noise)

            valid_output = {'preds': x0_hat, 'targets': gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'gt': gt, 'gen': x0_hat,}

        return metrics, named_imgs