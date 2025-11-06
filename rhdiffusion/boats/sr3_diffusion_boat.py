import torch

from rhdiffusion.boats.base_diffusion_boat import BaseDiffusionBoat
from rhtrain.utils.ddp_utils import move_to_device

class SR3DiffusionBoat(BaseDiffusionBoat):

    def predict(self, noise, upsampled):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
            
        x0_hat = self.models['solver'].solve(network_in_use, noise, upsampled)
        
        result = torch.clamp(x0_hat, -1, 1)
        
        return result
        
    def training_calc_losses(self, batch):

        gt = batch['gt']
        lq = batch['lq']

        upsampled = self.models['upsample'](lq)

        batch_size = gt.size(0)

        noise = torch.randn_like(gt)

        timesteps = self.models['scheduler'].sample_timesteps(batch_size, self.device)
        
        xt = self.models['scheduler'].perturb(gt, noise, timesteps)

        targets = self.models['scheduler'].get_targets(gt, noise, timesteps)
        
        x0_hat = self.models['net'](xt, timesteps, encoder_hidden_states=upsampled)['sample']
        
        weights = self.models['scheduler'].get_loss_weights(timesteps)

        train_output = {
            'preds': x0_hat,
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
        lq = batch.get('lq', None)

        with torch.no_grad():

            noise = torch.randn_like(gt)

            upsampled = self.models['upsample'](lq)
            
            x0_hat = self.predict(noise, upsampled)

            valid_output = {'preds': x0_hat, 'targets': gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'high res': gt, 'low res': upsampled, 'super res': x0_hat,}

        return metrics, named_imgs
    