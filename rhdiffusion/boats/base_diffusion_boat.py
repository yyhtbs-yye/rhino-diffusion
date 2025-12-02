import torch

from rhcore.boats.base_boat import BaseBoat

from rhcore.utils.build_components import build_modules
from rhtrain.utils.ddp_utils import move_to_device


class BaseDiffusionBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        self.sampler_config = config['boat'].get('samplers')
        self.samplers = self.build_samplers(self.sampler_config)

        # The default training loop for single entity training
        if self.ordered_groups is None or len(self.ordered_groups) == 0:
            self.ordered_groups = [
                {
                    'boat_loss_method_str': 'diffusion_calc_losses',
                    'target_loss_name': 'total_loss',
                    'models': ['net'],
                    'optimizers': ['net'],
                    'train_interval': 1,
                },
            ]
        
        self.data_name = 'gt'

    def maybe_get_ema(self, name):
        return self.models[f'{name}_ema'] if f'{name}_ema' in self.models and self.use_ema else self.models[name]

    @torch.no_grad()
    def predict(self, noise):
        
        net = self.maybe_get_ema('net')
        data_hat = self.samplers['net'].solve(net, noise)
        
        return torch.clamp(data_hat, -1, 1)
        
    def diffusion_calc_losses(self, batch):

        data = batch[self.data_name]

        batch_size = data.size(0)
        
        noise = torch.randn_like(data)

        timestep = self.samplers['net'].sample_timestep(batch_size, self.device)
        perturbed = self.samplers['net'].perturb(data, noise, timestep)

        eps = self.samplers['net'].get_target(data, noise, timestep)
        eps_hat = self.models['net'](perturbed, timestep)['sample']
        
        weight = self.samplers['net'].get_loss_weight(timestep)

        train_output = {'prediction': eps_hat, 'target': eps, 'weight': weight,
                        **batch}
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        data = batch[self.data_name]

        noise = torch.randn_like(data)

        data_hat = self.predict(noise)

        valid_output = {'generation': data_hat, 'target': data,}

        metrics = self.calc_metrics(valid_output)

        named_imgs = {'generation': data_hat, 'target': data,}

        return metrics, named_imgs

    def build_samplers(self, sampler_config):
        
        return build_modules(sampler_config)
    