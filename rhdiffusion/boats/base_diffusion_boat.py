import torch

from rhcore.boats.base_boat import BaseBoat

from rhcore.utils.build_components import build_modules
from rhtrain.utils.ddp_utils import move_to_device

class BaseDiffusionBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        self.sampler_config = self.boat_config.get('samplers')
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
    def predict(self, xT):
        
        net = self.maybe_get_ema('net')
        x0_hat = self.samplers['net'].solve(net, xT)
        
        return torch.clamp(x0_hat, -1, 1)
        
    def diffusion_calc_losses(self, batch):

        x0 = batch[self.data_name]

        batch_size = x0.size(0)
        
        xT = torch.randn_like(x0)

        timestep = self.samplers['net'].sample_timestep(batch_size, self.device)
        xt = self.samplers['net'].perturb(x0, xT, timestep)

        eps = self.samplers['net'].get_target(x0, xT, timestep)
        eps_hat = self.models['net'](xt, timestep)['sample']
        
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

        x0 = batch[self.data_name]

        xT = torch.randn_like(x0)

        x0_hat = self.predict(xT)

        valid_output = {'generation': x0_hat, 'target': x0,}

        metrics = self.calc_metrics(valid_output)

        named_imgs = {'generation': x0_hat, 'target': x0,}

        return metrics, named_imgs

    def build_samplers(self, sampler_config):
        
        return build_modules(sampler_config)