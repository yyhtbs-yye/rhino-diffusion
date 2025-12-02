import torch

from rhdiffusion.boats.base_diffusion_boat import BaseDiffusionBoat

from rhcore.utils.build_components import build_modules
from rhtrain.utils.ddp_utils import move_to_device

class ConditionedLatentDiffusionBoat(BaseDiffusionBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        self.data_name = 'gt'
        self.cond_name = 'cond'
        self.latent_dim = None

    @torch.no_grad()
    def predict(self, noise, cond=None):
        
        net = self.maybe_get_ema('net')

        if cond is not None:
            cond = self.processors['preproc'](cond)
            data_hat = self.samplers['net'].solve(net, noise, cond)
        else:
            data_hat = self.samplers['net'].solve(net, noise)

        if 'latent_encoder' in self.pretrained:
            data_hat = self.pretrained['latent_encoder'].decode(data_hat)

        return torch.clamp(data_hat, -1, 1)

    def diffusion_calc_losses(self, batch):

        data = batch[self.data_name]
        cond = batch.get(self.cond_name, None)

        if cond is not None:
            cond = self.processors['preproc'](cond)

        if 'latent_encoder' in self.pretrained:
            data = self.pretrained['latent_encoder'].encode(data)
            self.latent_dim = data.shape[1:]

        batch_size = data.size(0)
        
        noise = torch.randn_like(data)

        t = self.samplers['net'].sample_timestep(batch_size, self.device)
        perturbed = self.samplers['net'].perturb(data, noise, t)

        eps = self.samplers['net'].get_target(data, noise, t)

        weight = self.samplers['net'].get_loss_weight(t)
        
        if cond is not None:
            eps_hat = self.models['net'](perturbed, t, cond)
        else:
            eps_hat = self.models['net'](perturbed, t)
        
        train_output = {'prediction': eps_hat, 'target': eps, 'weight': weight,
                        **batch}
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        cond = batch.get(self.cond_name, None)

        data = batch[self.data_name]
        batch_size = data.size(0)

        if self.latent_dim is not None and 'latent_encoder' in self.pretrained:
            noise = torch.randn([batch_size, *self.latent_dim], device=self.device)
        elif self.latent_dim is None and 'latent_encoder' in self.pretrained:
            latent_example = self.pretrained['latent_encoder'].encode(data[:1])
            self.latent_dim = latent_example.shape[1:]
            noise = torch.randn([batch_size, *self.latent_dim], device=self.device) 
            del latent_example
        else:
            noise = torch.randn_like(data)

        data_hat = self.predict(noise, cond)

        valid_output = {'generation': data_hat, 'target': data,}

        metrics = self.calc_metrics(valid_output)

        named_imgs = {'generation': data_hat, 'target': data,}

        if cond is not None and isinstance(cond, torch.Tensor) and cond.dim() == 4:
            named_imgs['condition'] = cond

        return metrics, named_imgs

    def build_samplers(self, sampler_config):
        
        return build_modules(sampler_config)