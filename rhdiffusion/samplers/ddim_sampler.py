from copy import deepcopy

import torch
import inspect
from diffusers import DDIMScheduler

class DDIMSampler:
    def __init__(self, scheduler, eta):
        self.scheduler = scheduler
        self.eta = eta

    @classmethod
    def from_config(cls, config: dict):

        num_inference_steps = config.pop('num_inference_steps', 50)

        config = deepcopy(config)

        eta = config.pop('eta', None)

        scheduler = DDIMScheduler.from_config(config)

        scheduler.set_timesteps(num_inference_steps)

        return cls(scheduler=scheduler, eta=eta)

    @torch.no_grad()
    def solve(self, network, noise, cond=None, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=network.device).manual_seed(seed)
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)
        
        # Prepare noise
        noise = noise * self.scheduler.init_noise_sigma
        
        # Denoising loop
        timesteps_iter = self.scheduler.timesteps

        for i, t in enumerate(timesteps_iter):
            model_input = self.scheduler.scale_model_input(noise, t)
            ts = t.repeat(model_input.size(0)).to(model_input.device)
            
            if cond is not None:
                noise_pred = network(model_input, ts, cond)
            else:
                noise_pred = network(model_input, ts)

            if hasattr(noise_pred, 'sample'):
                noise_pred = noise_pred.sample
            if isinstance(noise_pred, dict) and 'sample' in noise_pred:
                noise_pred = noise_pred['sample']
            
            noise = self.scheduler.step(noise_pred, t, noise, **extra_step_kwargs)["prev_sample"]

        return noise

    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for the scheduler step."""
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        
        # Check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs
    
    def sample_timestep(self, batch_size, device, low=0, high=None):
        
        high = high or self.scheduler.config.num_train_timesteps
        return torch.randint(low, high, (batch_size,), device=device).long()
    
    def perturb(self, data, noise, timestep):
        return self.scheduler.add_noise(data, noise, timestep)

    # This is a pure epsilon prediction, so just return noise
    def get_target(self, data, noise, timestep):
        return noise
    
    def get_loss_weight(self, timestep):
        return torch.ones_like(timestep)