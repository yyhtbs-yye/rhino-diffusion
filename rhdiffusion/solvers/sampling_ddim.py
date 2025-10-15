import torch
import inspect
from diffusers import DDIMScheduler

class DDIMSampler:
    def __init__(self, sampler, eta):
        self.sampler = sampler
        self.eta = eta

    @classmethod
    def from_config(cls, config: dict) -> 'DDIMSampler':

        num_inference_steps = config.pop('num_inference_steps', 50)
        eta = config.pop('eta', None)

        sampler = DDIMScheduler.from_config(config)

        sampler.set_timesteps(num_inference_steps)

        return cls(sampler=sampler, eta=eta)

    @torch.no_grad()
    def solve(self, network, noise, encoder_hidden_states=None, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=network.device).manual_seed(seed)
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)
        
        # Prepare noise
        noise = noise * self.sampler.init_noise_sigma
        
        # Denoising loop
        timesteps_iter = self.sampler.timesteps

        for i, t in enumerate(timesteps_iter):
            model_input = self.sampler.scale_model_input(noise, t)
            noise_pred = network(model_input, t, encoder_hidden_states)

            if hasattr(noise_pred, 'sample'):
                noise_pred = noise_pred.sample
            
            noise = self.sampler.step(noise_pred, t, noise, **extra_step_kwargs)["prev_sample"]

        return noise
    
    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for the scheduler step."""
        accepts_eta = 'eta' in set(
            inspect.signature(self.sampler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        
        # Check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.sampler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs