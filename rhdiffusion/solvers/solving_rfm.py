import torch
from rhdiffusion.schedulers.scheduling_rfm import RFMScheduler

class RFMSolver:
    def __init__(self, sampler):
        self.sampler = sampler
    
    @classmethod
    def from_config(cls, config: dict) -> 'RFMSolver':

        num_inference_steps = config.pop('num_inference_steps', 50)

        sampler = RFMScheduler.from_config(config)

        sampler.set_timesteps(num_inference_steps)

        return cls(sampler=sampler)

    @torch.no_grad()
    def solve(self, network, noise, seed=None):

        t_start, t_end = 1.0, 0.0
        dt = (t_end - t_start) / self.sampler.num_inference_steps

        for step in range(self.sampler.num_inference_steps):
            t = t_start + step * (t_end - t_start) / self.sampler.num_inference_steps
            timestep = int(t * (self.num_train_timesteps - 1))

            v_pred = network(noise, timestep).sample

            if hasattr(v_pred, 'sample'):
                v_pred = v_pred.sample

            noise = noise + v_pred * dt

        return noise