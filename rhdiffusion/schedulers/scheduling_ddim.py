import diffusers
import torch

class DDIMScheduler(diffusers.DDIMScheduler):

    def sample_timesteps(self, batch_size, device, low=0, high=None):
        
        high = high or self.config.num_train_timesteps
        return torch.randint(low, high, (batch_size,), device=device).long()
    
    def perturb(self, imgs, noise, timesteps):
        return self.add_noise(imgs, noise, timesteps)

    def get_targets(self, imgs, noises, timesteps):
        return noises
    
    def get_loss_weights(self, timesteps):
        return torch.ones_like(timesteps)