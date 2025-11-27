import torch

from rhdiffusion.boats.base_diffusion_boat import BaseDiffusionBoat
from rhtrain.utils.ddp_utils import move_to_device

class LatentDiffusionBoat(BaseDiffusionBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        # Override the names
        self.raw_name = 'gt'
        self.data_name = 'latent'

    def predict(self, zT):
        
        net = self.maybe_get_ema('net')

        z0_hat = self.samplers['net'].solve(net, zT)
        
        x0_hat = self.pretrained['latent_encoder'].decode(z0_hat)
        
        return torch.clamp(x0_hat, -1, 1)
        
    def diffusion_calc_losses(self, batch):

        with torch.no_grad():
            batch[self.data_name] = self.pretrained['latent_encoder'].encode(batch[self.raw_name])
        
        return super().diffusion_calc_losses(batch)
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        x0 = batch[self.raw_name]

        z0 = self.pretrained['latent_encoder'].encode(x0)

        zT = torch.randn_like(z0)
        
        x0_hat = self.predict(zT)

        valid_output = {'generation': x0_hat, 'target': x0,}
        metrics = self.calc_metrics(valid_output)

        named_imgs = {'generation': x0_hat, 'target': x0,}

        return metrics, named_imgs