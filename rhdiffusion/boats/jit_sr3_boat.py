import torch

from rhdiffusion.boats.conditioned_diffusion_boat import BaseDiffusionBoat
from rhcore.utils.build_components import build_modules
from rhtrain.utils.ddp_utils import move_to_device

class JiTSR3Boat(BaseDiffusionBoat):
    # The prediction is x0_hat (clean SR image)
    # The loss is on the flow-matching velocity field

    def __init__(self, config={}):
        super().__init__(config=config)

        self.data_name = 'gt'
        self.cond_name = 'cond'

    @torch.no_grad()
    def predict(self, noise, cond=None):
        """
        Sampling: start from noise and integrate the learned flow
        using self.samplers['net'].solve(net, ...).

        Assumes the network predicts x0_hat and the sampler knows
        how to turn that into a velocity internally.
        """
        net = self.maybe_get_ema('net')

        if cond is not None:
            data_hat = self.samplers['net'].solve(net, noise, cond)
        else:
            data_hat = self.samplers['net'].solve(net, noise)

        return torch.clamp(data_hat, -1, 1)

    def diffusion_calc_losses(self, batch):
        """
        Flow matching, denoiser parameterization:

        - x0 := data  ~ p_data
        - z  := noise ~ N(0, I)
        - t  ~ Uniform(eps, 1 - eps)

        Path:          x_t = (1 - kappa_t) * z + kappa_t * x0      (kappa_t = t)
        True velocity: v_t(x_t)   = (x0 - x_t) / (1 - kappa_t)
        Pred velocity: v_hat(x_t) = (x0_hat - x_t) / (1 - kappa_t)

        Network output is x0_hat; we compute the velocity and train in v-space.
        """

        data = batch[self.data_name]             # clean HR image (x0)
        cond = batch.get(self.cond_name, None)   # LR condition

        device = data.device
        batch_size = data.size(0)

        # source noise z ~ N(0, I)
        noise = torch.randn_like(data)

        # sample t ~ Uniform(eps, 1 - eps) to avoid 1 / (1 - t) blow-up
        eps = 1e-3
        t = eps + (1.0 - 2.0 * eps) * torch.rand(batch_size, device=device)  # (B,)

        # reshape t for broadcasting over image dimensions
        while t.ndim < data.ndim:
            t = t.unsqueeze(-1)   # (B,1,1,1) for images

        kappa = t                # using kappa_t = t
        one_minus_kappa = 1.0 - kappa

        # flow-matching probability path: x_t = (1 - kappa_t) * z + kappa_t * x0
        x_t = one_minus_kappa * noise + kappa * data

        # ---- network prediction: x0_hat (clean image) ----
        if cond is not None:
            x0_hat = self.models['net'](x_t, t.squeeze(-1).squeeze(-1).squeeze(-1), cond)
        else:
            x0_hat = self.models['net'](x_t, t.squeeze(-1).squeeze(-1).squeeze(-1))

        # ---- true + predicted velocities (denoiser parameterization) ----
        # u_t(x_t | x0)   = (x0     - x_t)   / (1 - kappa_t)
        # u_hat_t(x_t)    = (x0_hat - x_t)   / (1 - kappa_t)
        v_true = (data   - x_t)   / one_minus_kappa
        v_hat  = (x0_hat - x_t)   / one_minus_kappa

        # simple uniform weight over t; adjust if you want a custom schedule
        weight = torch.ones(data.size(0), device=device)

        train_output = {
            'prediction': v_hat,
            'target': v_true,
            'weight': weight,
            **batch,
        }

        losses = {'total_loss': torch.tensor(0.0, device=device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, epoch):

        batch = move_to_device(batch, self.device)

        cond = batch.get(self.cond_name, None)
        data = batch[self.data_name]

        noise = torch.randn_like(data)

        data_hat = self.predict(noise, cond)

        valid_output = {'generation': data_hat, 'target': data}
        metrics = self.calc_metrics(valid_output)

        named_imgs = {'generation': data_hat, 'target': data}

        if cond is not None and isinstance(cond, torch.Tensor) and cond.dim() == 4:
            named_imgs['condition'] = cond

        return metrics, named_imgs

    def build_samplers(self, sampler_config):
        return build_modules(sampler_config)
