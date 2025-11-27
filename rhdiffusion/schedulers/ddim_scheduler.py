import math
import torch

class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",  # "linear" or "cosine"
        eta=0.0,                 # 0 = deterministic DDIM
        clip_sample=True
    ):
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta
        self.clip_sample = clip_sample

        # --------- Beta schedule ---------
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            timesteps = torch.arange(num_train_timesteps + 1, dtype=torch.float32) / num_train_timesteps
            s = 0.008
            alphas_cumprod = torch.cos(((timesteps + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError("Unsupported beta_schedule: {}".format(beta_schedule))

        self.betas = betas                              # β_t
        self.alphas = 1.0 - betas                       # α_t = 1 − β_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # \bar α_t

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)               # √\bar α_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # √(1−\bar α_t)

        # will be set later for inference
        self.timesteps = None
        self.num_inference_steps = None

    # --------- Helpers ---------

    def _normalize_timesteps(self, timesteps, device):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=device, dtype=torch.long)
        else:
            timesteps = timesteps.to(device).long()
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        return timesteps

    def _extract(self, a, t, x_shape):
        """
        Extract values from a 1D tensor `a` at indices `t`
        and reshape to broadcast over a tensor with shape `x_shape`.
        """
        a = a.to(t.device)
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    # --------- Training-side API (like your subclass) ---------

    def sample_timesteps(self, batch_size, device, low=0, high=None):
        high = high or self.num_train_timesteps
        return torch.randint(low, high, (batch_size,), device=device).long()

    def add_noise(self, original_samples, noise, timesteps):
        """
        q(x_t | x_0):
        x_t = √\bar α_t x_0 + √(1−\bar α_t) ε
        """
        timesteps = self._normalize_timesteps(timesteps, original_samples.device)
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, timesteps, original_samples.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape
        )
        return sqrt_alphas_cumprod_t * original_samples + sqrt_one_minus_alphas_cumprod_t * noise

    # For compatibility with your code
    def perturb(self, imgs, noise, timesteps):
        return self.add_noise(imgs, noise, timesteps)

    def get_targets(self, imgs, noises, timesteps):
        # Standard ε-prediction
        return noises

    def get_loss_weights(self, timesteps):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps)
        return torch.ones_like(timesteps, dtype=torch.float32, device=timesteps.device)

    def scale_model_input(self, sample, timestep):
        # DDIM generally uses unscaled inputs (like DDPM)
        return sample

    # --------- Inference-time setup ---------

    def set_timesteps(self, num_inference_steps, device=None):
        """
        Creates a sequence of inference timesteps, e.g. [999, ..., 0].
        """
        self.num_inference_steps = num_inference_steps
        # Linearly spaced indices from T-1 down to 0
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
        )
        if device is not None:
            timesteps = timesteps.to(device)
        self.timesteps = timesteps

    # --------- DDIM sampling step ---------

    def step(self, model_output, timestep, sample, eta=None, generator=None):
        """
        One DDIM step: x_t -> x_{t_prev}.

        model_output: ε_θ(x_t, t) (noise prediction)
        timestep: scalar int or (batch,) tensor, must be in self.timesteps
        sample: x_t
        eta: optional override, 0 = deterministic DDIM
        """
        if self.timesteps is None:
            raise ValueError("You must call set_timesteps(...) before using step().")

        device = sample.device
        timestep = self._normalize_timesteps(timestep, device)

        if eta is None:
            eta = self.eta

        # Assume same t for all batch elements
        t = timestep[0].item()

        # Find index of current timestep in self.timesteps
        timesteps_cpu = self.timesteps.to("cpu")
        idx = (timesteps_cpu == t).nonzero()
        if idx.numel() == 0:
            raise ValueError("Timestep {} is not in self.timesteps.".format(t))
        idx = idx[0].item()

        # Next (previous in the chain) timestep index, going toward 0
        if idx == len(self.timesteps) - 1:
            t_prev = -1  # special case: will map to ᾱ_prev = 1
        else:
            t_prev = self.timesteps[idx + 1].item()

        # ᾱ_t and ᾱ_{t_prev}
        alpha_prod_t = self._extract(self.alphas_cumprod, timestep, sample.shape)

        if t_prev >= 0:
            t_prev_tensor = torch.full_like(timestep, t_prev)
            alpha_prod_t_prev = self._extract(self.alphas_cumprod, t_prev_tensor, sample.shape)
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1.0 - alpha_prod_t)
        sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)

        # Predict x_0 from ε prediction
        pred_original_sample = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        # Equation from DDIM:
        # x_{t_prev} = √ᾱ_{t_prev} x_0 +
        #              √(1−ᾱ_{t_prev}−σ_t^2) ε_θ(x_t, t) + σ_t z
        # where
        # σ_t = η * sqrt( (1−ᾱ_{t_prev})/(1−ᾱ_t) * (1−ᾱ_t/ᾱ_{t_prev}) )
        sigma = 0.0
        if eta > 0:
            sigma = (
                (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
                * (1.0 - alpha_prod_t / alpha_prod_t_prev)
            )
            sigma = torch.sqrt(torch.clamp(sigma, min=0.0)) * eta

        noise = 0.0
        if eta > 0:
            if generator is None:
                noise = torch.randn_like(sample)
            else:
                noise = torch.randn(sample.shape, device=device, generator=generator)

        pred_dir = torch.sqrt(
            torch.clamp(1.0 - alpha_prod_t_prev - sigma**2, min=0.0)
        ) * model_output

        prev_sample = sqrt_alpha_prod_t_prev * pred_original_sample + pred_dir + sigma * noise

        return {
            "prev_sample": prev_sample,                 # x_{t_prev}
            "pred_original_sample": pred_original_sample,  # x_0 estimate
        }
