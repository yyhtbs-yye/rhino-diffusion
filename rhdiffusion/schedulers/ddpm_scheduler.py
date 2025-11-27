import math
import torch

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000,
                 beta_start=1e-4, beta_end=0.02,
                 beta_schedule="linear"):
        self.num_train_timesteps = num_train_timesteps

        # ---- Beta schedule ----
        if beta_schedule == "linear":
            # Linearly increasing betas from beta_start to beta_end
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal 2021
            timesteps = torch.arange(num_train_timesteps + 1, dtype=torch.float32) / num_train_timesteps
            s = 0.008
            alphas_cumprod = torch.cos(((timesteps + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError("Unsupported beta_schedule: {}".format(beta_schedule))

        # ---- Precompute all useful quantities ----
        self.betas = betas                                          # β_t
        self.alphas = 1.0 - betas                                   # α_t = 1 − β_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)     # \bar α_t
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), self.alphas_cumprod[:-1]], dim=0
        )                                                           # \bar α_{t-1}

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)               # √\bar α_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # √(1 − \bar α_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)                   # 1/√α_t

        # Posterior variance β̃_t = Var[q(x_{t-1} | x_t, x_0)]
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    # ---------- Small helpers ----------

    def _normalize_timesteps(self, timesteps, device):
        # Make sure we always have a 1D LongTensor on the right device
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

    # ---------- Training API (DDPM-style ε-prediction) ----------

    def sample_timesteps(self, batch_size, device, low=0, high=None):
        high = self.num_train_timesteps if high is None else high
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

    # For compatibility with your DDIMScheduler subclass
    def perturb(self, imgs, noise, timesteps):
        return self.add_noise(imgs, noise, timesteps)

    def get_targets(self, imgs, noises, timesteps):
        # Standard DDPM objective: predict ε
        return noises

    def get_loss_weights(self, timesteps):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps)
        return torch.ones_like(timesteps, dtype=torch.float32, device=timesteps.device)

    def scale_model_input(self, sample, timestep):
        # DDPM does not require any special input scaling
        return sample

    # ---------- Sampling (reverse diffusion) ----------

    def step(self, model_output, timestep, sample, generator=None):
        """
        One reverse diffusion step: x_t -> x_{t-1}

        model_output: ε_θ(x_t, t), predicted noise
        timestep: int or tensor of shape [batch]
        sample: x_t
        """
        device = sample.device
        timestep = self._normalize_timesteps(timestep, device)

        # If single scalar t is given but batch > 1, broadcast t over batch
        if timestep.numel() == 1 and sample.shape[0] > 1:
            timestep = timestep.expand(sample.shape[0])

        # Gather per-timestep scalars and reshape to broadcast over sample
        betas_t = self._extract(self.betas, timestep, sample.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timestep, sample.shape
        )
        sqrt_recip_alphas_t = self._extract(
            self.sqrt_recip_alphas, timestep, sample.shape
        )

        # Equation (11) in the DDPM paper:
        # μ_θ(x_t, t) = 1/√α_t * (x_t − β_t / √(1 − \bar α_t) * ε_θ(x_t, t))
        model_mean = sqrt_recip_alphas_t * (
            sample - betas_t / sqrt_one_minus_alphas_cumprod_t * model_output
        )

        posterior_variance_t = self._extract(
            self.posterior_variance, timestep, sample.shape
        )

        # Sample noise
        if generator is None:
            noise = torch.randn_like(sample)
        else:
            noise = torch.randn(sample.shape, device=device, generator=generator)

        # No noise when t == 0
        nonzero_mask = (timestep != 0).float().view(-1, *([1] * (sample.dim() - 1)))
        sample_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

        # Predicted x_0 for monitoring/guidance:
        # x_0 = (x_t − √(1 − \bar α_t) ε_θ) / √\bar α_t
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, timestep, sample.shape
        )
        pred_original_sample = (sample - sqrt_one_minus_alphas_cumprod_t * model_output) / sqrt_alphas_cumprod_t

        return {
            "prev_sample": sample_prev,           # x_{t-1}
            "pred_original_sample": pred_original_sample,  # x_0 estimate
            "model_mean": model_mean,             # μ_θ(x_t, t)
        }

if __name__ == "__main__":

    def test_add_noise_and_recover_x0():
        print("Running test_add_noise_and_recover_x0...")
        torch.manual_seed(0)
        device = "cpu"

        scheduler = DDPMScheduler(num_train_timesteps=1000)

        batch_size = 8
        x0 = torch.randn(batch_size, 3, 32, 32, device=device)
        eps = torch.randn_like(x0)

        # Random timesteps for each batch element
        timesteps = scheduler.sample_timesteps(batch_size, device=device)

        # Forward diffusion: x_t = sqrt(alphā_t) x0 + sqrt(1 - alphā_t) ε
        x_t = scheduler.add_noise(x0, eps, timesteps)

        # Reverse step, using the *true* noise as model output
        out = scheduler.step(model_output=eps, timestep=timesteps, sample=x_t)
        x0_pred = out["pred_original_sample"]

        # Check that the reconstruction of x0 is numerically accurate
        max_error = (x0 - x0_pred).abs().max().item()
        print("  max |x0 - x0_pred| =", max_error)

        assert max_error < 1e-4, "Predicted x0 does not match original x0 closely enough"


    def test_shapes_and_ranges():
        print("Running test_shapes_and_ranges...")
        torch.manual_seed(1)
        device = "cpu"

        scheduler = DDPMScheduler(num_train_timesteps=1000)

        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32, device=device)
        eps = torch.randn_like(x0)

        # sample_timesteps
        timesteps = scheduler.sample_timesteps(batch_size, device=device, low=0, high=scheduler.num_train_timesteps)
        print("  timesteps:", timesteps)

        assert timesteps.shape == (batch_size,)
        assert timesteps.dtype == torch.long
        assert (timesteps >= 0).all() and (timesteps < scheduler.num_train_timesteps).all()

        # add_noise
        x_t = scheduler.add_noise(x0, eps, timesteps)
        assert x_t.shape == x0.shape, "add_noise output shape mismatch"

        # step
        out = scheduler.step(model_output=eps, timestep=timesteps, sample=x_t)
        prev_sample = out["prev_sample"]
        x0_pred = out["pred_original_sample"]

        assert prev_sample.shape == x0.shape, "prev_sample shape mismatch"
        assert x0_pred.shape == x0.shape, "pred_original_sample shape mismatch"

        print("  Shapes OK.")


    def smoke_test_sampling_loop():
        print("Running smoke_test_sampling_loop...")
        torch.manual_seed(2)
        device = "cpu"

        scheduler = DDPMScheduler(num_train_timesteps=50)  # shorter chain for the test

        batch_size, C, H, W = 2, 3, 16, 16
        x = torch.randn(batch_size, C, H, W, device=device)  # x_T ~ N(0, I)

        # Dummy model: predicts zero noise (just to check the loop runs)
        def dummy_model(x_t, t):
            return torch.zeros_like(x_t)

        for t in reversed(range(scheduler.num_train_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps_pred = dummy_model(x, t_batch)
            out = scheduler.step(eps_pred, t_batch, x)
            x = out["prev_sample"]

        # If we got here without errors, sampling loop is at least numerically consistent
        print("  Sampling loop ran without errors. Final sample shape:", x.shape)


    test_add_noise_and_recover_x0()
    test_shapes_and_ranges()
    smoke_test_sampling_loop()
    print("All tests passed!")
