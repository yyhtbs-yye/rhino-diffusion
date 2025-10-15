import torch
from diffusers import DDIMScheduler
from diffusers.configuration_utils import register_to_config

class RFMScheduler(DDIMScheduler):
    """
    Scheduler for Rectified Flow Matching.
    
    This scheduler implements the rectified flow matching process as a pure ODE, 
    where the path between the data distribution and the reference distribution 
    is rectified to make the trajectories more direct and deterministic.
    """
    
    @register_to_config
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        clip_sample_range=1.0,
        timestep_spacing="leading",
        rectification_strength=0.1,
    ):
        """
        Initialize the RFMScheduler by extending DDIMScheduler.
        
        Args:
            rectification_strength (float): Strength of the rectification effect (0-1).
                Higher values make paths more direct but might affect stability.
        """
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type='v_prediction',
            clip_sample_range=clip_sample_range,
            timestep_spacing=timestep_spacing,
        )
        
        # Calculate rectification coefficients
        self.rectification_coeffs = self._calculate_rectification_coeffs(rectification_strength)
    
    def _calculate_rectification_coeffs(self, strength=0.1):
        """Calculate rectification coefficients for the flow."""
        num_timesteps = len(self.alphas_cumprod)
        t = torch.linspace(0, 1, num_timesteps, dtype=torch.float32)
        
        # Simple sinusoidal rectification
        coeffs = 1.0 - strength * torch.sin(t * torch.pi)
        
        return coeffs

    def step(
        self,
        model_output,
        timestep,
        sample,
        eta=0.0,  # Always ignored in RFM as it's purely deterministic
        use_clipped_model_output=False,
        generator=None,
        variance_noise=None,
        return_dict=True,
    ):
        """
        Perform a deterministic ODE step for Rectified Flow Matching.
        
        Args:
            model_output (torch.Tensor): Output from the model (velocity prediction).
            timestep (int): Current timestep.
            sample (torch.Tensor): Current noisy sample.
            return_dict (bool): Whether to return a dictionary.
            
        Returns:
            torch.Tensor or dict: Predicted sample for the next timestep.
        """
        # Get indices for current and previous timesteps
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        # Apply rectification coefficients
        rect_t = self.rectification_coeffs[timestep].to(sample.device)
        rect_t_prev = self.rectification_coeffs[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0).to(sample.device)
        
        # Modify alphas with rectification for ODE path
        rectified_alpha_prod_t = alpha_prod_t * rect_t
        rectified_alpha_prod_t_prev = alpha_prod_t_prev * rect_t_prev
        
        # For RFM with v-prediction
        alpha_prod_t_sqrt = torch.sqrt(rectified_alpha_prod_t)
        alpha_prod_t_sqrt = alpha_prod_t_sqrt.flatten()
        
        # Extract predicted x_0 from velocity prediction
        pred_original_sample = (alpha_prod_t_sqrt.view(-1, 1, 1, 1) * sample - 
                               torch.sqrt(1 - rectified_alpha_prod_t).view(-1, 1, 1, 1) * model_output)
        pred_original_sample = pred_original_sample / alpha_prod_t_sqrt.view(-1, 1, 1, 1)
        
        # Clip predicted sample if needed
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range
            )
        
        # Deterministic update for next step using ODE solver
        alpha_prod_t_prev_sqrt = torch.sqrt(rectified_alpha_prod_t_prev)
        
        # Calculate predicted noise vector
        pred_noise = (sample - alpha_prod_t_sqrt.view(-1, 1, 1, 1) * pred_original_sample) / torch.sqrt(1 - rectified_alpha_prod_t).view(-1, 1, 1, 1)
        
        # Deterministic update formula
        prev_sample = alpha_prod_t_prev_sqrt.view(-1, 1, 1, 1) * pred_original_sample + torch.sqrt(1 - rectified_alpha_prod_t_prev).view(-1, 1, 1, 1) * pred_noise
        
        if not return_dict:
            return (prev_sample,)
        
        return {
            "prev_sample": prev_sample,
            "pred_original_sample": pred_original_sample,
        }

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples with rectification applied."""
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=original_samples.device)
        
        # Get alphas with rectification
        alphas = self.alphas_cumprod[timesteps].to(original_samples.device)
        rectified_alphas = alphas * self.rectification_coeffs[timesteps].to(alphas.device)
        
        # Apply deterministic noise addition
        sqrt_alpha = torch.sqrt(rectified_alphas).flatten().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1 - rectified_alphas).flatten().view(-1, 1, 1, 1)
        
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise
    
    def get_velocity(self, sample, noise, timesteps):
        """
        Calculate velocity vector for Rectified Flow Matching.
        
        Args:
            sample (torch.Tensor): Noisy sample.
            noise (torch.Tensor): Noise component.
            timesteps (torch.Tensor): Timesteps.
            
        Returns:
            torch.Tensor: Velocity vector.
        """
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=sample.device)
            
        # Get alphas with rectification
        alphas = self.alphas_cumprod[timesteps].to(sample.device)
        rectified_alphas = alphas * self.rectification_coeffs[timesteps].to(alphas.device)
        
        # Calculate velocity vector
        sqrt_alpha = torch.sqrt(rectified_alphas).view(-1, 1, 1, 1)
        sqrt_1m_alpha = torch.sqrt(1 - rectified_alphas).view(-1, 1, 1, 1)
        
        # v = sqrt(1-α)·x_0 - sqrt(α)·ε
        velocity = sqrt_1m_alpha * sample - sqrt_alpha * noise
        
        return velocity

    def perturb(self, imgs, noise, timesteps):
        return self.add_noise(imgs, noise, timesteps)

    def get_targets(self, original_samples, noise, timesteps):
        return self.get_velocity(original_samples, noise, timesteps)
    
    def get_loss_weights(self, timesteps):
        """Get loss weights based on SNR."""
        alphas = self.alphas_cumprod[timesteps].to(timesteps.device)
        rectified_alphas = alphas * self.rectification_coeffs[timesteps].to(alphas.device)
        
        # Simple SNR weighting
        snr = rectified_alphas / (1 - rectified_alphas)
        weights = snr / snr.mean()
        
        return weights