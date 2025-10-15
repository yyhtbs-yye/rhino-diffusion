import torch
from diffusers.models import UNet2DConditionModel, UNet2DModel

class UNet2DConcatConditionModel(UNet2DModel):
    def __init__(self, sample_size=None, in_channels=None, out_channels=None,
                 down_block_types=("DownBlock2D",), up_block_types=("UpBlock2D",),
                 block_out_channels=(64,), layers_per_block=1, attention_head_dim=8,
                 norm_num_groups=32, norm_eps=1e-5, sample_channels=None, condition_channels=None,
                 **kwargs):

        self.sample_channels = sample_channels
        self.condition_channels = condition_channels if condition_channels is not None else 0
        
        in_channels = sample_channels + condition_channels
            
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            **kwargs
        )
    
    def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        """
        Overrides the forward method to handle conditioning through concatenation.
        
        Args:
            sample: The input sample tensor
            timestep: The diffusion timestep
            encoder_hidden_states: The conditioning input to concatenate with sample
        """
        # Concatenate sample and encoder_hidden_states along channel dimension
        concat_input = torch.cat([sample, encoder_hidden_states], dim=1)
            
        # Call parent forward
        return super().forward(concat_input, timestep, **kwargs)

class UNet2DFlexibleWrapper:
    """
    Factory class for creating UNet models with different conditioning approaches.
    This class does not inherit from any other class and only provides the from_config method.
    """
    @classmethod
    def from_config(cls, config):
        """
        Factory method that creates the appropriate model based on the mode in config.
        
        Args:
            config (dict): Configuration dictionary for the model
            
        Returns:
            Either a UNet2DConditionModel (for cross-attention conditioning) or 
            UNet2DConcatConditionModel (for concatenation-based conditioning)
        """
        # Extract mode from config, default to cross_attention
        mode = config.pop("mode", "none")
        
        # Make a copy of the config to avoid modifying the original
        config_copy = config.copy()
        
        # Extract parameters needed for both models
        sample_channels = config_copy.pop("sample_channels", None)
        condition_channels = config_copy.pop("condition_channels", None)
        
        if mode not in ["none", "concat", "cross_attention"]:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'none', 'concat', and 'cross_attention'.")
        
        if mode == "none":
            # For no conditioning, create a UNet2DModel
            
            # Set in_channels to sample_channels only
            config_copy["in_channels"] = sample_channels
            
            # Remove cross-attention specific parameters
            config_copy.pop("cross_attention_dim", None)
            # Remove condition_channels specific parameters
            config_copy.pop("condition_channels", None)
                
            # Create and return UNet2DModel
            return UNet2DConcatConditionModel(**config_copy)
        elif mode == "concat":
            # For concat mode, create a UNet2DConcatConditionModel
            
            # Set in_channels for concatenation
            config_copy["in_channels"] = sample_channels + condition_channels
            
            # Store the original channel dimensions
            config_copy["sample_channels"] = sample_channels
            config_copy["condition_channels"] = condition_channels
            
            # Remove cross-attention specific parameters
            config_copy.pop("cross_attention_dim", None)
            
            # Ensure block types don't have cross attention
            if "down_block_types" in config_copy:
                config_copy["down_block_types"] = [
                    block_type.replace("CrossAttn", "") 
                    for block_type in config_copy["down_block_types"]
                ]
            
            if "up_block_types" in config_copy:
                config_copy["up_block_types"] = [
                    block_type.replace("CrossAttn", "") 
                    for block_type in config_copy["up_block_types"]
                ]
                
            # Create and return UNet2DConcatConditionModel
            return UNet2DConcatConditionModel(**config_copy)
        else:
            # For cross-attention mode, create a UNet2DConditionModel
            
            # Set in_channels to sample_channels only
            config_copy["in_channels"] = sample_channels
            
            # Ensure cross_attention_dim is set if not present
            if "cross_attention_dim" not in config_copy and condition_channels is not None:
                config_copy["cross_attention_dim"] = condition_channels
                
            # Create and return UNet2DConditionModel
            return UNet2DConditionModel(**config_copy)