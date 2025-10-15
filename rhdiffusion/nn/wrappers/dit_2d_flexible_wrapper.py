import torch
from typing import Optional
from diffusers.models import DiTTransformer2DModel

class DiTConcatConditionModel(DiTTransformer2DModel):

    def __init__(
        self,
        sample_size: int = 32,
        sample_channels: Optional[int] = None,
        condition_channels: Optional[int] = None,
        **kwargs,
    ):
        self.sample_channels     = sample_channels or kwargs.get("in_channels", 4)
        self.condition_channels  = condition_channels or 0

        in_channels = self.sample_channels + self.condition_channels
        kwargs.pop("in_channels", None)

        super().__init__(in_channels=in_channels, sample_size=sample_size, **kwargs)

    def forward(self, sample, timestep, encoder_hidden_states=None, class_labels=None, **kwargs,):
        """
        If ``encoder_hidden_states`` is supplied, we concatenate it with the sample
        on the **channel** dimension before running the DiT backbone.

        Args:
            sample: The input sample tensor
            timestep: The diffusion timestep
            encoder_hidden_states: The conditioning input to concatenate with sample
        """
        concat_input = torch.cat([sample, encoder_hidden_states], dim=1)

        # provide zeros if user didn't pass a label
        if class_labels is None:
            class_labels = torch.zeros(concat_input.shape[0], dtype=torch.long, device=concat_input.device,)

        return super().forward(concat_input, timestep, class_labels=class_labels, **kwargs,)

class DiT2DFlexibleWrapper:
    """
    Tiny convenience factory that mirrors your UNet helper.

    ``mode`` in the config determines what gets built:
        - "none"   : plain   ``DiTTransformer2DModel``
        - "concat" : custom  ``DiTConcatConditionModel``

    Cross-attention is **not** currently implemented for DiT in diffusers,
    so any other mode raises.
    """

    SUPPORTED_MODES = {"none", "concat"}

    @classmethod
    def from_config(cls, config: dict):
        # ---- pull bookkeeping options --------------------------------
        mode               = config.pop("mode", "none")
        sample_channels    = config.pop("sample_channels", None)
        condition_channels = config.pop("condition_channels", None)

        # ---- sanity --------------------------------------------------
        if mode not in cls.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. "
                f"Choose from {sorted(cls.SUPPORTED_MODES)}."
            )

        # ---- 'concat'  ----------------------------------------------
        if mode == "concat":
            return DiTConcatConditionModel(
                sample_channels=sample_channels,
                condition_channels=condition_channels,
                **config,
            )

        # ---- 'none' (baseline DiT) ----------------------------------
        # Strip any conditioning-only keys the user might have provided.
        config.pop("condition_channels", None)
        config["in_channels"] = sample_channels  # override / ensure

        return DiTTransformer2DModel(**config)
