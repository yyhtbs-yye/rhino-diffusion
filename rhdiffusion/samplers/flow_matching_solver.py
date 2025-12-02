from copy import deepcopy
import torch

class FlowMatchingSolver:
    """
    Simple ODE solver for Flow Matching models.

    Assumptions:
      - We integrate the ODE
            dx/dt = u_theta(x, t, cond)
        from t = t_min to t = t_max (default: 0 -> 1 - eps_t).

      - Denoiser parameterization (default):
            network(x_t, t, cond) -> x1_hat(x_t, t, cond),
        and the generating velocity field is
            u_theta(x_t, t) = (x1_hat - x_t) / (1 - t)
        (i.e. kappa(t) = t, kappa_dot(t) = 1).

      - Optionally, we can also treat the network output directly
        as velocity by setting prediction_type="velocity".
    """

    def __init__(
        self,
        num_steps: int = 50,
        t_min: float = 0.0,
        t_max: float = 1.0,
        eps_t: float = 1e-3,
        method: str = "heun",
        prediction_type: str = "x0",  # "x0" (denoiser-style) or "velocity"
    ):
        assert num_steps > 0, "num_steps must be > 0"
        assert 0.0 <= t_min < t_max <= 1.0, "t_min/t_max must be in [0,1]"
        assert method.lower() in ["euler", "heun"], "method must be 'euler' or 'heun'"
        assert prediction_type in ["x0", "velocity"], "prediction_type must be 'x0' or 'velocity'"

        self.num_steps = num_steps
        self.t_min = t_min
        self.t_max = t_max
        self.eps_t = eps_t
        self.method = method.lower()
        self.prediction_type = prediction_type

    # --------------------------------------------------------------------- #
    # Config constructor (mirrors DDIMSampler.from_config, but no scheduler)
    # --------------------------------------------------------------------- #
    @classmethod
    def from_config(cls, config: dict):
        """
        Expected config keys (all optional):
          - num_inference_steps / num_steps: int
          - t_min: float in [0, 1]
          - t_max: float in [0, 1]
          - eps_t: float, small margin to avoid t=1 exactly
          - method: "euler" or "heun"
          - prediction_type: "x0" or "velocity"
        """
        cfg = deepcopy(config)

        num_steps = cfg.pop("num_inference_steps", cfg.pop("num_steps", 50))
        t_min = cfg.pop("t_min", 0.0)
        t_max = cfg.pop("t_max", 1.0)
        eps_t = cfg.pop("eps_t", 1e-3)
        method = cfg.pop("method", "heun")
        prediction_type = cfg.pop("prediction_type", "x0")

        # ignore any remaining keys (they might be leftovers from older configs)
        return cls(
            num_steps=num_steps,
            t_min=t_min,
            t_max=t_max,
            eps_t=eps_t,
            method=method,
            prediction_type=prediction_type,
        )

    # --------------------------------------------------------------------- #
    # Main solver
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def solve(self, network, noise, cond=None, seed=None):
        """
        Integrate the learned flow from t_min to t_max-eps_t.

        Args:
            network: callable (x, t, cond) -> output
                     If prediction_type == "x0":
                         output is x1_hat at time t (denoiser form).
                     If prediction_type == "velocity":
                         output is u_theta(x, t, cond) directly.
            noise:   initial sample x_{t_min} (e.g. standard Gaussian).
            cond:    optional conditioning.
            seed:    optional integer for reproducibility; here we only
                     use it to set torch.manual_seed(seed).

        Returns:
            x: final sample at the last time step (≈ data sample).
        """
        x = noise

        if seed is not None:
            torch.manual_seed(seed)

        device = x.device
        dtype = x.dtype

        # we don't integrate exactly to t=1 to avoid division by (1 - t)
        t_start = self.t_min
        t_end = self.t_max - self.eps_t

        # time grid: t_0, ..., t_N  (N = num_steps)
        ts = torch.linspace(t_start, t_end, self.num_steps + 1, device=device, dtype=dtype)

        for i in range(self.num_steps):
            t = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t

            # shape (B,) for the network
            t_batch = t.expand(x.shape[0])

            if cond is not None:
                out_1 = network(x, t_batch, cond)
            else:
                out_1 = network(x, t_batch)

            # convert network output to velocity
            v1 = self._to_velocity(x, out_1, t_batch)

            if self.method == "euler":
                # simple forward Euler
                x = x + dt * v1
            else:
                # Heun's method (2nd-order Runge–Kutta)
                x_euler = x + dt * v1

                t_next_batch = t_next.expand(x.shape[0])
                if cond is not None:
                    out_2 = network(x_euler, t_next_batch, cond)
                else:
                    out_2 = network(x_euler, t_next_batch)

                v2 = self._to_velocity(x_euler, out_2, t_next_batch)
                x = x + 0.5 * dt * (v1 + v2)

        return x

    # --------------------------------------------------------------------- #
    # Helper: convert network output -> velocity field
    # --------------------------------------------------------------------- #
    def _to_velocity(self, x, net_out, t_batch):
        """
        x:        current state, shape (B, C, H, W)
        net_out:  network output (x1_hat or velocity), same shape as x
        t_batch:  shape (B,) with scalar time t for each batch element
        """
        if self.prediction_type == "velocity":
            # network already outputs u_theta(x,t)
            return net_out

        # prediction_type == "x0": denoiser parameterization
        # u_theta(x,t) = (x1_hat - x) / (1 - t)
        # broadcast t -> (B,1,1,1)
        while t_batch.ndim < x.ndim:
            t_batch = t_batch.view(t_batch.shape[0], *([1] * (x.ndim - 1)))

        one_minus_t = 1.0 - t_batch

        # numerically safer: enforce a floor
        if self.eps_t is not None and self.eps_t > 0.0:
            one_minus_t = one_minus_t.clamp_min(self.eps_t)

        v = (net_out - x) / one_minus_t
        return v
