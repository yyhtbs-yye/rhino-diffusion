from copy import deepcopy
import torch
from rhdiffusion.samplers.flow_matching_solver import FlowMatchingSolver

class JiTSolver(FlowMatchingSolver):

    @torch.no_grad()
    def solve(self, network, noise, cond=None, seed=None):
        """
        Integrate the learned flow from t_min to t_max.

        Args:
            network: callable (x, t, cond) -> output
                    If prediction_type == "x0":
                        output is x1_hat at time t (denoiser form).
                    If prediction_type == "velocity":
                        output is u_theta(x, t, cond) directly.
            noise:   initial sample x_{t_min} (e.g. standard Gaussian).
                    # [JiT-compat] For exact JiT, pass noise_scale * randn_like(...)
            cond:    optional conditioning (ignored in unconditional JiT).
            seed:    optional integer for reproducibility; here we only
                    use it to set torch.manual_seed(seed).

        Returns:
            x at t = t_max (default 1.0), evolved by the ODE.
        """
        x = noise

        if seed is not None:
            torch.manual_seed(seed)

        device = x.device
        dtype = x.dtype

        # [JiT-compat] JiT integrates exactly from 0.0 -> 1.0 and
        # relies on a clamp in the denominator (t_eps) to avoid div by 0.
        # So we go all the way to t_max and only use eps_t inside _to_velocity.
        t_start = self.t_min
        t_end = self.t_max  # was: self.t_max - self.eps_t

        # time grid: t_0, ..., t_N  (N = num_steps), like JiT:
        # torch.linspace(0.0, 1.0, steps+1)
        ts = torch.linspace(t_start, t_end, self.num_steps + 1,
                            device=device, dtype=dtype)

        for i in range(self.num_steps):
            t = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t

            # expand scalar t -> (B,) to match network's expected t shape
            t_batch = t.expand(x.shape[0])

            # network forward
            if cond is not None:
                out_1 = network(x, t_batch, cond)
            else:
                out_1 = network(x, t_batch)

            # convert network output to velocity
            v1 = self._to_velocity(x, out_1, t_batch)

            # [JiT-compat] JiT uses:
            #   - Euler if method == "euler"
            #   - Heun for all but the very last step when method == "heun",
            #     and Euler on the last step.
            use_heun = (self.method == "heun") and (i < self.num_steps - 1)

            if not use_heun:
                # Euler (used for all steps if method="euler",
                # and for the final step if method="heun")
                x = x + dt * v1
            else:
                # Heun's method (2nd-order Runge–Kutta), as in JiT
                x_euler = x + dt * v1

                t_next_batch = t_next.expand(x.shape[0])
                if cond is not None:
                    out_2 = network(x_euler, t_next_batch, cond)
                else:
                    out_2 = network(x_euler, t_next_batch)

                v2 = self._to_velocity(x_euler, out_2, t_next_batch)
                x = x + 0.5 * dt * (v1 + v2)

        return x

    def _to_velocity(self, x, net_out, t_batch):
        """
        x:        current state, shape (B, C, H, W)
        net_out:  network output (x1_hat or velocity), same shape as x
        t_batch:  shape (B,), scalar time per batch element

        Returns:
            velocity field u_theta(x, t), same shape as x.
        """
        if self.prediction_type == "velocity":
            # network already predicts the velocity field
            return net_out

        # prediction_type == "x0": denoiser parameterization
        # u_theta(x,t) = (x1_hat - x) / (1 - t)
        # broadcast t -> (B,1,1,1)
        while t_batch.ndim < x.ndim:
            t_batch = t_batch.view(t_batch.shape[0], *([1] * (x.ndim - 1)))

        one_minus_t = 1.0 - t_batch

        # [JiT-compat] This is the analogue of JiT's (1 - t).clamp_min(t_eps):
        # eps_t should match the t_eps used in the original Denoiser.
        if self.eps_t is not None and self.eps_t > 0.0:
            one_minus_t = one_minus_t.clamp_min(self.eps_t)

        v = (net_out - x) / one_minus_t
        return v
