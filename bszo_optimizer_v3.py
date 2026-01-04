"""
Bayesian Subspace Zeroth-Order (BSZO) Optimizer V3 for LLM Fine-tuning

Core idea: Bayesian gradient estimation in a random k-dim subspace
- Treat gradient as a random variable with prior N(0, sigma_prior^2 I)
- Each directional derivative is a noisy observation: y = d^T g + noise
- Update posterior via Kalman filter
- Use posterior mean for gradient descent
"""

import numpy as np
import torch
from typing import Callable, List, Tuple, Optional


class BayesianSubspaceGradient:
    """
    Bayesian gradient estimator in k-dim subspace.
    Maintains posterior N(mu, Sigma) with adaptive noise estimation.
    """

    def __init__(self, dim: int, sigma_prior: float = 1.0, sigma_noise: float = 0.1,
                 adaptive_noise: bool = True, noise_ema_alpha: float = 0.1):
        self.dim = dim
        self.sigma_prior = sigma_prior
        self.sigma_noise_init = sigma_noise
        self.sigma_noise = sigma_noise

        # Adaptive noise estimation
        self.adaptive_noise = adaptive_noise
        self.noise_ema_alpha = noise_ema_alpha
        self.residual_sq_ema = sigma_noise ** 2

        self.reset()

    def reset(self):
        """Reset to prior (keep noise estimate across batches)"""
        self.mu = np.zeros(self.dim)
        self.Sigma = self.sigma_prior**2 * np.eye(self.dim)
        self.n_observations = 0

    def update(self, d: np.ndarray, y: float):
        """
        Update posterior after observing directional derivative.

        Args:
            d: sampling direction in subspace
            y: observed directional derivative y ≈ d^T g
        """
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            return self.mu, self.Sigma

        # Adaptive noise estimation
        if self.adaptive_noise:
            y_pred = d @ self.mu
            residual = y - y_pred
            residual_normalized = residual / d_norm
            self.residual_sq_ema = (1 - self.noise_ema_alpha) * self.residual_sq_ema + \
                                    self.noise_ema_alpha * residual_normalized ** 2
            self.sigma_noise = max(self.sigma_noise_init * 0.1, np.sqrt(self.residual_sq_ema))

        # Kalman update
        Sigma_d = self.Sigma @ d
        denominator = d @ Sigma_d + self.sigma_noise**2 * d_norm**2
        K = Sigma_d / denominator

        innovation = y - d @ self.mu
        self.mu = self.mu + K * innovation
        self.Sigma = self.Sigma - np.outer(K, d @ self.Sigma)

        # Keep symmetric positive definite
        self.Sigma = (self.Sigma + self.Sigma.T) / 2
        self.Sigma = self.Sigma + 1e-10 * np.eye(self.dim)

        self.n_observations += 1
        return self.mu, self.Sigma

    def get_sigma_noise(self) -> float:
        return self.sigma_noise

    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mu.copy(), self.Sigma.copy()

    def get_total_uncertainty(self) -> float:
        return np.trace(self.Sigma)

    def get_confidence(self) -> float:
        initial = self.dim * self.sigma_prior**2
        current = np.trace(self.Sigma)
        return max(0.0, min(1.0, 1.0 - current / initial))

    def suggest_direction(self) -> np.ndarray:
        """Return max uncertainty direction (largest eigenvector of Sigma)"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.Sigma)
        return eigenvectors[:, -1]


class BSZOOptimizer:
    """
    BSZO: Bayesian Subspace Zeroth-Order optimizer for LLM fine-tuning.

    - Works in k-dim random subspace (memory efficient)
    - Bayesian gradient estimation with Kalman filter
    - Supports adaptive sampling and one-sided finite difference
    """

    def __init__(self,
                 named_parameters: List[Tuple[str, torch.nn.Parameter]],
                 loss_fn: Callable,
                 args,
                 eval_fn: Optional[Callable] = None):

        self.named_parameters = named_parameters
        self.loss_fn = loss_fn
        self.args = args
        self.eval_fn = eval_fn

        # Compute param info
        self.param_shapes = []
        self.param_sizes = []
        self.n = 0
        for name, param in named_parameters:
            self.param_shapes.append(param.shape)
            size = param.numel()
            self.param_sizes.append(size)
            self.n += size

        # Core hyperparams
        self.zo_eps = getattr(args, 'zo_eps', 1e-3)
        self.learning_rate = getattr(args, 'learning_rate', 1e-7)
        self.fixed_subspace_dim = getattr(args, 'bszo_fixed_subspace_dim', 2)

        # Auto sigma: scale with sqrt(n)
        auto_sigma = np.sqrt(self.n / 1e6)

        # Bayesian params
        sigma_prior_arg = getattr(args, 'bayesian_sigma_prior', 'auto')
        sigma_noise_arg = getattr(args, 'bayesian_sigma_noise', 'auto')

        if sigma_prior_arg == 'auto' or sigma_prior_arg is None:
            self.sigma_prior = auto_sigma
        else:
            self.sigma_prior = float(sigma_prior_arg)

        if sigma_noise_arg == 'auto' or sigma_noise_arg is None:
            self.sigma_noise = auto_sigma
        else:
            self.sigma_noise = float(sigma_noise_arg)

        self.num_samples = getattr(args, 'bayesian_num_samples', 3)
        self.adaptive_sampling = getattr(args, 'bayesian_adaptive_sampling', True)
        self.one_sided = getattr(args, 'bayesian_one_sided', True)
        self.adaptive_noise = getattr(args, 'bayesian_adaptive_noise', True)
        self.noise_ema_alpha = getattr(args, 'bayesian_noise_ema_alpha', 0.1)

        # Bayesian estimator
        self.bayes = BayesianSubspaceGradient(
            dim=self.fixed_subspace_dim,
            sigma_prior=self.sigma_prior,
            sigma_noise=self.sigma_noise,
            adaptive_noise=self.adaptive_noise,
            noise_ema_alpha=self.noise_ema_alpha
        )

        # State
        self.iteration = 0
        self.loss_history = []
        self.grad_norm_history = []
        self.confidence_history = []
        self.val_best = float('inf')
        self._last_info = None

        # Device
        self.device = None
        for name, param in named_parameters:
            if self.device is None:
                self.device = param.device
                break
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Print config
        fwd_count = 1 + self.num_samples if self.one_sided else 1 + 2 * self.num_samples
        print(f"[BSZO-V3] n={self.n:.2e}, dim={self.fixed_subspace_dim}, samples={self.num_samples}")
        print(f"[BSZO-V3] lr={self.learning_rate:.2e}, eps={self.zo_eps:.2e}, fwd/step={fwd_count}")

    def params_to_vector(self) -> np.ndarray:
        param_list = []
        for name, param in self.named_parameters:
            param_list.append(param.data.flatten().cpu())
        return torch.cat(param_list).numpy()

    def vector_to_params(self, x: np.ndarray):
        offset = 0
        for (name, param), shape, size in zip(
            self.named_parameters, self.param_shapes, self.param_sizes):
            param.data.copy_(
                torch.from_numpy(x[offset:offset+size])
                .reshape(shape)
                .to(param.device, dtype=param.dtype)
            )
            offset += size

    def _perturb_params_with_seed(self, seed: int, scale: float):
        """Perturb params: theta += scale * z (z generated from seed)"""
        torch.manual_seed(seed)
        for name, param in self.named_parameters:
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=scale)

    def _compute_inner_product_with_seed(self, seed: int, g_coeff: float) -> None:
        """Apply gradient step: theta -= lr * g_coeff * z"""
        torch.manual_seed(seed)
        for name, param in self.named_parameters:
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=-self.learning_rate * g_coeff)

    def _evaluate_directional_derivative_with_seeds(self, model, inputs,
                                                      seeds: List[int],
                                                      d_sub: np.ndarray) -> float:
        """Central difference: (f(x+eps*d) - f(x-eps*d)) / (2*eps)"""
        epsilon = self.zo_eps

        # Forward perturbation
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self._perturb_params_with_seed(seed, epsilon * d_sub[i])
        f_plus = self.loss_fn(model, inputs).item()

        # Backward perturbation
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self._perturb_params_with_seed(seed, -2 * epsilon * d_sub[i])
        f_minus = self.loss_fn(model, inputs).item()

        # Restore
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self._perturb_params_with_seed(seed, epsilon * d_sub[i])

        return (f_plus - f_minus) / (2 * epsilon)

    def _evaluate_directional_derivative_one_sided(self, model, inputs,
                                                    seeds: List[int],
                                                    d_sub: np.ndarray,
                                                    f0: float) -> float:
        """One-sided difference: (f(x+eps*d) - f(x)) / eps"""
        epsilon = self.zo_eps

        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self._perturb_params_with_seed(seed, epsilon * d_sub[i])
        f_plus = self.loss_fn(model, inputs).item()

        # Restore
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self._perturb_params_with_seed(seed, -epsilon * d_sub[i])

        return (f_plus - f0) / epsilon

    def _bayesian_gradient_estimation_efficient(self, model, inputs,
                                                 seeds: List[int]) -> Tuple[np.ndarray, float, dict]:
        """Bayesian gradient estimation (memory efficient)"""
        dim = len(seeds)
        self.bayes.reset()

        f0 = self.loss_fn(model, inputs).item()
        y_cache = {}

        observations = []
        for sample_idx in range(self.num_samples):
            if self.adaptive_sampling and sample_idx >= dim:
                # Adaptive: sample along max uncertainty direction
                d_suggested = self.bayes.suggest_direction()
                axis_idx = np.argmax(np.abs(d_suggested))
                d_sub = np.zeros(dim)
                d_sub[axis_idx] = 1.0
                y = y_cache[axis_idx]
            else:
                # Basic: sample along coordinate axes
                d_sub = np.zeros(dim)
                d_sub[sample_idx % dim] = 1.0

                if self.one_sided:
                    y = self._evaluate_directional_derivative_one_sided(model, inputs, seeds, d_sub, f0)
                else:
                    y = self._evaluate_directional_derivative_with_seeds(model, inputs, seeds, d_sub)

                y_cache[sample_idx % dim] = y

            self.bayes.update(d_sub, y)
            observations.append({'d_sub': d_sub.copy(), 'y': y})

        g_sub, Sigma = self.bayes.get_estimate()
        confidence = self.bayes.get_confidence()

        info = {
            'f0': f0,
            'g_sub': g_sub.copy(),
            'g_sub_norm': np.linalg.norm(g_sub),
            'confidence': confidence,
            'sigma_noise': self.bayes.get_sigma_noise(),
        }

        return g_sub, confidence, info

    def optimize_on_batch(self, model, inputs, maxiter: int = 1) -> float:
        """
        Single batch optimization.
        1. Generate random seeds for subspace basis
        2. Bayesian gradient estimation via sampling
        3. Gradient descent: theta -= lr * sum(g_i * z_i)
        """
        fx = None

        for _ in range(maxiter):
            seeds = [np.random.randint(0, 2**31) for _ in range(self.fixed_subspace_dim)]
            g_sub, confidence, info = self._bayesian_gradient_estimation_efficient(model, inputs, seeds)

            # Gradient descent update
            for i, seed in enumerate(seeds):
                if abs(g_sub[i]) > 1e-10:
                    self._compute_inner_product_with_seed(seed, g_sub[i])

            fx = info['f0']
            self.grad_norm_history.append(info['g_sub_norm'])
            self.confidence_history.append(confidence)
            self._last_info = info

        self.loss_history.append(fx)

        # Trim history
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
            self.grad_norm_history = self.grad_norm_history[-100:]
            self.confidence_history = self.confidence_history[-100:]

        self.iteration += 1
        return fx

    def get_monitoring_metrics(self):
        return {'zo/iteration': self.iteration}

    def check_early_stop_and_adjust_tr(self, val_loss: float) -> Tuple[bool, bool]:
        if val_loss < self.val_best:
            self.val_best = val_loss
        return False, False

    def step(self, model, inputs) -> float:
        maxiter = getattr(self.args, 'bszo_iter_per_step', 1)
        return self.optimize_on_batch(model, inputs, maxiter=maxiter)
