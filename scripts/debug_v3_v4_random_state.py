#!/usr/bin/env python
"""
调试脚本：使用真实 OPT-1.3B 模型找出 v3 和 v4 随机状态分叉的位置

用法：
    python scripts/debug_v3_v4_random_state.py v3
    python scripts/debug_v3_v4_random_state.py v4

对比两个版本的输出，找出：
1. 从哪一步开始 seeds 不同
2. 是 np.random 还是 torch 的状态分叉
3. 前向传播消耗了什么随机数
"""

import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# 随机状态工具函数
# ============================================================================

def get_np_state_hash():
    """获取 np.random 状态的简短哈希"""
    state = np.random.get_state()
    return state[1][0]


def get_torch_state_hash():
    """获取 torch 随机状态的简短哈希"""
    state = torch.get_rng_state()
    return int(state[:8].sum().item())


def print_random_state(tag: str):
    """打印当前随机状态"""
    print(f"    [{tag}] np={get_np_state_hash()}, torch={get_torch_state_hash()}")


# ============================================================================
# 简化版贝叶斯估计器
# ============================================================================

class DebugBayesianSubspaceGradient:
    def __init__(self, dim: int, sigma_prior: float = 1.0, sigma_noise: float = 0.1):
        self.dim = dim
        self.sigma_prior = sigma_prior
        self.sigma_noise = sigma_noise
        self.reset()

    def reset(self):
        self.mu = np.zeros(self.dim)
        self.Sigma = self.sigma_prior**2 * np.eye(self.dim)

    def update(self, d: np.ndarray, y: float):
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            return

        Sigma_d = self.Sigma @ d
        denominator = d @ Sigma_d + self.sigma_noise**2 * d_norm**2
        K = Sigma_d / denominator

        innovation = y - d @ self.mu
        self.mu = self.mu + K * innovation
        self.Sigma = self.Sigma - np.outer(K, d @ self.Sigma)
        self.Sigma = (self.Sigma + self.Sigma.T) / 2

    def suggest_direction(self) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(self.Sigma)
        return eigenvectors[:, -1]

    def get_estimate(self):
        return self.mu.copy(), self.Sigma.copy()


# ============================================================================
# 调试优化器（使用真实模型）
# ============================================================================

class DebugOptimizer:
    def __init__(self, version: str, model, tokenizer, device,
                 dim: int = 2, num_samples: int = 3, zo_eps: float = 1e-3):
        self.version = version
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dim = dim
        self.num_samples = num_samples
        self.zo_eps = zo_eps
        self.bayes = DebugBayesianSubspaceGradient(dim)
        self.iteration = 0

        # 获取参数列表
        self.named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.n_params = sum(p.numel() for _, p in self.named_parameters)
        print(f"  模型参数量: {self.n_params:,}")

    def _compute_loss(self, tag: str) -> float:
        """计算 loss，并打印随机状态变化"""
        np_before = get_np_state_hash()
        torch_before = get_torch_state_hash()

        # 创建一个简单的输入
        inputs = self.tokenizer("Hello, how are you?", return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

        np_after = get_np_state_hash()
        torch_after = get_torch_state_hash()

        np_changed = np_before != np_after
        torch_changed = torch_before != torch_after

        print(f"    [{tag}] loss={loss:.6f}")
        print(f"    [{tag}] np: {np_before} -> {np_after} (changed={np_changed})")
        print(f"    [{tag}] torch: {torch_before} -> {torch_after} (changed={torch_changed})")

        return loss

    def _perturb_params_with_seed(self, seed: int, scale: float):
        """用种子扰动参数"""
        torch.manual_seed(seed)
        for name, param in self.named_parameters:
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=scale)

    def _evaluate_directional_derivative(self, seeds: List[int], d_sub: np.ndarray,
                                          f0: float, tag: str) -> float:
        """计算方向导数（单侧差分）"""
        print(f"    [deriv {tag}] 开始计算方向导数")
        print_random_state(f"deriv {tag} start")

        epsilon = self.zo_eps

        # 正向扰动
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self._perturb_params_with_seed(seed, epsilon * d_sub[i])

        print_random_state(f"deriv {tag} after perturb")

        # 计算 f_plus
        f_plus = self._compute_loss(f"f_plus {tag}")

        # 恢复
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self._perturb_params_with_seed(seed, -epsilon * d_sub[i])

        print_random_state(f"deriv {tag} after restore")

        y = (f_plus - f0) / epsilon
        print(f"    [deriv {tag}] y = (f_plus - f0) / eps = {y:.6f}")

        return y

    def optimize_step(self):
        """执行一步优化"""
        print(f"\n{'='*70}")
        print(f"[{self.version.upper()}] Step {self.iteration}")
        print(f"{'='*70}")

        # 打印当前随机状态
        print(f"\n  === 初始状态 ===")
        print(f"  np.random: {get_np_state_hash()}")
        print(f"  torch: {get_torch_state_hash()}")

        # 生成 seeds
        print(f"\n  === 生成 seeds ===")
        print_random_state("before seeds")
        seeds = [np.random.randint(0, 2**31) for _ in range(self.dim)]
        print(f"  seeds: {seeds}")
        print_random_state("after seeds")

        # 重置贝叶斯估计器
        self.bayes.reset()

        # 计算 f0
        print(f"\n  === 计算 f0 ===")
        f0 = self._compute_loss("f0")

        # 缓存（v3 用）
        y_cache = {}

        # 采样循环
        for sample_idx in range(self.num_samples):
            print(f"\n  === Sample {sample_idx} ===")
            print_random_state(f"sample {sample_idx} start")

            if sample_idx >= self.dim:  # 自适应采样
                d_suggested = self.bayes.suggest_direction()
                axis_idx = np.argmax(np.abs(d_suggested))
                d_sub = np.zeros(self.dim)
                d_sub[axis_idx] = 1.0
                print(f"  自适应采样: suggest_direction -> axis {axis_idx}")

                if self.version == 'v3':
                    # v3: 使用缓存
                    y = y_cache[axis_idx]
                    print(f"  [V3] 使用缓存 y={y:.6f} (跳过前向传播)")
                    print_random_state("v3 after cache")
                else:
                    # v4: 重新计算
                    print(f"  [V4] 重新计算 (执行前向传播)")
                    y = self._evaluate_directional_derivative(seeds, d_sub, f0, f"adaptive_{sample_idx}")
            else:
                # 基础：坐标轴方向
                d_sub = np.zeros(self.dim)
                d_sub[sample_idx] = 1.0
                print(f"  坐标轴采样: d[{sample_idx}] = 1")

                y = self._evaluate_directional_derivative(seeds, d_sub, f0, f"axis_{sample_idx}")
                y_cache[sample_idx] = y
                print(f"  已缓存 y_cache[{sample_idx}] = {y:.6f}")

            # 贝叶斯更新
            self.bayes.update(d_sub, y)
            print_random_state(f"sample {sample_idx} end")

        # 获取梯度估计
        g_sub, _ = self.bayes.get_estimate()

        print(f"\n  === 结果 ===")
        print(f"  g_sub: {g_sub}")
        print(f"  最终 np.random: {get_np_state_hash()}")
        print(f"  最终 torch: {get_torch_state_hash()}")

        self.iteration += 1
        return g_sub


# ============================================================================
# 主函数
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python debug_v3_v4_random_state.py [v3|v4]")
        print("\n运行两次，分别用 v3 和 v4，然后对比输出：")
        print("  python scripts/debug_v3_v4_random_state.py v3 > debug_v3.txt 2>&1")
        print("  python scripts/debug_v3_v4_random_state.py v4 > debug_v4.txt 2>&1")
        print("  diff debug_v3.txt debug_v4.txt")
        sys.exit(1)

    version = sys.argv[1].lower()
    if version not in ['v3', 'v4']:
        print(f"未知版本: {version}")
        print("请使用 v3 或 v4")
        sys.exit(1)

    # 设置随机种子
    seed = 42
    print(f"{'='*70}")
    print(f"调试 {version.upper()} 版本的随机状态")
    print(f"{'='*70}")
    print(f"\n设置随机种子: {seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"初始 np.random: {get_np_state_hash()}")
    print(f"初始 torch: {get_torch_state_hash()}")

    # 加载模型
    model_path = "/mnt/innovator/chl/models/opt-1.3b"
    print(f"\n加载模型: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print_random_state("before model load")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 使用 fp32 以便调试
    ).to(device)
    model.eval()

    print_random_state("after model load")

    # 创建优化器
    print(f"\n创建优化器...")
    optimizer = DebugOptimizer(
        version=version,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dim=2,
        num_samples=3,
        zo_eps=1e-3
    )

    print_random_state("after optimizer init")

    # 运行几步
    num_steps = 3
    print(f"\n运行 {num_steps} 步优化...")

    for step in range(num_steps):
        g = optimizer.optimize_step()

    print(f"\n{'='*70}")
    print(f"{version.upper()} 调试完成！")
    print(f"{'='*70}")
    print(f"\n对比方法:")
    print(f"  1. 运行: python scripts/debug_v3_v4_random_state.py v3 > debug_v3.txt 2>&1")
    print(f"  2. 运行: python scripts/debug_v3_v4_random_state.py v4 > debug_v4.txt 2>&1")
    print(f"  3. 对比: diff debug_v3.txt debug_v4.txt")
    print(f"\n关注点:")
    print(f"  - seeds 从哪一步开始不同")
    print(f"  - 是 np.random 还是 torch 先分叉")
    print(f"  - 前向传播是否改变了随机状态")


if __name__ == "__main__":
    main()
