#!/usr/bin/env python
"""
完整测试脚本：在真实训练环境中对比 v3 和 v4 的随机状态

用法：
    python scripts/test_v3_v4_random_full.py v3
    python scripts/test_v3_v4_random_full.py v4

然后对比输出：
    diff debug_full_v3.txt debug_full_v4.txt
"""

import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import random
from typing import List, Tuple
from dataclasses import dataclass

# 导入项目模块
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 随机状态工具
# =============================================================================

def get_random_state():
    """获取所有随机状态"""
    return {
        'np': np.random.get_state()[1][0],
        'torch': int(torch.get_rng_state()[:8].sum().item()),
        'random': random.getstate()[1][0],
    }

def print_state(tag: str):
    """打印随机状态"""
    state = get_random_state()
    print(f"  [{tag}] np={state['np']}, torch={state['torch']}, random={state['random']}")


# =============================================================================
# 简单数据集
# =============================================================================

class SimpleDataset(Dataset):
    """简单的文本分类数据集"""
    def __init__(self, tokenizer, num_samples=100, max_length=64):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        # 生成一些简单的样本
        self.texts = [
            "This movie is great and I love it.",
            "This is a terrible film, I hate it.",
            "The weather is nice today.",
            "I feel so sad and depressed.",
            "What a wonderful experience!",
        ] * (num_samples // 5 + 1)
        self.texts = self.texts[:num_samples]
        self.labels = [1, 0, 1, 0, 1] * (num_samples // 5 + 1)
        self.labels = self.labels[:num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),  # 用于 causal LM
        }


# =============================================================================
# 贝叶斯估计器（从优化器复制）
# =============================================================================

class BayesianSubspaceGradient:
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


# =============================================================================
# 测试优化器
# =============================================================================

class TestOptimizer:
    def __init__(self, version: str, model, device, dim=2, num_samples=3, zo_eps=1e-3):
        self.version = version
        self.model = model
        self.device = device
        self.dim = dim
        self.num_samples = num_samples
        self.zo_eps = zo_eps
        self.bayes = BayesianSubspaceGradient(dim)
        self.iteration = 0
        self.learning_rate = 1e-6

        self.named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        print(f"  参数数量: {sum(p.numel() for _, p in self.named_parameters):,}")

    def compute_loss(self, inputs) -> float:
        """计算 loss"""
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels']
            )
            return outputs.loss.item()

    def perturb_params(self, seed: int, scale: float):
        """扰动参数"""
        torch.manual_seed(seed)
        for name, param in self.named_parameters:
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=scale)

    def get_param_hash(self):
        """获取参数的简单哈希（用于检测漂移）"""
        first_param = self.named_parameters[0][1]
        return float(first_param.data[:10].sum().item())

    def evaluate_derivative(self, seeds: List[int], d_sub: np.ndarray, f0: float) -> float:
        """计算方向导数（单侧差分）"""
        epsilon = self.zo_eps

        # 记录扰动前的参数状态
        param_before = self.get_param_hash()

        # 正向扰动
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self.perturb_params(seed, epsilon * d_sub[i])

        # 获取当前 batch 的 inputs（需要从外部传入）
        f_plus = self._current_loss_fn()

        # 恢复
        for i, seed in enumerate(seeds):
            if abs(d_sub[i]) > 1e-10:
                self.perturb_params(seed, -epsilon * d_sub[i])

        # 记录恢复后的参数状态
        param_after = self.get_param_hash()

        # 检测漂移
        if abs(param_before - param_after) > 1e-10:
            print(f"    [漂移检测] 参数漂移! before={param_before:.10f}, after={param_after:.10f}, diff={param_after-param_before:.2e}")

        return (f_plus - f0) / epsilon

    def step(self, inputs):
        """执行一步优化"""
        print(f"\n{'='*70}")
        print(f"[{self.version.upper()}] Iteration {self.iteration}")
        print(f"{'='*70}")

        # 保存当前 inputs 的 loss 计算函数
        self._current_loss_fn = lambda: self.compute_loss(inputs)

        # 1. 打印初始状态
        print_state("step start")

        # 2. 生成 seeds
        print(f"\n  --- 生成 seeds ---")
        print_state("before np.random.randint")
        seeds = [np.random.randint(0, 2**31) for _ in range(self.dim)]
        print(f"  seeds = {seeds}")
        print_state("after np.random.randint")

        # 3. 计算 f0
        print(f"\n  --- 计算 f0 ---")
        print_state("before f0")
        f0 = self.compute_loss(inputs)
        print(f"  f0 = {f0:.6f}")
        print_state("after f0")

        # 4. 重置贝叶斯估计器
        self.bayes.reset()

        # 5. 缓存
        y_cache = {}

        # 6. 采样循环
        for sample_idx in range(self.num_samples):
            print(f"\n  --- Sample {sample_idx} ---")
            print_state(f"sample {sample_idx} start")

            if sample_idx >= self.dim:  # 自适应采样
                # 打印当前协方差矩阵 Σ
                mu, Sigma = self.bayes.get_estimate()
                print(f"  当前 Σ (协方差矩阵):")
                print(f"    [[{Sigma[0,0]:.6f}, {Sigma[0,1]:.6f}],")
                print(f"     [{Sigma[1,0]:.6f}, {Sigma[1,1]:.6f}]]")

                # 计算特征值和特征向量
                eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
                print(f"  特征值: {eigenvalues}")
                print(f"  特征向量:")
                print(f"    v0 = [{eigenvectors[0,0]:.6f}, {eigenvectors[1,0]:.6f}] (λ={eigenvalues[0]:.6f})")
                print(f"    v1 = [{eigenvectors[0,1]:.6f}, {eigenvectors[1,1]:.6f}] (λ={eigenvalues[1]:.6f})")

                d_suggested = self.bayes.suggest_direction()
                print(f"  suggest_direction() 返回: [{d_suggested[0]:.16e}, {d_suggested[1]:.16e}]")

                axis_idx = np.argmax(np.abs(d_suggested))
                d_sub = np.zeros(self.dim)
                d_sub[axis_idx] = 1.0
                print(f"  自适应: axis_idx={axis_idx}, d_sub=[{d_sub[0]:.1f}, {d_sub[1]:.1f}]")

                # 检查 d_suggested 是否为精确的坐标轴方向
                is_exact_axis = (abs(abs(d_suggested[0]) - 1.0) < 1e-6 and abs(d_suggested[1]) < 1e-6) or \
                                (abs(d_suggested[0]) < 1e-6 and abs(abs(d_suggested[1]) - 1.0) < 1e-6)
                print(f"  是否精确坐标轴方向: {is_exact_axis}")

                if self.version == 'v3':
                    # v3: 使用缓存 + 强制坐标轴方向
                    y = y_cache[axis_idx]
                    print(f"  [V3] 使用缓存 y={y:.6f}")
                    print(f"  [V3] 使用 d_sub=[{d_sub[0]:.1f}, {d_sub[1]:.1f}] (强制坐标轴)")
                    print_state("v3 cache")
                else:
                    # v4: 重新计算 + 使用实际 suggest_direction
                    # 关键区别：v4 使用 d_suggested，不是 d_sub
                    print(f"  [V4] 重新计算，使用 d_suggested=[{d_suggested[0]:.6f}, {d_suggested[1]:.6f}]")
                    print_state("v4 before forward")
                    y = self.evaluate_derivative(seeds, d_suggested, f0)  # 使用 d_suggested 而非 d_sub
                    print(f"  [V4] y={y:.6f}")
                    print_state("v4 after forward")
                    d_sub = d_suggested  # v4 贝叶斯更新也用 d_suggested
            else:
                d_sub = np.zeros(self.dim)
                d_sub[sample_idx] = 1.0
                print(f"  坐标轴: d[{sample_idx}]=1")

                print_state("before derivative")
                y = self.evaluate_derivative(seeds, d_sub, f0)
                print(f"  y={y:.6f}")

                # 立即重新计算一次，检验是否一致
                y_check = self.evaluate_derivative(seeds, d_sub, f0)
                if abs(y - y_check) > 1e-6:
                    print(f"  [!!! 不一致 !!!] 首次 y={y:.6f}, 再算 y={y_check:.6f}, diff={y-y_check:.6f}")

                print_state("after derivative")

                y_cache[sample_idx] = y

            self.bayes.update(d_sub, y)
            print_state(f"sample {sample_idx} end")

        # 7. 获取梯度并更新
        g_sub, Sigma_final = self.bayes.get_estimate()
        print(f"\n  --- 更新参数 ---")
        print(f"  g_sub = [{g_sub[0]:.16e}, {g_sub[1]:.16e}]")
        print(f"  最终 Σ (高精度):")
        print(f"    [[{Sigma_final[0,0]:.16e}, {Sigma_final[0,1]:.16e}],")
        print(f"     [{Sigma_final[1,0]:.16e}, {Sigma_final[1,1]:.16e}]]")
        print_state("before param update")

        for i, seed in enumerate(seeds):
            if abs(g_sub[i]) > 1e-10:
                torch.manual_seed(seed)
                for name, param in self.named_parameters:
                    z = torch.normal(mean=0, std=1, size=param.data.size(),
                                   device=param.device, dtype=param.dtype)
                    param.data.add_(z, alpha=-self.learning_rate * g_sub[i])

        print_state("after param update")

        self.iteration += 1
        return f0


# =============================================================================
# 主函数
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python test_v3_v4_random_full.py [v3|v4]")
        print("\n完整测试步骤：")
        print("  1. python scripts/test_v3_v4_random_full.py v3 > debug_full_v3.txt 2>&1")
        print("  2. python scripts/test_v3_v4_random_full.py v4 > debug_full_v4.txt 2>&1")
        print("  3. diff debug_full_v3.txt debug_full_v4.txt")
        sys.exit(1)

    version = sys.argv[1].lower()
    if version not in ['v3', 'v4']:
        print(f"未知版本: {version}")
        sys.exit(1)

    # 设置随机种子
    seed = 42
    print(f"{'='*70}")
    print(f"完整测试: {version.upper()}")
    print(f"{'='*70}")
    print(f"\n设置全局随机种子: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print_state("initial")

    # 加载模型
    model_path = "/mnt/innovator/chl/models/opt-1.3b"
    print(f"\n加载模型: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print_state("before model load")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 和实际实验一致
    ).to(device)
    model.eval()

    print_state("after model load")

    # 创建数据集和 DataLoader
    print(f"\n创建数据集...")
    dataset = SimpleDataset(tokenizer, num_samples=1000, max_length=64)

    # 使用固定的 generator 确保 DataLoader 的随机性可控
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        generator=g
    )

    print_state("after dataloader creation")

    # 创建优化器
    print(f"\n创建优化器 ({version})...")
    optimizer = TestOptimizer(
        version=version,
        model=model,
        device=device,
        dim=2,
        num_samples=3,
        zo_eps=1e-3
    )

    print_state("after optimizer creation")

    # 运行10步训练（快速验证数值精度）
    num_steps = 10
    print(f"\n开始训练 {num_steps} 步...")
    print(f"{'='*70}")

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        # 移动到 device
        batch = {k: v.to(device) for k, v in batch.items()}

        print(f"\n\n>>> DataLoader batch {step} <<<")
        print_state(f"dataloader batch {step}")

        loss = optimizer.step(batch)
        print(f"\n  Batch {step} loss: {loss:.6f}")

    print(f"\n{'='*70}")
    print(f"{version.upper()} 测试完成！")
    print(f"{'='*70}")

    # 打印最终状态
    print(f"\n最终随机状态:")
    print_state("final")


if __name__ == "__main__":
    main()
