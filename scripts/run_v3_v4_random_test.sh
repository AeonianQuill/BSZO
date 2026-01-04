#!/bin/bash
# =============================================================================
# 一键测试 v3 vs v4 随机状态差异
# =============================================================================

cd /mnt/innovator/chl/zo-bench

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/innovator/chl/huggingface
export TRANSFORMERS_OFFLINE=1

echo "============================================="
echo "V3 vs V4 随机状态完整测试"
echo "============================================="

# 清理旧文件
rm -f debug_full_v3.txt debug_full_v4.txt

# 运行 v3
echo ""
echo "[1/3] 运行 V3 测试..."
python scripts/test_v3_v4_random_full.py v3 > debug_full_v3.txt 2>&1
echo "  V3 完成，输出: debug_full_v3.txt"

# 运行 v4
echo ""
echo "[2/3] 运行 V4 测试..."
python scripts/test_v3_v4_random_full.py v4 > debug_full_v4.txt 2>&1
echo "  V4 完成，输出: debug_full_v4.txt"

# 对比
echo ""
echo "[3/3] 对比差异..."
echo "============================================="
echo ""

# 统计差异行数
DIFF_COUNT=$(diff debug_full_v3.txt debug_full_v4.txt | grep -c "^[<>]")
echo "差异行数: ${DIFF_COUNT}"
echo ""

# 显示关键差异（seeds 和随机状态）
echo "=== Seeds 差异 ==="
diff debug_full_v3.txt debug_full_v4.txt | grep -E "seeds|np=|torch="
echo ""

# 显示完整 diff
echo "=== 完整差异 ==="
diff debug_full_v3.txt debug_full_v4.txt

echo ""
echo "============================================="
echo "测试完成！"
echo ""
echo "详细输出文件："
echo "  - debug_full_v3.txt"
echo "  - debug_full_v4.txt"
echo ""
echo "如果 seeds 在某一步开始不同，说明之前的步骤消耗了不同数量的随机数"
echo "============================================="
