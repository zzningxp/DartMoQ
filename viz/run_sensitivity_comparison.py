"""
MSE vs IPE 敏感度对比实验 - 运行脚本

支持两种模式：
1. demo：用合成数据快速测试
2. real：用真实模型权重
"""

import os
import sys

# 添加父目录和当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

import torch
import argparse

from mse_vs_ipe_sensitivity import (
    TurboQuantSensitivityTest,
    create_synthetic_weight
)


def main():
    parser = argparse.ArgumentParser(
        description='TurboQuant: MSE vs IPE 敏感度对比',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['demo', 'real'],
        default='demo',
        help='运行模式：demo(合成数据) / real(真实模型)'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='结果保存目录 (默认: ../plot/mse_vs_ipe)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='运行设备 (默认自动选择)'
    )

    parser.add_argument(
        '--small',
        action='store_true',
        help='用更小的矩阵(更快)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='真实模型路径(real模式)'
    )

    parser.add_argument(
        '--layer_idx',
        type=int,
        default=0,
        help='层索引(real模式)'
    )

    parser.add_argument(
        '--expert_idx',
        type=int,
        default=0,
        help='专家索引(real模式)'
    )

    parser.add_argument(
        '--module_name',
        type=str,
        default='up_proj',
        help='模块名：up_proj / gate_proj / down_proj'
    )

    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo_mode(args)
    else:
        run_real_mode(args)


def run_demo_mode(args):
    print("\n" + "="*60)
    print("运行模式：DEMO (合成数据)")
    print("="*60)

    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"使用设备: {device}")

    if args.save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.save_dir = os.path.join(os.path.dirname(current_dir), 'plot', 'mse_vs_ipe')

    if args.small:
        out_dim, in_dim = 128, 256
        print("\n创建小尺寸合成权重 (128x256)...")
    else:
        out_dim, in_dim = 256, 512
        print("\n创建合成权重 (256x512)...")

    weight, calib = create_synthetic_weight(
        out_dim=out_dim, in_dim=in_dim, seed=42, device=device
    )

    print(f"权重形状: {weight.shape}")
    print(f"校准输入形状: {calib.shape}")

    print("\n初始化测试...")
    tester = TurboQuantSensitivityTest(
        weight_matrix=weight,
        calibration_inputs=calib,
        seed=42,
        device=device
    )

    tester.apply_orthogonal_rotation()
    results = tester.run_all_tests(save_dir=args.save_dir)

    print_summary(results)


def run_real_mode(args):
    if args.model_path is None:
        print("错误：real模式需要指定 --model_path")
        print("提示：也可以用 --mode demo 运行合成数据测试")
        sys.exit(1)

    print("\n" + "="*60)
    print("运行模式：REAL (真实模型)")
    print("="*60)

    if args.save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.save_dir = os.path.join(os.path.dirname(current_dir), 'plot', 'mse_vs_ipe')

    # 尝试从中间结果或检查点加载
    weight, calib = load_from_project(
        args.model_path,
        args.layer_idx,
        args.expert_idx,
        args.module_name,
        args.device
    )

    print(f"权重形状: {weight.shape}")
    print(f"校准输入形状: {calib.shape}")

    print("\n初始化测试...")
    tester = TurboQuantSensitivityTest(
        weight_matrix=weight,
        calibration_inputs=calib,
        seed=42,
        device=args.device
    )

    tester.apply_orthogonal_rotation()
    results = tester.run_all_tests(save_dir=args.save_dir)

    print_summary(results)


def load_from_project(model_path, layer_idx, expert_idx, module_name, device):
    """从项目已有结构加载权重"""
    import os

    # 方式1：尝试从 intermediate_result 加载
    inter_dir = 'intermediate_result'
    if os.path.exists(inter_dir):
        print(f"检查中间结果目录: {inter_dir}")
        weight = try_load_from_intermediate(inter_dir, layer_idx, expert_idx, module_name, device)
        if weight is not None:
            print("从中间结果加载成功")
            in_dim = weight.shape[1]
            calib = torch.randn(256, in_dim, device=device) * 0.1
            return weight, calib

    # 方式2：尝试从 transformers 模型加载
    print("尝试从transformers模型加载...")
    try:
        weight = load_from_transformers(model_path, layer_idx, expert_idx, module_name, device)
        if weight is not None:
            in_dim = weight.shape[1]
            calib = torch.randn(256, in_dim, device=device) * 0.1
            return weight, calib
    except Exception as e:
        print(f"从transformers加载失败: {e}")

    # 方式3：如果都失败，退回到合成数据
    print("\n无法加载真实权重，改用合成数据...")
    return create_synthetic_weight(out_dim=512, in_dim=1024, seed=42, device=device)


def try_load_from_intermediate(inter_dir, layer_idx, expert_idx, module_name, device):
    """尝试从中间结果加载"""
    # 根据项目实际结构调整
    return None


def load_from_transformers(model_path, layer_idx, expert_idx, module_name, device):
    """从transformers模型加载"""
    from transformers import AutoModelForCausalLM

    print(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype='auto',
        trust_remote_code=True,
        device_map='cpu'
    )

    # 尝试找到MoE层
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        layers = model.transformer.layers
    elif hasattr(model, 'lm'):
        layers = model.lm.layers
    else:
        layers = model.layers

    print(f"模型有 {len(layers)} 层")

    layer = layers[layer_idx]

    # 获取MLP权重
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
    else:
        mlp = layer

    # 获取专家权重
    if hasattr(mlp, 'experts'):
        expert = mlp.experts[expert_idx]
        if hasattr(expert, module_name):
            weight = getattr(expert, module_name).weight.data.to(device)
        else:
            weight = expert.weight.data.to(device)
    else:
        if hasattr(mlp, module_name):
            weight = getattr(mlp, module_name).weight.data.to(device)
        else:
            weight = mlp.weight.data.to(device)

    return weight


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*60)
    print("结果摘要")
    print("="*60)

    if 'test1' in results:
        t1 = results['test1']
        print(f"\nTest1 - MSE vs IPE:")
        print(f"  MSE CV: {t1['mse_stats']['cv']:.4f}")
        print(f"  IPE CV: {t1['ipe_stats']['cv']:.4f}")
        ratio = t1['ipe_stats']['cv'] / (t1['mse_stats']['cv'] + 1e-12)
        print(f"  区分度提升: {ratio:.2f}x")

    if 'test2' in results:
        t2 = results['test2']
        print(f"\nTest2 - 旋转效应:")
        print(f"  高敏感神经元: Top10% {t2['high_orig']['top10_ratio']:.3f} -> {t2['high_rot']['top10_ratio']:.3f}")
        print(f"  低敏感神经元: Top10% {t2['low_orig']['top10_ratio']:.3f} -> {t2['low_rot']['top10_ratio']:.3f}")

    if 'test3' in results:
        t3 = results['test3']
        print(f"\nTest3 - DP对比:")
        mse_first = t3['mse_results'][0]['loss']
        ipe_first = t3['ipe_results'][0]['loss']
        improv = (mse_first - ipe_first) / (mse_first + 1e-12) * 100
        print(f"  首点改进: {improv:.1f}%")

    if 'test4' in results:
        t4 = results['test4']
        corr = t4['corr_matrix']
        print(f"\nTest4 - 秩相关:")
        if corr.shape[0] >= 2:
            print(f"  1bit vs 2bit: {corr[0, 1]:.4f}")
            print(f"  1bit vs 4bit: {corr[0, -1]:.4f}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
