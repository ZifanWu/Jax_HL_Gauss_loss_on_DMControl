import numpy as np
import jax.numpy as jnp
from jax import random
from jax.nn import softmax
import matplotlib.pyplot as plt


def sample_normal_and_softmax(M, mean=0.0, std=1.0, key=None, use_jax=True):
    """
    从正态分布中采样M+1次，然后应用softmax函数
    
    Args:
        M: 采样次数-1 (总共采样M+1次)
        mean: 正态分布的均值
        std: 正态分布的标准差
        key: JAX随机数生成器的key (如果使用JAX)
        use_jax: 是否使用JAX实现
    
    Returns:
        softmax_values: 应用softmax后的概率分布序列
        raw_samples: 原始采样值
    """
    if use_jax and key is not None:
        key, subkey = random.split(key)
        # 使用JAX实现
        raw_samples = random.normal(subkey, shape=(M+1,)) * std + mean
        softmax_values = softmax(raw_samples)
        return softmax_values, key
    else:
        # 使用NumPy实现
        raw_samples = np.random.normal(mean, std, size=M+1)
        # 计算softmax: exp(x_i) / sum(exp(x_j) for all j)
        exp_values = np.exp(raw_samples - np.max(raw_samples))  # 减去最大值避免数值溢出
        softmax_values = exp_values / np.sum(exp_values)
    
        return softmax_values


def demo_normal_sampling_softmax():
    """演示从正态分布采样并应用softmax的过程"""
    
    # 设置参数
    M = 10  # 总共采样M+1=11次
    mean = 0.0
    std = 1.0
    
    print(f"从正态分布 N({mean}, {std}²) 中采样 {M+1} 次，然后应用softmax")
    print("=" * 60)
    
    # NumPy实现
    print("\n使用NumPy实现:")
    np_softmax, np_raw = sample_normal_and_softmax(M, mean, std, use_jax=False)
    
    print(f"原始采样值: {np_raw}")
    print(f"Softmax后的值: {np_softmax}")
    print(f"Softmax值之和: {np.sum(np_softmax):.6f}")
    
    # JAX实现
    print("\n使用JAX实现:")
    key = random.PRNGKey(42)
    jax_softmax, jax_raw = sample_normal_and_softmax(M, mean, std, key=key, use_jax=True)
    
    print(f"原始采样值: {jax_raw}")
    print(f"Softmax后的值: {jax_softmax}")
    print(f"Softmax值之和: {jnp.sum(jax_softmax):.6f}")
    
    return np_softmax, np_raw, jax_softmax, jax_raw


def visualize_results(np_softmax, np_raw, jax_softmax, jax_raw):
    """可视化采样和softmax结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 绘制原始采样值
    axes[0, 0].bar(range(len(np_raw)), np_raw)
    axes[0, 0].set_title('NumPy: 原始正态分布采样值')
    axes[0, 0].set_xlabel('样本索引')
    axes[0, 0].set_ylabel('采样值')
    
    axes[0, 1].bar(range(len(jax_raw)), jax_raw)
    axes[0, 1].set_title('JAX: 原始正态分布采样值')
    axes[0, 1].set_xlabel('样本索引')
    axes[0, 1].set_ylabel('采样值')
    
    # 绘制softmax后的值
    axes[1, 0].bar(range(len(np_softmax)), np_softmax)
    axes[1, 0].set_title('NumPy: Softmax后的概率分布')
    axes[1, 0].set_xlabel('样本索引')
    axes[1, 0].set_ylabel('概率')
    
    axes[1, 1].bar(range(len(jax_softmax)), jax_softmax)
    axes[1, 1].set_title('JAX: Softmax后的概率分布')
    axes[1, 1].set_xlabel('样本索引')
    axes[1, 1].set_ylabel('概率')
    
    plt.tight_layout()
    plt.savefig('/home/zifan/Documents/Jax_HL_Gauss_loss_on_DMControl/normal_sampling_softmax_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def compare_different_parameters():
    """比较不同参数下的结果"""
    
    print("\n" + "=" * 60)
    print("比较不同标准差下的结果:")
    print("=" * 60)
    
    M = 5
    means = [0.0, 1.0]
    stds = [0.5, 1.0, 2.0]
    
    for mean in means:
        for std in stds:
            softmax_vals, raw_vals = sample_normal_and_softmax(M, mean, std, use_jax=False)
            print(f"\n均值={mean}, 标准差={std}:")
            print(f"  原始值: {raw_vals}")
            print(f"  Softmax: {softmax_vals}")
            print(f"  最大值索引: {np.argmax(softmax_vals)}")


if __name__ == "__main__":
    # 运行演示
    np_softmax, np_raw, jax_softmax, jax_raw = demo_normal_sampling_softmax()
    
    # 可视化结果
    try:
        visualize_results(np_softmax, np_raw, jax_softmax, jax_raw)
    except ImportError:
        print("\n注意: matplotlib未安装，跳过可视化部分")
    
    # 比较不同参数
    compare_different_parameters()
    
    print("\n" + "=" * 60)
    print("总结:")
    print("1. 从正态分布采样得到M+1个值")
    print("2. 对这些值应用softmax函数得到概率分布")
    print("3. Softmax确保所有值的和为1，且所有值都为正")
    print("4. 较大的原始值在softmax后会得到较大的概率")
    print("=" * 60)
