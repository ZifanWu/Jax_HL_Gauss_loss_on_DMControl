#!/usr/bin/env python3
"""
测试JAX随机数生成修复是否有效
"""

import jax
from jax import random
import jax.numpy as jnp

def test_jax_random_fixes():
    """测试JAX随机数生成函数"""
    print("测试JAX随机数生成修复...")
    
    # 测试随机数生成
    key = random.PRNGKey(42)
    
    # 测试uniform函数
    try:
        beta = random.uniform(key, (), 0.5, 1.0)
        print(f"✓ random.uniform 工作正常: {beta}")
    except Exception as e:
        print(f"✗ random.uniform 失败: {e}")
    
    # 测试randint函数
    try:
        M = random.randint(key, (1,), 2, 6)[0]
        print(f"✓ random.randint 工作正常: {M}")
    except Exception as e:
        print(f"✗ random.randint 失败: {e}")
    
    # 测试split函数
    try:
        key1, key2 = random.split(key)
        print(f"✓ random.split 工作正常")
    except Exception as e:
        print(f"✗ random.split 失败: {e}")

def test_normal_sampling_import():
    """测试normal_sampling_softmax导入"""
    print("\n测试normal_sampling_softmax导入...")
    
    try:
        from jaxrl.agents.drq.normal_sampling_softmax import sample_normal_and_softmax
        print("✓ 导入成功")
        
        # 测试函数调用
        key = random.PRNGKey(42)
        alphas, raw_samples = sample_normal_and_softmax(5, mean=0.0, std=1.0, key=key)
        print(f"✓ 函数调用成功: alphas={alphas}")
        
    except Exception as e:
        print(f"✗ 导入或函数调用失败: {e}")

if __name__ == "__main__":
    test_jax_random_fixes()
    test_normal_sampling_import()
    print("\n测试完成!")
