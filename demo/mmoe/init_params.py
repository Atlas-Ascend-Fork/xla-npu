# -*- coding:utf-8 -*-
"""
初始化MMOE模型参数并保存到文件
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from mmoe import create_mmoe, save_params_with_npz


def main():
    # Configuration - 与mmoe.py中的配置保持一致
    # 当前配置: input_dim=4, batch_size=2, num_experts=2, num_tasks=2
    #           expert_dnn_hidden_units=(4, 2), gate_dnn_hidden_units=(2,), tower_dnn_hidden_units=(2,)
    input_dim = 4
    batch_size = 2
    num_experts = 2
    num_tasks = 2

    print("Initializing MMOE model parameters...")
    print(f"Input dim: {input_dim}, Batch size: {batch_size}")
    print(f"Num experts: {num_experts}, Num tasks: {num_tasks}")

    # Create model - 使用更小的隐藏层
    # expert_dnn_hidden_units=(4, 2), gate_dnn_hidden_units=(2,), tower_dnn_hidden_units=(2,)
    model = create_mmoe(
        input_dim=input_dim,
        num_experts=num_experts,
        num_tasks=num_tasks,
        expert_dnn_hidden_units=(4, 2),
        gate_dnn_hidden_units=(2,),
        tower_dnn_hidden_units=(2,),
        dnn_activation='relu',
        task_types=('binary', 'binary'),
    )

    # Initialize parameters (all ones for debugging)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, input_dim))
    key, init_key = jax.random.split(key)
    params = model.init(init_key, dummy_input, train=False)

    # Save parameters
    save_filepath = os.path.join(os.path.dirname(__file__), "model_params_small.npz")
    save_params_with_npz(params, save_filepath)
    print(f"\nParameters saved to: {save_filepath}")


if __name__ == "__main__":
    main()
