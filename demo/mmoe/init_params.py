# -*- coding:utf-8 -*-
"""
Initialize MMOE model parameters and save to file
Automatically reads configuration from model.py
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Import configuration from model.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    create_mmoe,
    save_params_with_npz,
    CONFIG_INPUT_DIM,
    CONFIG_BATCH_SIZE,
    CONFIG_NUM_EXPERTS,
    CONFIG_NUM_TASKS,
    CONFIG_EXPERT_DNN_HIDDEN_UNITS,
    CONFIG_GATE_DNN_HIDDEN_UNITS,
    CONFIG_TOWER_DNN_HIDDEN_UNITS,
    CONFIG_DNN_ACTIVATION,
    CONFIG_TASK_TYPES,
    CONFIG_MODEL_PARAMS_FILE,
)


def main():
    print("Initializing MMOE model parameters...")
    print(f"Input dim: {CONFIG_INPUT_DIM}, Batch size: {CONFIG_BATCH_SIZE}")
    print(f"Num experts: {CONFIG_NUM_EXPERTS}, Num tasks: {CONFIG_NUM_TASKS}")
    print(f"Expert hidden units: {CONFIG_EXPERT_DNN_HIDDEN_UNITS}")
    print(f"Gate hidden units: {CONFIG_GATE_DNN_HIDDEN_UNITS}")
    print(f"Tower hidden units: {CONFIG_TOWER_DNN_HIDDEN_UNITS}")

    # Create model with configuration from model.py
    model = create_mmoe(
        input_dim=CONFIG_INPUT_DIM,
        num_experts=CONFIG_NUM_EXPERTS,
        num_tasks=CONFIG_NUM_TASKS,
        expert_dnn_hidden_units=CONFIG_EXPERT_DNN_HIDDEN_UNITS,
        gate_dnn_hidden_units=CONFIG_GATE_DNN_HIDDEN_UNITS,
        tower_dnn_hidden_units=CONFIG_TOWER_DNN_HIDDEN_UNITS,
        dnn_activation=CONFIG_DNN_ACTIVATION,
        task_types=CONFIG_TASK_TYPES,
    )

    # Initialize parameters with random seed
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, CONFIG_INPUT_DIM))
    key, init_key = jax.random.split(key)
    params = model.init(init_key, dummy_input, train=False)

    # Save parameters
    save_filepath = os.path.join(os.path.dirname(__file__), CONFIG_MODEL_PARAMS_FILE)
    save_params_with_npz(params, save_filepath)
    print(f"\nParameters saved to: {save_filepath}")


if __name__ == "__main__":
    main()
