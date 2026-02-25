# -*- coding:utf-8 -*-
"""
JAX implementation of MMOE (Multi-gate Mixture-of-Experts)

Author: JAX Conversion
Based on PyTorch implementation by zanshuxun

Reference:
    [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]
    https://dl.acm.org/doi/10.1145/3219819.3220007
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple, List, Optional, Callable

try:
    # Try to import MLIR library for StableHLO
    from jax._src.lib import mlir
    HAS_MLIR = True
except ImportError:
    HAS_MLIR = False

try:
    # For older JAX versions
    from jaxlib import hlo_module_string
    HAS_HLO_MODULE_STRING = True
except ImportError:
    HAS_HLO_MODULE_STRING = False


class ActivationLayer(nn.Module):
    """Activation layer supporting multiple activation functions."""
    act_name: str = 'relu'

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.act_name.lower() == 'sigmoid':
            return jax.nn.sigmoid(x)
        elif self.act_name.lower() == 'linear':
            return x
        elif self.act_name.lower() == 'relu':
            return nn.relu(x)
        elif self.act_name.lower() == 'leaky_relu':
            return jax.nn.leaky_relu(x, negative_slope=0.01)
        elif self.act_name.lower() == 'tanh':
            return jnp.tanh(x)
        elif self.act_name.lower() == 'elu':
            return jax.nn.elu(x)
        elif self.act_name.lower() == 'selu':
            return jax.nn.selu(x)
        elif self.act_name.lower() == 'gelu':
            return jax.nn.gelu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.act_name}")


class DNN(nn.Module):
    """Multi-Layer Perceptron.

    Args:
        hidden_units: List of positive integers, layer units in each layer.
        activation: Activation function to use.
        dropout_rate: Dropout rate.
        use_bn: Whether to use BatchNorm.
        use_bias: Whether to use bias in linear layers.
    """

    hidden_units: List[int]
    activation: str = 'relu'
    dropout_rate: float = 0.0
    use_bn: bool = False
    use_bias: bool = True
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # Create layers on the fly using compact
        for i in range(len(self.hidden_units) - 1):
            x = nn.Dense(
                features=self.hidden_units[i + 1],
                use_bias=self.use_bias,
                kernel_init=nn.initializers.normal(0.0001)
            )(x)

            if self.use_bn:
                x = nn.BatchNorm(momentum=0.1, epsilon=1e-5)(x, use_running_average=not train)

            if self.activation.lower() == 'sigmoid':
                x = jax.nn.sigmoid(x)
            elif self.activation.lower() == 'linear':
                pass
            elif self.activation.lower() == 'relu':
                x = nn.relu(x)
            elif self.activation.lower() == 'leaky_relu':
                x = jax.nn.leaky_relu(x, negative_slope=0.01)
            elif self.activation.lower() == 'tanh':
                x = jnp.tanh(x)
            elif self.activation.lower() == 'elu':
                x = jax.nn.elu(x)
            elif self.activation.lower() == 'selu':
                x = jax.nn.selu(x)
            elif self.activation.lower() == 'gelu':
                x = jax.nn.gelu(x)

            if self.dropout_rate > 0.0 and train:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        return x


class PredictionLayer(nn.Module):
    """Prediction layer for binary classification or regression tasks.

    Args:
        task: Task type - 'binary' or 'regression'.
        use_bias: Whether to add bias.
    """

    task: str = 'binary'
    use_bias: bool = True

    def setup(self):
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (1,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_bias:
            x = x + self.bias

        return jax.nn.sigmoid(x) if self.task == 'binary' else x


class MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts architecture in JAX.

    Args:
        num_experts: Number of expert networks.
        num_tasks: Number of tasks.
        input_dim: Input feature dimension.
        expert_dnn_hidden_units: Hidden units for expert DNNs.
        gate_dnn_hidden_units: Hidden units for gate DNNs.
        tower_dnn_hidden_units: Hidden units for tower DNNs.
        dnn_activation: Activation function for DNNs.
        dnn_dropout: Dropout rate.
        dnn_use_bn: Whether to use BatchNorm.
        task_types: List of task types ['binary', 'regression', ...].
        task_names: List of task names.
        key: JAX random key.

    Returns:
        Output for all tasks concatenated.
    """

    num_experts: int = 3
    num_tasks: int = 2
    input_dim: int = 100
    expert_dnn_hidden_units: Tuple[int, ...] = (256, 128)
    gate_dnn_hidden_units: Tuple[int, ...] = (64,)
    tower_dnn_hidden_units: Tuple[int, ...] = (64,)
    dnn_activation: str = 'relu'
    dnn_dropout: float = 0.0
    dnn_use_bn: bool = False
    task_types: Tuple[str, ...] = ('binary', 'binary')
    task_names: Tuple[str, ...] = ('ctr', 'ctcvr')

    def setup(self):
        # Validate inputs
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if self.num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(self.task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in self.task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError(f"task must be binary or regression, {task_type} is illegal")

        # Expert DNNs
        expert_input_dim = self.input_dim
        expert_hidden = [expert_input_dim] + list(self.expert_dnn_hidden_units)

        self.expert_dnns = [
            DNN(
                hidden_units=self.expert_dnn_hidden_units,
                activation=self.dnn_activation,
                dropout_rate=self.dnn_dropout,
                use_bn=self.dnn_use_bn
            )
            for _ in range(self.num_experts)
        ]

        # Gate DNNs
        gate_input_dim = self.input_dim
        if len(self.gate_dnn_hidden_units) > 0:
            self.gate_dnns = [
                DNN(
                    hidden_units=self.gate_dnn_hidden_units,
                    activation=self.dnn_activation,
                    dropout_rate=self.dnn_dropout,
                    use_bn=self.dnn_use_bn
                )
                for _ in range(self.num_tasks)
            ]
            gate_final_input_dim = self.gate_dnn_hidden_units[-1]
        else:
            gate_final_input_dim = gate_input_dim

        # Gate final layers (produce weights for experts)
        self.gate_final_layers = [
            nn.Dense(
                features=self.num_experts,
                use_bias=False,
                kernel_init=nn.initializers.normal(0.0001)
            )
            for _ in range(self.num_tasks)
        ]

        # Tower DNNs
        tower_input_dim = self.expert_dnn_hidden_units[-1]
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnns = [
                DNN(
                    hidden_units=self.tower_dnn_hidden_units,
                    activation=self.dnn_activation,
                    dropout_rate=self.dnn_dropout,
                    use_bn=self.dnn_use_bn
                )
                for _ in range(self.num_tasks)
            ]
            tower_final_input_dim = self.tower_dnn_hidden_units[-1]
        else:
            tower_final_input_dim = tower_input_dim

        # Tower final layers
        self.tower_final_layers = [
            nn.Dense(
                features=1,
                use_bias=False,
                kernel_init=nn.initializers.normal(0.0001)
            )
            for _ in range(self.num_tasks)
        ]

        # Prediction layers
        self.prediction_layers = [
            PredictionLayer(task=task_type, use_bias=True)
            for task_type in self.task_types
        ]

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor with shape (batch_size, input_dim).
            train: Whether in training mode.

        Returns:
            Output tensor with shape (batch_size, num_tasks).
        """
        batch_size = x.shape[0]

        # Expert DNNs
        expert_outputs = []
        for expert_dnn in self.expert_dnns:
            expert_out = expert_dnn(x, train=train)
            expert_outputs.append(expert_out)
        expert_outputs = jnp.stack(expert_outputs, axis=1)  # (batch_size, num_experts, expert_output_dim)

        # Gate networks and MoE output
        mmoe_outputs = []
        for task_idx in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnns[task_idx](x, train=train)
                gate_weights = self.gate_final_layers[task_idx](gate_dnn_out)
            else:
                gate_weights = self.gate_final_layers[task_idx](x)

            # Softmax on expert weights
            gate_weights = jax.nn.softmax(gate_weights, axis=-1)  # (batch_size, num_experts)
            gate_weights = gate_weights[:, :, jnp.newaxis]  # (batch_size, num_experts, 1)

            # Weighted combination of expert outputs
            mmoe_out = jnp.sum(gate_weights * expert_outputs, axis=1)  # (batch_size, expert_output_dim)
            mmoe_outputs.append(mmoe_out)

        # Tower DNNs
        task_outputs = []
        for task_idx in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_out = self.tower_dnns[task_idx](mmoe_outputs[task_idx], train=train)
                tower_logit = self.tower_final_layers[task_idx](tower_out)
            else:
                tower_logit = self.tower_final_layers[task_idx](mmoe_outputs[task_idx])

            prediction = self.prediction_layers[task_idx](tower_logit)
            task_outputs.append(prediction)

        # Concatenate all task outputs
        output = jnp.concatenate(task_outputs, axis=-1)  # (batch_size, num_tasks)

        return output


# Convenience function to initialize the model
def create_mmoe(
    input_dim: int,
    num_experts: int = 3,
    num_tasks: int = 2,
    expert_dnn_hidden_units: Tuple[int, ...] = (256, 128),
    gate_dnn_hidden_units: Tuple[int, ...] = (64,),
    tower_dnn_hidden_units: Tuple[int, ...] = (64,),
    dnn_activation: str = 'relu',
    dnn_dropout: float = 0.0,
    dnn_use_bn: bool = False,
    task_types: Tuple[str, ...] = ('binary', 'binary'),
    task_names: Tuple[str, ...] = ('ctr', 'ctcvr'),
    key: Optional[jax.Array] = None,
) -> Tuple[MMOE, dict]:
    """Create and initialize the MMOE model.

    Args:
        input_dim: Input feature dimension.
        num_experts: Number of expert networks.
        num_tasks: Number of tasks.
        expert_dnn_hidden_units: Hidden units for expert DNNs.
        gate_dnn_hidden_units: Hidden units for gate DNNs.
        tower_dnn_hidden_units: Hidden units for tower DNNs.
        dnn_activation: Activation function.
        dnn_dropout: Dropout rate.
        dnn_use_bn: Whether to use BatchNorm.
        task_types: Task types tuple.
        task_names: Task names tuple.
        key: JAX random key.

    Returns:
        Tuple of (initialized model, parameters dictionary).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    model = MMOE(
        num_experts=num_experts,
        num_tasks=num_tasks,
        input_dim=input_dim,
        expert_dnn_hidden_units=expert_dnn_hidden_units,
        gate_dnn_hidden_units=gate_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        dnn_activation=dnn_activation,
        dnn_dropout=dnn_dropout,
        dnn_use_bn=dnn_use_bn,
        task_types=task_types,
        task_names=task_names,
    )

    # Initialize with dummy input
    dummy_input = jnp.zeros((1, input_dim))
    key, init_key = jax.random.split(key)
    params = model.init(init_key, dummy_input, train=False)

    return model, params


if __name__ == "__main__":
    # Example usage
    print("MMOE JAX Implementation Test")
    print("=" * 50)

    # Configuration
    input_dim = 100
    batch_size = 32
    num_experts = 3
    num_tasks = 2

    # Create model
    key = jax.random.PRNGKey(42)
    model, params = create_mmoe(
        input_dim=input_dim,
        num_experts=num_experts,
        num_tasks=num_tasks,
        expert_dnn_hidden_units=(256, 128),
        gate_dnn_hidden_units=(64,),
        tower_dnn_hidden_units=(64,),
        dnn_activation='relu',
        task_types=('binary', 'binary'),
        key=key,
    )

    print(f"\nModel created successfully!")
    print(f"Input dimension: {input_dim}")
    print(f"Number of experts: {num_experts}")
    print(f"Number of tasks: {num_tasks}")
    print(f"Task types: {model.task_types}")
    print(f"Task names: {model.task_names}")

    # Generate random input
    x = jax.random.normal(key, (batch_size, input_dim))

    # Forward pass (inference mode)
    output = model.apply(params, x, train=False)
    print(f"\nForward pass result:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_tasks})")

    # Forward pass (training mode)
    key, dropout_key = jax.random.split(key)
    output_train = model.apply(params, x, train=True, rngs={'dropout': dropout_key})
    print(f"\nTraining mode output shape: {output_train.shape}")

    # Sample output values
    print(f"\nSample output values (first 2 samples):")
    print(output[:2])

    # ==================== HLO/StableHLO Debug ====================
    print("\n" + "=" * 50)
    print("HLO/StableHLO Debug Information")
    print("=" * 50)

    # Define jitted function for HLO analysis
    @jax.jit
    def jitted_forward(params, x):
        return model.apply(params, x, train=False)

    # Get HLO text
    print("\n--- StableHLO Text ---")
    lowered = jitted_forward.lower(params, x)
    stablehlo_text = lowered.as_text()
    print(stablehlo_text)

    print("\n--- HLO Text ---")
    hlo_text = lowered.compile().as_text()
    print(hlo_text)

    # Print compilation statistics
    print("\n--- Compilation Statistics ---")
    print(f"Number of ops: {len(hlo_text.split('%'))}")
    print(f"Number of functions: {hlo_text.count('func.func ')}")
    print(f"Number of dot operations: {hlo_text.count('stablehlo.dot_general')}")
    print(f"Number of softmax operations: {hlo_text.count('stablehlo.reduce')}")
    print(f"Number of broadcast operations: {hlo_text.count('stablehlo.broadcast_in_dim')}")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
