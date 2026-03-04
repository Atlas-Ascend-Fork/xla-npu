import jax
import jax.numpy as jnp
import os
import numpy as np


class MMOEFromStableHLO:
    def __init__(self):
        self.input_dim = 100
        self.hidden_dim = 128
        self.num_experts = 3
        self.num_tasks = 2
        self.batch_size = 32

    def init_params(self):
        params = {
            'expert_dnns_0': {
                'kernel': np.zeros((self.input_dim, self.hidden_dim)),
                'bias': np.zeros((self.hidden_dim,))
            },
            'expert_dnns_1': {
                'kernel': np.zeros((self.input_dim, self.hidden_dim)),
                'bias': np.zeros((self.hidden_dim,))
            },
            'expert_dnns_2': {
                'kernel': np.zeros((self.input_dim, self.hidden_dim)),
                'bias': np.zeros((self.hidden_dim,))
            },
            'gate_final_layers_0': {
                'kernel': np.zeros((self.input_dim, self.num_experts))
            },
            'gate_final_layers_1': {
                'kernel': np.zeros((self.input_dim, self.num_experts))
            },
            'tower_final_layers_0': {
                'kernel': np.zeros((self.hidden_dim, 1))
            },
            'tower_final_layers_1': {
                'kernel': np.zeros((self.hidden_dim, 1))
            },
            'prediction_layers_0': {
                'bias': np.zeros((1,))
            },
            'prediction_layers_1': {
                'bias': np.zeros((1,))
            }
        }
        return params

    def relu(self, x):
        return jnp.maximum(x, 0)

    def __call__(self, params, x):
        expert_dnns_0_kernel = params['expert_dnns_0']['kernel']
        expert_dnns_0_bias = params['expert_dnns_0']['bias']
        expert_dnns_1_kernel = params['expert_dnns_1']['kernel']
        expert_dnns_1_bias = params['expert_dnns_1']['bias']
        expert_dnns_2_kernel = params['expert_dnns_2']['kernel']
        expert_dnns_2_bias = params['expert_dnns_2']['bias']
        gate_final_layers_0_kernel = params['gate_final_layers_0']['kernel']
        gate_final_layers_1_kernel = params['gate_final_layers_1']['kernel']
        tower_final_layers_0_kernel = params['tower_final_layers_0']['kernel']
        tower_final_layers_1_kernel = params['tower_final_layers_1']['kernel']
        prediction_layers_0_bias = params['prediction_layers_0']['bias']
        prediction_layers_1_bias = params['prediction_layers_1']['bias']

        x0 = jnp.dot(x, expert_dnns_0_kernel)
        x0 = x0 + expert_dnns_0_bias
        x0 = self.relu(x0)

        x1 = jnp.dot(x, expert_dnns_1_kernel)
        x1 = x1 + expert_dnns_1_bias
        x1 = self.relu(x1)

        x2 = jnp.dot(x, expert_dnns_2_kernel)
        x2 = x2 + expert_dnns_2_bias
        x2 = self.relu(x2)

        expert_outputs = jnp.stack([x0, x1, x2], axis=1)

        gate0 = jnp.dot(x, gate_final_layers_0_kernel)
        gate0_max = jnp.max(gate0, axis=-1, keepdims=True)
        gate0_exp = jnp.exp(gate0 - gate0_max)
        gate0_sum = jnp.sum(gate0_exp, axis=-1, keepdims=True)
        gate0_weights = gate0_exp / gate0_sum

        gate1 = jnp.dot(x, gate_final_layers_1_kernel)
        gate1_max = jnp.max(gate1, axis=-1, keepdims=True)
        gate1_exp = jnp.exp(gate1 - gate1_max)
        gate1_sum = jnp.sum(gate1_exp, axis=-1, keepdims=True)
        gate1_weights = gate1_exp / gate1_sum

        mmoe_out0 = jnp.sum(gate0_weights[:, :, jnp.newaxis] * expert_outputs, axis=1)
        mmoe_out1 = jnp.sum(gate1_weights[:, :, jnp.newaxis] * expert_outputs, axis=1)

        tower0 = jnp.dot(mmoe_out0, tower_final_layers_0_kernel)
        tower0 = tower0 + prediction_layers_0_bias
        pred0 = jax.nn.sigmoid(tower0)

        tower1 = jnp.dot(mmoe_out1, tower_final_layers_1_kernel)
        tower1 = tower1 + prediction_layers_1_bias
        pred1 = jax.nn.sigmoid(tower1)

        output = jnp.concatenate([pred0, pred1], axis=-1)
        return output


if __name__ == "__main__":
    # Npu platform init
    PLATFORM_NAME = 'npu' # Use a simple name without underscores for safety
    LIBRARY_PATH = os.path.abspath("pjrt_c_api_npu_plugin.so")
    os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"
    os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

    print("MMOE Model from StableHLO")
    print("=" * 50)

    model = MMOEFromStableHLO()

    params = model.init_params()
    x = np.zeros((model.batch_size, model.input_dim))

    print(f"\nInput shape: {x.shape}")
    print(f"Expected input shape: ({model.batch_size}, {model.input_dim})")

    @jax.jit
    def forward(params, x):
        return model(params, x)

    output = forward(params, x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected output shape: ({model.batch_size}, {model.num_tasks})")

    # print("\n--- StableHLO Text ---")
    # lowered = forward.lower(params, x)
    # stablehlo_text = lowered.as_text()
    # print(stablehlo_text)

    # print("\n--- Verification ---")
    # print("Comparing with original StableHLO structure:")
    # print("- 3 Expert DNNs: ✓")
    # print("- 2 Gate networks: ✓")
    # print("- 2 Tower DNNs: ✓")
    # print("- 2 Prediction layers with sigmoid: ✓")
    # print("- Output shape (32, 2): ✓")
