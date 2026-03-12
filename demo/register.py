import os
import traceback

import jax
import numpy as np

# Note: No need for ctypes.CDLL() anymore, JAX loads the library automatically with the correct ENV var
# 1. Define the platform name and library path
PLATFORM_NAME = 'npu'  # Use a simple name without underscores for safety
LIBRARY_PATH = os.path.abspath("pjrt_c_api_npu_plugin.so")

# 2. Set the mandatory environment variable
#    The format is 'platform_name:path/to/library.so'
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"

# 3. (Optional but good practice) Ensure JAX prefers your platform
#    This is the variable you used before, useful to explicitly request it
os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

# --- Run a simple JAX program ---
try:
    print(f"\nJAX Default Backend: {jax.default_backend()}")
    print(f"Devices type: {type(jax.devices())}")
    print(f"Devices available: {jax.devices()}")

    x1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    print(f"input array: {x1}")
    x2 = x1 + 1
    print(f"input array: {x2}")


    # 定义多融合算子场景的函数
    def multi_fusion_func(x1, x2):
        y = jax.nn.gelu(x1) + jax.nn.relu(x2)
        # y =  jax.nn.gelu(x1)
        return y


    func = jax.jit(multi_fusion_func)
    y = func(x1, x2)
    result = y.block_until_ready()
    print(f"output array: {result}")

except RuntimeError as e:
    traceback.print_exc()
    print(f"\nJAX Integration Error: {e}")
    print("Check compilation steps and ensure all mandatory C++ functions are implemented.")
