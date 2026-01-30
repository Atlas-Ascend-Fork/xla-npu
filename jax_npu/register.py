import os
# os.environ['TC_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '99'  # 全局VLOG级别设为99
# os.environ['TF_CPP_VMODULE'] = '*=99'      # 所有文件VLOG级别设为99
os.environ['XLA_FLAGS'] = '--xla_dump_to=./xla_dump'      # 设置xla dump目录

import numpy as np
import jax
import jax.numpy as jnp
import warnings
import traceback
import sys
import logging

# logging.basicConfig(level=logging.DEBUG)  # 把jax日志输出到 stderr

# Note: No need for ctypes.CDLL() anymore, JAX loads the library automatically with the correct ENV var
# 1. Define the platform name and library path
PLATFORM_NAME = 'npu' # Use a simple name without underscores for safety
LIBRARY_PATH = os.path.abspath("pjrt_c_api_npu_plugin.so")

# 2. Set the mandatory environment variable
#    The format is 'platform_name:path/to/library.so'
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"

# 3. (Optional but good practice) Ensure JAX prefers your platform
#    This is the variable you used before, useful to explicitly request it
os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

# 4. Import jax.lib.xla_client to trigger the backend discovery mechanism
import jax.lib.xla_client
print(f"JAX Rerunning client discovery for platform: {PLATFORM_NAME}")

# Debug: Test npu device memory malloc
# np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32, copy=True, order='K')
# print("src date: ", np_arr)
# npu_device = jax.devices("npu")[0]
# npu_arr = jax.device_put(np_arr, device=npu_device) # call PJRT_Client_BufferFromHostBuffer from xla
# cpu_arr = jax.device_get(npu_arr) # call PJRT_Buffer_ToHostBuffer from xla
# print("dst data: ", cpu_arr)

# --- Run a simple JAX program ---
try:
    print(f"\nJAX Default Backend: {jax.default_backend()}")
    print(f"Devices type: {type(jax.devices())}")
    print(f"Devices available: {jax.devices()}")

    # def f():
    #     x = jnp.array([1, 2, 3])
    #     return x + x
    # f_jitted = jax.jit(f)
    # f_jitted()

    # Creates a JAX array, calling Mock_Buffer_FromHost()
    # x = jnp.array([8.0, 2.0, 3.0])
    # print(f"Created JAX array on device: {x.devices()}")
    # print(x)

    # x = np.ones((1024, 1024), dtype=np.float32)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    # x = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"input array: {x}")

    func = jax.jit(lambda x: jax.nn.gelu(x))
    # func = jax.jit(lambda x: x + 2)
    y = func(x)
    result = y.block_until_ready()
    print(f"output array: {result}")

    # Triggers execution
    # print(f"Computed result: {y.block_until_ready()}")

except RuntimeError as e:
    traceback.print_exc()
    print(f"\nJAX Integration Error: {e}")
    print("Check compilation steps and ensure all mandatory C++ functions are implemented.")