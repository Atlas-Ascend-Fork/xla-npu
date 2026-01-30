# xla-npu

## Dialect Converter Quick Start
### 采用cmake构建
#### Set CMake Options
~~~
-DLLVM_BUILD_DIR="用户编译构建llvm后的构建目录"
-DLLVM_SOURCE_DIR="用户下载的llvm的源码目录"
~~~

#### Add git submodule
~~~
git submodule add https://github.com/openxla/stablehlo.git xla_npu/mair/third_party/stablehlo
cd xla_npu/mair/third_party/stablehlo
git checkout main  # 或特定commit id指定的版本
~~~

#### Load Cmake Project

#### Build Project
##### 1. 使用IDE构建
##### 2. 命令行构建
~~~
cd xla_npu
cmake -G Ninja .. \
    -DLLVM_BUILD_DIR="用户编译构建llvm后的构建目录" \
    -DLLVM_SOURCE_DIR="用户下载的llvm的源码目录"
    
cmake --build . --target demo
~~~

### 采用bazel构建
#### 1. 参考“xla-npu中pjrt-npu构建流程”此issue搭建bazel构建环境，拉取xla代码到本地，将xla-npu仓中的xla_npu目录软链接到openxla代码的xla目录下
#### 2. 执行如下命令构建demo
~~~
bazel build --incompatible_disallow_empty_glob=false --remote_download_minimal --spawn_strategy=local --verbose_failures --copt=-w //xla/xla_npu/mair:demo
~~~

### Run Demo

1. 找到构建得到的demo可执行文件
2. 执行以下命令运行demo，echo中的内容为自定义mlir，-emit选项用于指定输出的mlir格式，当前仅支持mair格式，即stablehlo dialect转换为mair dialect
~~~
echo 'module {
  func.func @simple_add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %res = stablehlo.add %a, %b : tensor<2x3xf32>
    return %res : tensor<2x3xf32>
  }
}' | ./demo -emit=mair

~~~
3. 查看输出结果
~~~
module {
  func.func @simple_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "mair.Add"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
~~~

## Pjrt Npu Usage Guide
### Build And Debug
参考issue中的指导文档 https://gitcode.com/cann/xla-npu/issues/1?ref=&did=3642491#tid-3642491

## AscIR Backend Configuration

### Build Targets

主要构建目标：
- `afir_to_ascir_native` - AFIR 到 ASCIR 的转换库
- `ascir_backend_test` - AscIR 后端测试工具
- `hlo_to_afir_tool` - HLO 到 AFIR 的转换工具

构建命令：
```bash
cd /path/to/xla  # xla 工作空间根目录
bazel build //xla/xla_npu/service/convert_to_afir/afir_to_ascir_native:AFIRToASCIRTextNative
bazel build //xla/xla_npu/service/convert_to_afir:ascir_backend_test
bazel build //xla/xla_npu/service/convert_to_afir:hlo_to_afir_tool
```

### Environment Variables Configuration

运行 `ascir_backend_test` 时需要设置以下环境变量：

1. **ASCEND_MLIR_PYTHON_PATH** (必需)
   - 指向 Ascend-MLIR Python 运行时目录
   - 例如：`/path/to/xla-npu/xla_npu/third_party/Ascend-MLIR/python`

2. **PYTHON_LIB_DIR** (编译时配置)
   - Python 库路径，在 `xla_configure.bazelrc` 中配置
   - 例如：`--define PYTHON_LIB_DIR=/home/user/.pyenv/versions/3.10.12/lib`

### Running ascir_backend_test

```bash
# 设置环境变量
export ASCEND_MLIR_PYTHON_PATH=/path/to/xla-npu/xla_npu/third_party/Ascend-MLIR/python

# 运行测试
./bazel-bin/xla/xla_npu/service/convert_to_afir/ascir_backend_test \
  --hlo_file=test.hlo \
  --output_dir=./output
```

### Dependencies

- Ascend-MLIR: 通过 Git submodule 管理，位于 `xla_npu/third_party/Ascend-MLIR`
- Ascend CANN Toolkit (用于运行时编译)