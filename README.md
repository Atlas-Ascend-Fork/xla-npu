# xla-npu

## Dialect Converter Quick Start
### Set CMake Options
~~~
-DLLVM_BUILD_DIR="用户编译构建llvm后的构建目录"
-DLLVM_SOURCE_DIR="用户下载的llvm的源码目录"
~~~

### Add git submodule
~~~
git submodule add https://github.com/openxla/stablehlo.git xla_npu/mair/third_party/stablehlo
cd xla_npu/mair/third_party/stablehlo
git checkout main  # 或特定commit id指定的版本
~~~

### Load Cmake Project

### Build Project
#### 1. 使用IDE构建
#### 2. 命令行构建
~~~
cd xla_npu
cmake -G Ninja .. \
    -DLLVM_BUILD_DIR="用户编译构建llvm后的构建目录" \
    -DLLVM_SOURCE_DIR="用户下载的llvm的源码目录"
    
cmake --build . --target demo
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