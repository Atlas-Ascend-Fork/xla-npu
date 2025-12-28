#include <iostream>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/CAPI/Registration.h"

#include "mlir-c/IR.h"

#include "../ir/mair_ops.h"
#include "../translations/export_graphdef.h"
#include "../translations/status.h"

//MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(mair, mair, mlir::mair::MairDialect)

using namespace mlir;
using namespace mlir::air;

void buildGeluModule(mlir::ModuleOp module, mlir::MLIRContext& context) {
    // 输入输出shape
    std::vector<int> broadcastTensorShape{10, 20};

    // 创建函数： 接受tensor输入， 返回tensor
    auto f32Type = mlir::Float32Type::get(&context);
    auto inputOutputTensorType = mlir::RankedTensorType::get({10, 20}, f32Type);
    auto funcType = mlir::FunctionType::get(&context, {inputOutputTensorType}, {inputOutputTensorType});
    auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context), "main", funcType);

    // 创建函数体
    auto* block = func.addEntryBlock();
    mlir::OpBuilder builder(block, block->begin());
    auto loc = mlir::UnknownLoc::get(&context);

    // 标量类型
    auto scalarType = mlir::RankedTensorType::get({}, f32Type);

    // 后续broadcast操作的目标shape
    auto broadcastShapeTensorType = mlir::RankedTensorType::get({2}, mlir::IntegerType::get(&context, 32));
    auto broadcastShapeElements = mlir::DenseElementsAttr::get(broadcastShapeTensorType, llvm::ArrayRef<int>(broadcastTensorShape));
    auto broadcastShapeValue = ConstantOp::create(builder, loc, broadcastShapeTensorType, broadcastShapeElements).getResult();

    // 获取输入参数x: tensor<10x20xf32>
    auto x = block->getArgument(0);
    
    // x_cube = x * x * x
    auto x_squared = MulOp::create(builder, loc, inputOutputTensorType, x, x);
    auto x_cube = MulOp::create(builder, loc, inputOutputTensorType, x_squared, x);

    // a_scalar = 0.044715
    auto a_scalar = ConstantOp::create(builder, loc, scalarType, builder.getF32FloatAttr(0.044715f));
    auto a_scalar_tensor = BroadcastOp::create(builder, loc, inputOutputTensorType, a_scalar, broadcastShapeValue);

    // x3_mul_scalar = x3 * 0.044715
    auto x3_mul_scalar = MulOp::create(builder, loc, inputOutputTensorType, a_scalar_tensor.getResult(), x_cube.getResult());

    // x3_mul_scalar_add_x = x + x3 * 0.044715
    auto x3_mul_scalar_add_x = AddOp::create(builder, loc, inputOutputTensorType, x3_mul_scalar.getResult(), x);

    // scalar_sqrt_2_dev_pi = 0.797884583  sqrt(2/pi)
    auto scalar_sqrt_2_dev_pi = ConstantOp::create(builder, loc, scalarType, builder.getF32FloatAttr(0.797884583f));
    auto scalar_sqrt_2_dev_pi_tensor = BroadcastOp::create(builder, loc, inputOutputTensorType, scalar_sqrt_2_dev_pi, broadcastShapeValue);

    // x_calc_1 = sqrt(2/pi) * (x + x3 * 0.044715 )
    auto x_calc_1 = MulOp::create(builder, loc, inputOutputTensorType, scalar_sqrt_2_dev_pi_tensor.getResult(), x3_mul_scalar_add_x.getResult());

    // tanh_x_calc_1 = tanh(x_calc_1)
    auto tanh_x_calc_1 = TanhOp::create(builder, loc, inputOutputTensorType, x_calc_1.getResult());

    // scalar_1 = 1
    auto scalar_1 = ConstantOp::create(builder, loc, scalarType, builder.getF32FloatAttr(1.0f));
    auto scalar_1_tensor = BroadcastOp::create(builder, loc, inputOutputTensorType, scalar_1, broadcastShapeValue);
    
    // x_cale_2 = 1+tanh(sqrt(2/pi) * (x + x3 * 0.044715 ))
    auto x_calc_2 = AddOp::create(builder, loc, inputOutputTensorType, scalar_1_tensor.getResult(), tanh_x_calc_1.getResult());

    // scalar_1_dev_2 = 0.5
    auto scalar_1_dev_2 = ConstantOp::create(builder, loc, scalarType, builder.getF32FloatAttr(0.5f));
    auto scalar_1_dev_2_tensor = BroadcastOp::create(builder, loc, inputOutputTensorType, scalar_1_dev_2, broadcastShapeValue);

    // result = 0.5 * x * x_calc_2
    auto result = MulOp::create(builder, loc, inputOutputTensorType,  MulOp::create(builder, loc, inputOutputTensorType, scalar_1_dev_2_tensor.getResult(), x_calc_2.getResult()), x);

    // 返回结果
    mlir::func::ReturnOp::create(builder, loc, result.getResult());

    // 添加函数到模块
    module.getBody()->push_back(func);

    return;
}

int main(int argc, char** argv) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::air::AirDialect>();

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    buildGeluModule(module, context);
    std::cout << "Generated GELU MLIR:\n";
    module.print(llvm::outs());
    std::cout << std::endl;

    std::unique_ptr<ge::Graph> graphPtr = std::make_unique<ge::Graph>();
    std::vector<ge::TensorDesc> outputTensorDesc;
    std::map<std::int64_t, std::int64_t> inputOutputMap;
    Exporter exporter;
    exporter.Convert(module, graphPtr, outputTensorDesc, inputOutputMap);

    std::cout << "Convert to GE Grap:\n";
    // module.print(llvm::outs());
    std::cout << std::endl;

    return 0;
}