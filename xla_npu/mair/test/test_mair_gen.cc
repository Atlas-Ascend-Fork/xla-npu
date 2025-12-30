#include "./test_mair_gen.h"
#include "./test_mair_gen.h"

//MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(mair, mair, mlir::mair::MairDialect)

using namespace mlir;
using namespace mlir::air;

void GraphGenTester::buildGeluModule(mlir::ModuleOp& module, mlir::MLIRContext& context) {
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
    auto broadcastShapeValue = func.getArgument(0);

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

void GraphGenTester::buildAddMulModule(mlir::ModuleOp& module, mlir::MLIRContext& context) {
    std::cout << "Start build addNul mlir" << std::endl;
    // 输入输出shape
    std::vector<int> broadcastTensorShape{10, 20};

    // 创建函数： 接受tensor输入， 返回tensor
    auto f32Type = mlir::Float32Type::get(&context);
    auto inputOutputTensorType = mlir::RankedTensorType::get({10, 20}, f32Type);
    auto funcType = mlir::FunctionType::get(&context, {inputOutputTensorType, inputOutputTensorType}, {inputOutputTensorType});
    auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context), "main", funcType);

    // 创建函数体
    auto* block = func.addEntryBlock();
    mlir::OpBuilder builder(block, block->begin());
    auto loc = mlir::UnknownLoc::get(&context);

    // 标量类型
    auto scalarType = mlir::RankedTensorType::get({}, f32Type);

    // 后续broadcast操作的目标shape
    auto broadcastShapeValue = func.getArgument(0);

    // 获取输入参数x: tensor<10x20xf32>
    auto x1 = block->getArgument(0);
    auto x2 = block->getArgument(1);

    // add
    auto add = AddOp::create(builder, loc, inputOutputTensorType, x1, x2);

    // a_scalar = 0.044715
    auto a_scalar = ConstantOp::create(builder, loc, scalarType, builder.getF32FloatAttr(0.044715f));
    auto a_scalar_tensor = BroadcastOp::create(builder, loc, inputOutputTensorType, a_scalar, broadcastShapeValue);

    // x3_mul_scalar = x3 * 0.044715
    auto result = MulOp::create(builder, loc, inputOutputTensorType, a_scalar_tensor.getResult(), add.getResult());

    // 返回结果
    mlir::func::ReturnOp::create(builder, loc, result.getResult());

    // 添加函数到模块
    module.getBody()->push_back(func);

    return;
}

void GraphGenTester::buildAddModule(mlir::ModuleOp& module, mlir::MLIRContext& context) {
    // 输入输出shape
    std::cout << "Start build add mlir" << std::endl;
    // 创建函数： 接受tensor输入， 返回tensor
    auto f32Type = mlir::Float32Type::get(&context);
    auto inputOutputTensorType = mlir::RankedTensorType::get({10, 20}, f32Type);
    auto funcType = mlir::FunctionType::get(&context, {inputOutputTensorType, inputOutputTensorType}, {inputOutputTensorType});
    auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context), "main", funcType);

    // 创建函数体
    auto* block = func.addEntryBlock();
    mlir::OpBuilder builder(block, block->begin());
    auto loc = mlir::UnknownLoc::get(&context);

    // 获取输入参数x: tensor<10x20xf32>
    auto x1 = block->getArgument(0);
    auto x2 = block->getArgument(1);

    // add
    auto result = AddOp::create(builder, loc, inputOutputTensorType, x1, x2);

    // 返回结果
    mlir::func::ReturnOp::create(builder, loc, result.getResult());

    // 添加函数到模块
    module.getBody()->push_back(func);

    return;
}

void GraphGenTester::buildAEmptyModule(mlir::ModuleOp& module, mlir::MLIRContext& context) {
    // 输入输出shape
    std::cout << "Start build add mlir" << std::endl;
    // 创建函数： 接受tensor输入， 返回tensor
    auto f32Type = mlir::Float32Type::get(&context);
    auto inputOutputTensorType = mlir::RankedTensorType::get({10, 20}, f32Type);
    auto funcType = mlir::FunctionType::get(&context, {inputOutputTensorType}, {inputOutputTensorType});
    auto func = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context), "main", funcType);

    // 创建函数体
    auto* block = func.addEntryBlock();
    mlir::OpBuilder builder(block, block->begin());
    auto loc = mlir::UnknownLoc::get(&context);

    // 获取输入参数x: tensor<10x20xf32>
    auto x1 = block->getArgument(0);

    // 返回结果
    mlir::func::ReturnOp::create(builder, loc, x1);

    // 添加函数到模块
    module.getBody()->push_back(func);

    return;
}

Status GraphGenTester::Convert(mlir::ModuleOp& module, ge::GraphPtr &graphPtr) {
    Exporter exporter;
    return exporter.Convert(module, graphPtr);
}

mlir::ModuleOp GraphGenTester::BuildMAIRGeluModul() {
    static mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::air::AirDialect>();

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    // buildAddModule(module, context);
    // buildAddMulModule(module, context);
    // buildAEmptyModule(module, context);
    buildGeluModule(module, context);
    
    std::cout << "Generated MLIR:\n";
    module.print(llvm::outs());
    std::cout << std::endl;
    return module;
}

int main(int argc, char** argv) {
    auto module = GraphGenTester::BuildMAIRGeluModul();

    ge::GraphPtr graphPtr = std::make_shared<ge::Graph>("XLAGraph");;
    if (!module) {
        log("null module.");
    }
    
    GraphGenTester::Convert(module, graphPtr);
    std::cout << "Finished convert to GE Grap:\n";
    // module.print(llvm::outs());
    std::cout << std::endl;

    return 0;
}