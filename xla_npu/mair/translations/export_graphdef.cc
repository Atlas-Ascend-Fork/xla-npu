#include "./export_graphdef.h"
#include "./status.h"
#include "./convert_type.h"
#include "./convert_tensor.h"

#include <iostream>
#include <fstream>

#include "../ir/mair_ops.h"
// #include "utils/utils.h"
// #include "translations/export_utils.h"
// #include "utils/convert_type.h"
// #include "utils/convert_tensor.h"
// #include "graph/operator_factory.h"
// #include "graph/operator.h"

#include <utility>
#include <map>
namespace mlir::air {
using llvm::dyn_cast;
using llvm::isa;
using mlir::BlockArgument;
using mlir::Dialect;
using mlir::Operation;
using mlir::Value;

const std::unordered_map<std::string, std::vector<std::string>> Exporter::inputNameMap = {
    {"Add", {"x1", "x2"}},
    {"Mul", {"x1", "x2"}},
    {"Tanh", {"x"}},
    {"BroadcastTo", {"x", "shape"}},
};


Status Exporter::Convert(mlir::ModuleOp module, std::unique_ptr<ge::Graph> &graphPtr,
                        std::vector<ge::TensorDesc> &outputTensorDesc,
                        std::map<std::int64_t, std::int64_t> &inputOutputMap) {

    std::optional<mlir::func::FuncOp> entry_func;
    // ===================================================
    // rebuild air graph
    // ===================================================
    mlir::StringAttr entry_func_id = mlir::StringAttr::get(module.getContext(), "main");

    for (auto function : module.getOps<mlir::func::FuncOp>()) {
        if (function.isExternal()) {
            log("External functions not supported");
            return Status::OK();
        }
        if (function.getName() == entry_func_id) {
            entry_func.emplace(function);
        }
    }
  
    if (!entry_func.has_value()) {
        log("entry function `main` must be present");
        return Status::OK();
    }


    // =========================================================
    // start to build ge graph
    // =========================================================
    mlir::func::FuncOp function = entry_func.value();
    mlir::Block& entryBlock = function.getBody().front();

    // ==================================
    // build input arguments
    // ==================================
        
    for (auto arg : entryBlock.getArguments()) {
        std::string mlirSymbol = "%arg" + std::to_string(arg.getArgNumber());
        std::string realSymbol = "input_" + std::to_string(inputInfos.size());
        log("Add arguments:" + mlirSymbol + " ; parse to :" + realSymbol);
        
        // 获取参数类型和形状
        mlir::TensorType arg_type = mlir::dyn_cast<mlir::TensorType>(arg.getType());

        ge::DataType dtype;
        ConvertToDataType(arg_type.getElementType(), &dtype);

        auto shape = ge::Shape();
        ConvertToShape(arg_type, &shape);

        auto format = ge::FORMAT_ND;
        ge::TensorDesc data_desc(shape, format, dtype);

        log("Gen ArgumentNode Data:[" + realSymbol + "] InputDesc/OutputDesc[" + mlir::debugString(arg_type) +"]");
        auto data = ge::OperatorFactory::CreateOperator(realSymbol, "Data");
        data.UpdateInputDesc("x", data_desc);
        data.UpdateOutputDesc(realSymbol.c_str(), data_desc);
        // ge::Operator data;
        
        
        // // 记录入参信息
        inputInfos.push_back({mlirSymbol, realSymbol, data, data_desc});
        
        // 建立映射
        symbolMap[mlirSymbol] = {
            .orgSymbol = mlirSymbol,
            .realSymbol = realSymbol,
            .geOp = data,
            .tensorDesc = data_desc
        };
        
        llvm::errs() << "入参: MLIR符号=" << mlirSymbol 
                    << ", 真实符号=" << realSymbol << "\n";
    }

    std::vector<ge::Operator> inputs;
    for (const auto& inputInfo : inputInfos) {
        inputs.emplace_back(inputInfo.geOp);
    }


    graphPtr->SetInputs(inputs);



    // ==================================
    // build ge oprators
    // ==================================
    // 算子计数器（用于生成唯一符号）
    int opCounter = 0;

    for (mlir::Operation& op : entryBlock.getOperations()) {
        log("--Start to deal with the op:" + op.getName().getStringRef().str());
        // ReturnOp
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            SymbolInfo symbolInfo = processReturnOp(op);
            std::vector<ge::Operator> outputs{symbolInfo.geOp};
            graphPtr->SetOutputs(outputs);
            log("设置图输出为: " +symbolInfo.realSymbol);
            continue;
        }
        
        // 处理普通操作
        processOperation(op, opCounter);
        log("--This op end.");
        opCounter++;
    }


    return Status::OK();
}


Exporter::SymbolInfo Exporter::processReturnOp(mlir::Operation& returnOp) {
    mlir::func::ReturnOp returnOpCast = mlir::cast<mlir::func::ReturnOp>(returnOp);
    
    mlir::Value returnValue = returnOpCast.getOperand(0);
    log("returnOp operand:" + mlir::debugString(returnValue));
    std::string mlirSymbol = getMlirSymbol(returnValue);
    
    // 查找真实输出符号
    auto it = symbolMap.find(mlirSymbol);
    
    const SymbolInfo& symbolInfo = it->second;
    llvm::errs() << "图输出: MLIR符号=" << mlirSymbol << "\n";
    
    return symbolInfo;
}

bool Exporter::processOperation(mlir::Operation& mlirOp, int opCounter) {
    // 1. 获取操作名（去掉"mair."前缀）
    std::string opName = mlirOp.getName().getStringRef().str();
    if (opName.find("mair.") != 0) {
        llvm::errs() << "警告: 跳过非mair操作: " << opName << "\n";
        return true;
    }
    
    std::string opType = opName.substr(5);  // 去掉"mair."
    
    llvm::errs() << "处理操作: " << opName << " -> " << opType 
                << " (序号=" << opCounter << ")\n";
    
    // 2. 创建目标操作
    ge::Operator geOp = ge::OperatorFactory::CreateOperator(opType, opType);
    // ge::Operator geOp;
    
    // 3. 处理输入
    if (!processOperationInputs(mlirOp, geOp, opType)) {
        return false;
    }
    
    // 4. 处理输出
    if (!processOperationOutputs(mlirOp, geOp, opCounter)) {
        return false;
    }
    
    // // 5. 添加到图----可能不需要
    // if (!graph.add(geOp)) {
    //     llvm::errs() << "错误：添加操作到图失败\n";
    //     return false;
    // }
    return true;
}
bool Exporter::processOperationInputs(mlir::Operation& mlirOp, ge::Operator& geOp, std::string opType) {
    unsigned numOperands = mlirOp.getNumOperands();
    
    for (unsigned i = 0; i < numOperands; i++) {
        mlir::Value operand = mlirOp.getOperand(i);
        log("trans the input: " + mlir::debugString(operand));
        
        // 获取MLIR符号
        std::string mlirSymbol = getMlirSymbol(operand);
        if (mlirSymbol.empty()) {
            llvm::errs() << "错误：无法获取操作数符号\n";
            return false;
        }
        
        // 查找真实符号
        auto it = symbolMap.find(mlirSymbol);
        if (it == symbolMap.end()) {
            llvm::errs() << "错误：未找到符号映射: " << mlirSymbol << "\n";
            return false;
        }
        
        std::string realSymbol = it->second.realSymbol;
        // 获取参数类型和形状
        mlir::TensorType arg_type = mlir::dyn_cast<mlir::TensorType>(operand.getType());
        ge::DataType dtype;
        ConvertToDataType(arg_type.getElementType(), &dtype);
        auto shape = ge::Shape();
        ConvertToShape(arg_type, &shape);
        auto format = ge::FORMAT_ND;
        ge::TensorDesc data_desc(shape, format, dtype);

        // 获取输入符号
        std::string_view inputName= getGeOpInputName(opType, i);

        // 设置输入描述
        geOp.SetInput(realSymbol.c_str(), it->second.geOp);
        geOp.UpdateInputDesc(realSymbol.c_str(), data_desc);
        
        
        llvm::errs() << "  输入[" << i << "]: MLIR符号=" << mlirSymbol 
                    << ", 真实符号=" << realSymbol << ", inputName=" << inputName << "\n";
    }
    
    return true;
}

// 处理操作输出
bool Exporter::processOperationOutputs(mlir::Operation& mlirOp, ge::Operator& geOp, int opCounter) {
    unsigned numResults = mlirOp.getNumResults();
    
    for (unsigned i = 0; i < numResults; i++) {
        mlir::Value result = mlirOp.getResult(i);
        
        // 获取MLIR符号
        std::string mlirSymbol = getMlirSymbol(result);
        if (mlirSymbol.empty()) {
            llvm::errs() << "错误：无法获取结果符号\n";
            return false;
        }
        
        // 生成唯一输出符号
        std::string realSymbol = generateOutputSymbol(opCounter, i);
        
        // 获取参数类型和形状
        mlir::TensorType arg_type = mlir::dyn_cast<mlir::TensorType>(result.getType());
        ge::DataType dtype;
        ConvertToDataType(arg_type.getElementType(), &dtype);
        auto shape = ge::Shape();
        ConvertToShape(arg_type, &shape);
        auto format = ge::FORMAT_ND;
        ge::TensorDesc data_desc(shape, format, dtype);
        // 设置输出描述
        geOp.UpdateOutputDesc(realSymbol.c_str(), data_desc);
        
        
        // 建立映射
        symbolMap[mlirSymbol] = {
            .orgSymbol = mlirSymbol,
            .realSymbol = realSymbol,
            .geOp = geOp,
            .tensorDesc = data_desc
        };
        
        llvm::errs() << "  输出[" << i << "]: MLIR符号=" << mlirSymbol 
                    << ", 真实符号=" << realSymbol << "\n";
    }
    
    return true;
}

std::string Exporter::getMlirSymbol(mlir::Value value) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) { // input
        return "%arg" + std::to_string(blockArg.getArgNumber());
    } else if (auto opResult = mlir::dyn_cast<mlir::OpResult>(value)) { // output
        mlir::Operation* definingOp = opResult.getDefiningOp();
        if (!definingOp) {
            return "";
        }
        
        // 查找结果在操作中的索引
        unsigned resultIndex = opResult.getResultNumber();
        // 尝试获取标准格式的符号
        if (definingOp->getNumResults() == 1) {
            return "%" + std::to_string(getOperationNumber(definingOp));
        } else {
            return "%" + std::to_string(getOperationNumber(definingOp)) 
                   + "." + std::to_string(resultIndex);
        }
    }
    return "";
}

// 获取操作序号（用于生成符号）
int Exporter::getOperationNumber(mlir::Operation* op) {
    // 简单实现：遍历块计数
    mlir::Block* block = op->getBlock();
    int counter = 0;
    
    for (mlir::Operation& blockOp : *block) {
        if (&blockOp == op) {
            return counter;
        }
        
        // 跳过返回操作
        if (!mlir::isa<mlir::func::ReturnOp>(blockOp)) {
            counter++;
        }
    }
    
    return -1;
}

// 生成输出符号
std::string Exporter::generateOutputSymbol(int opIndex, int outputIndex) {
    return "op_" + std::to_string(opIndex) + "_out_" + std::to_string(outputIndex);
}

std::string_view Exporter::getGeOpInputName(const std::string& opName, int index) {
    auto it = inputNameMap.find(opName);
    if (it == inputNameMap.end()) {
        return nullptr;  
    }

    const std::vector<std::string>& inputNames = it->second;
    if (inputNames.size() >= index+1) {
        return inputNames[index]; 
    }

    return nullptr;
}

} // namespace mlir::air