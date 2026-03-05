#include "./export_graphdef.h"
#include "./status.h"
#include "./convert_type.h"
#include "./convert_tensor.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <iomanip>

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
    {"Div", {"x1", "x2"}},
    {"Exp", {"x"}},
    {"Maximum", {"x1", "x2"}},
    {"Neg", {"x"}},
    {"Sub", {"x1", "x2"}},
    {"MatMul", {"x1", "x2"}},
    {"Reshape", {"x", "shape"}},
    {"ReduceSum", {"x", "axes"}},
    {"ReduceMax", {"x", "axes"}},
    {"ReduceMin", {"x", "axes"}},
    {"ConcatV2", {"x", "concat_dim"}},
    {"Cast", {"x"}},
};

const std::map<ge::DataType, std::string> DataTypeNames = {
  {ge::DataType::DT_FLOAT          , "DT_FLOAT         "},       // float type
  {ge::DataType::DT_FLOAT16        , "DT_FLOAT16       "},       // fp16 type
  {ge::DataType::DT_INT8           , "DT_INT8          "},       // int8 type
  {ge::DataType::DT_INT32          , "DT_INT32         "},       // int32 type
  {ge::DataType::DT_UINT8          , "DT_UINT8         "},       // uint8 type
                                                                  // reserved
  {ge::DataType::DT_INT16          , "DT_INT16         "},       // int16 type
  {ge::DataType::DT_UINT16         , "DT_UINT16        "},       // uint16 type
  {ge::DataType::DT_UINT32         , "DT_UINT32        "},        // unsigned int32
  {ge::DataType::DT_INT64          , "DT_INT64         "},         // int64 type
  {ge::DataType::DT_UINT64         , "DT_UINT64        "},        // unsigned int64
  {ge::DataType::DT_DOUBLE         , "DT_DOUBLE        "},        // double type
  {ge::DataType::DT_BOOL           , "DT_BOOL          "},        // bool type
  {ge::DataType::DT_STRING         , "DT_STRING        "},        // string type
  {ge::DataType::DT_DUAL_SUB_INT8  , "DT_DUAL_SUB_INT8 "},        // dual output int8 type
  {ge::DataType::DT_DUAL_SUB_UINT8 , "DT_DUAL_SUB_UINT8"},        // dual output uint8 type
  {ge::DataType::DT_COMPLEX64      , "DT_COMPLEX64     "},        // complex64 type
  {ge::DataType::DT_COMPLEX128     , "DT_COMPLEX128    "},        // complex128 type
  {ge::DataType::DT_QINT8          , "DT_QINT8         "},        // qint8 type
  {ge::DataType::DT_QINT16         , "DT_QINT16        "},        // qint16 type
  {ge::DataType::DT_QINT32         , "DT_QINT32        "},        // qint32 type
  {ge::DataType::DT_QUINT8         , "DT_QUINT8        "},        // quint8 type
  {ge::DataType::DT_QUINT16        , "DT_QUINT16       "},        // quint16 type
  {ge::DataType::DT_RESOURCE       , "DT_RESOURCE      "},        // resource type
  {ge::DataType::DT_STRING_REF     , "DT_STRING_REF    "},        // string ref type
  {ge::DataType::DT_DUAL           , "DT_DUAL          "},        // dual output type
  {ge::DataType::DT_VARIANT        , "DT_VARIANT       "},        // dt_variant type
  {ge::DataType::DT_BF16           , "DT_BF16          "},        // bf16 type
  {ge::DataType::DT_UNDEFINED      , "DT_UNDEFINED     "},        // Used to indicate a DataType field has not been set.
  {ge::DataType::DT_INT4           , "DT_INT4          "},        // int4 type
  {ge::DataType::DT_UINT1          , "DT_UINT1         "},        // uint1 type
  {ge::DataType::DT_INT2           , "DT_INT2          "},        // int2 type
  {ge::DataType::DT_UINT2          , "DT_UINT2         "},        // uint2 type
  {ge::DataType::DT_COMPLEX32      , "DT_COMPLEX32     "},        // complex32 type
  {ge::DataType::DT_HIFLOAT8       , "DT_HIFLOAT8      "},        // hifloat8 type
  {ge::DataType::DT_FLOAT8_E5M2    , "DT_FLOAT8_E5M2   "},        // float8_e5m2 type
  {ge::DataType::DT_FLOAT8_E4M3FN  , "DT_FLOAT8_E4M3FN "},        // float8_e4m3fn type
  {ge::DataType::DT_FLOAT8_E8M0    , "DT_FLOAT8_E8M0   "},        // float8_e8m0 type
  {ge::DataType::DT_FLOAT6_E3M2    , "DT_FLOAT6_E3M2   "},        // float6_e3m2 type
  {ge::DataType::DT_FLOAT6_E2M3    , "DT_FLOAT6_E2M3   "},        // float6_e2m3 type
  {ge::DataType::DT_FLOAT4_E2M1    , "DT_FLOAT4_E2M1   "},        // float4_e2m1 type
  {ge::DataType::DT_FLOAT4_E1M2    , "DT_FLOAT4_E1M2   "},        // float4_e1m2 type
  {ge::DataType::DT_MAX            , "DT_MAX           "}        // Mark the boundaries of data types
};




Status Exporter::Convert(mlir::ModuleOp& module, ge::GraphPtr &graphPtr) {
    log("Start to covert module to ge graph.");
    std::optional<mlir::func::FuncOp> entry_func;
    mlir::StringAttr entry_func_id = mlir::StringAttr::get(module.getContext(), "main");
    log("Get main func id.");
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
    log("===================Get mlir function done. Start to get arguments of the function.");
    // ==================================
    // build input arguments
    // ==================================
        
    // for (auto arg : entryBlock.getArguments()) {
    for (int i = 0; i < entryBlock.getNumArguments(); i++) {
        mlir::BlockArgument arg = entryBlock.getArgument(i);
        std::string mlirSymbol = "%arg" + std::to_string(arg.getArgNumber());
        std::string realSymbol = "Data_" + std::to_string(i);
        log("Add arguments:" + mlirSymbol + " ; parse to :" + realSymbol);
        
        // 获取参数类型和形状
        mlir::TensorType arg_type = mlir::dyn_cast<mlir::TensorType>(arg.getType());

        ge::DataType dtype;
        ConvertToDataType(arg_type.getElementType(), &dtype);

        auto shape = ge::Shape();
        ConvertToShape(arg_type, &shape);

        auto format = ge::FORMAT_ND;
        ge::TensorDesc data_desc(shape, format, dtype);

        // ge::TensorDesc desc(ge::Shape({10, 10}), ge::FORMAT_ND, ge::DT_FLOAT);

        log("Gen ArgumentNode Data:[" + realSymbol + "] InputDesc/OutputDesc[" + mlir::debugString(arg_type) +"]");
        auto data = ge::OperatorFactory::CreateOperator(realSymbol.c_str(), "Data");
        data.SetAttr("index", i);
        data.UpdateInputDesc("x", data_desc);
        data.UpdateOutputDesc("y", data_desc);
        
        // // 记录入参信息
        inputInfos.push_back({mlirSymbol, data, data_desc});
        
        // 建立映射
        symbolMap[mlirSymbol] = {
            .orgSymbol = mlirSymbol,
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


    log("--Set graph inputs. The inputs size =[" + std::to_string(inputs.size()) + "]");
    for (const auto& inputOp : inputs) {
        log("--Input operator name=[" + std::string(getOpName(inputOp)) + "]");
    }

    // ==================================
    // build ge oprators
    // ==================================
    // 算子计数器（用于生成唯一符号）
    int opCounter = 0;
    std::vector<ge::Operator> outputs;

    for (mlir::Operation& op : entryBlock.getOperations()) {
        log("================================Start to deal with the op:" + op.getName().getStringRef().str());
        // ReturnOp
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            SymbolInfo symbolInfo = processReturnOp(op);
            log("Return op is " + std::string(getOpName(symbolInfo.geOp)));
            outputs.emplace_back(symbolInfo.geOp);
            continue;
        }
        
        // 处理普通操作
        processOperation(op, opCounter, graphPtr);
        opCounter++;
    }
    
    log("--Set graph outputs. The outputs size =[" + std::to_string(outputs.size()) + "]");
    for (const auto& outputOp : outputs) {
        log("--Output operator name=[" + std::string(getOpName(outputOp)) + "]");
    }
    
    graphPtr->SetInputs(inputs).SetOutputs(outputs);

    log("============Finish All Operator");

    std::vector<GNode> nodes = graphPtr->GetAllNodes();
    log("============Get All GNodes size:" + std::to_string(nodes.size()));
    for (size_t i = 0; i < nodes.size(); i++) {
        ge::AscendString nodeName;
        nodes.at(i).GetName(nodeName);
        llvm::errs() << "Nodes [" << i << "] name[" << nodeName.GetString()<<"]\n";
    }
    

    return Status::OK();
}


Exporter::SymbolInfo Exporter::processReturnOp(mlir::Operation& returnOp) {
    mlir::func::ReturnOp returnOpCast = mlir::cast<mlir::func::ReturnOp>(returnOp);
    
    mlir::Value returnValue = returnOpCast.getOperand(0);
    log("-------returnOp operand:" + mlir::debugString(returnValue));
    std::string mlirSymbol = getMlirSymbol(returnValue);
    
    // 查找真实输出符号
    auto it = symbolMap.find(mlirSymbol);
    if (it == symbolMap.end()) {
        llvm::errs() << "Find [" << mlirSymbol << "] failed !!!\n";
    }
    
    const SymbolInfo& symbolInfo = it->second;
    printSymbolInfo(symbolInfo);
    llvm::errs() << "图输出: MLIR符号=" << mlirSymbol << "\n";
    
    return symbolInfo;
}

void Exporter::printSymbolInfo(const SymbolInfo& symbolInfo) {
   llvm::errs() << "SymbolInfo = mlirSymbol:[" << symbolInfo.orgSymbol << "]  opName:[" << getOpName(symbolInfo.geOp) <<"]" << "\n";
}

bool Exporter::processOperation(mlir::Operation& mlirOp, int opCounter, ge::GraphPtr &graphPtr) {
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
    ge::Operator geOp = ge::OperatorFactory::CreateOperator((opType + std::to_string(opCounter)).c_str(), opType.c_str());
    // ge::Operator geOp;
    
    // 3. 处理输入
    if (!processOperationInputs(mlirOp, geOp, opType)) {
        return false;
    }
    // geOp.SetInput("x1",symbolMap.find("%arg0")->second.geOp);
    // geOp.SetInput("x2",symbolMap.find("%arg1")->second.geOp);

    llvm::errs() << "Input number of this op is " << geOp.GetInputsSize() << "\n";
    
    // 4. 处理输出
    if (!processOperationOutputs(mlirOp, geOp, opCounter, opType)) {
        return false;
    }

    llvm::errs() << "Output number of this op is " << geOp.GetOutputsSize() << "\n";

    return true;
}
bool Exporter::processOperationInputs(mlir::Operation& mlirOp, ge::Operator& geOp, std::string opType) {
    if (mlirOp.getName().getStringRef() == "mair.ConcatV2") {
        auto concatV2Op = dyn_cast<mlir::air::ConcatV2Op>(mlirOp);
        mlir::ValueRange x_value = concatV2Op.getX();
        geOp.DynamicInputRegisterByIndex("x", x_value.size(), 0);
        unsigned i = 0;
        for (mlir::Value val : x_value) {
            log("----------trans the input x of ConcatV2: " + mlir::debugString(val));
            // 获取MLIR符号
            std::string mlirSymbol = getMlirSymbol(val);
            if (mlirSymbol.empty()) {
                llvm::errs() << "错误：无法获取操作数符号\n";
                return false;
            }
            // 查找真实geOperator
            auto it = symbolMap.find(mlirSymbol);
            if (it == symbolMap.end()) {
                llvm::errs() << "错误：未找到符号映射: " << mlirSymbol << "\n";
                return false;
            }

            log("Find input :");
            printSymbolInfo(it->second);
            ge::Operator inputOp = it->second.geOp;
            geOp.SetInput(i++, inputOp, 0U);
        }

        mlir::Value dim_value = concatV2Op.getConcatDim();
        const std::string* inputName= getGeOpInputName(opType, 1);
        log("----------trans the input dim of ConcatV2: " + mlir::debugString(dim_value));

        // 获取MLIR符号
        std::string mlirSymbol = getMlirSymbol(dim_value);
        if (mlirSymbol.empty()) {
            llvm::errs() << "错误：无法获取操作数符号\n";
            return false;
        }

        // 查找真实geOperator
        auto it = symbolMap.find(mlirSymbol);
        if (it == symbolMap.end()) {
            llvm::errs() << "错误：未找到符号映射: " << mlirSymbol << "\n";
            return false;
        }
        log("Find input :");
        printSymbolInfo(it->second);
        ge::Operator inputOp = it->second.geOp;
        geOp.SetInput(inputName->c_str(), inputOp);
        return true;
    }

    unsigned numOperands = mlirOp.getNumOperands();
    for (unsigned i = 0; i < numOperands; i++) {
        mlir::Value operand = mlirOp.getOperand(i);
        log("----------trans the input: " + mlir::debugString(operand));
        
        // 获取MLIR符号
        std::string mlirSymbol = getMlirSymbol(operand);
        if (mlirSymbol.empty()) {
            llvm::errs() << "错误：无法获取操作数符号\n";
            return false;
        }
        
        // 查找真实geOperator
        auto it = symbolMap.find(mlirSymbol);
        if (it == symbolMap.end()) {
            llvm::errs() << "错误：未找到符号映射: " << mlirSymbol << "\n";
            return false;
        }
        
        log("Find input :");
        printSymbolInfo(it->second);

        // 获取输入符号
        const std::string* inputName= getGeOpInputName(opType, i);

        ge::Operator inputOp = it->second.geOp;
        // 设置输入描述
        geOp.SetInput(inputName->c_str(), inputOp);
        
        llvm::errs() << "  Set opInput[" << i << "]: origin=" << mlirSymbol 
                    << ", inputName=[" << *inputName << "] opName=[" << getOpName(inputOp) <<"]\n";
    }
    
    return true;
}

const char * Exporter::getOpName(const ge::Operator& geOp) {
    ge::AscendString opName;
    geOp.GetName(opName);
    return opName.GetString();
}

// 处理操作输出
bool Exporter::processOperationOutputs(mlir::Operation& mlirOp, ge::Operator& geOp, int opCounter, std::string opType) {
    unsigned numResults = mlirOp.getNumResults();
    
    for (unsigned i = 0; i < numResults; i++) {
        mlir::Value result = mlirOp.getResult(i);
        
        // 获取MLIR符号
        std::string mlirSymbol = getMlirSymbol(result);
        if (mlirSymbol.empty()) {
            llvm::errs() << "错误：无法获取结果符号\n";
            return false;
        }
        
        // if (opType == "Const")
        // {
        //     mlir::Attribute valueAttr = mlirOp.getAttr("value");
        //     float constData = extractFloatValueSimple(valueAttr);
        //     llvm::errs() << "Const operatro. mlir::Attribute valueAttr  = " << valueAttr << " ; constData = " << constData << "\n";

        //             // 获取参数类型和形状
        //     mlir::TensorType arg_type = mlir::dyn_cast<mlir::TensorType>(result.getType());
        //     ge::DataType dtype;
        //     ConvertToDataType(arg_type.getElementType(), &dtype);
        //     auto shape = ge::Shape();
        //     ConvertToShape(arg_type, &shape);
        //     auto format = ge::FORMAT_ND;
        //     ge::TensorDesc data_desc(shape, format, dtype);

        //     uint32_t len = shape.GetShapeSize() * sizeof(float);
        //     ge::Tensor valueTensor(data_desc, reinterpret_cast<uint8_t*>(&constData), len);
        //     // if (!valueTensor.SetData(reinterpret_cast<uint8_t*>(&constData), len))
        //     // {
        //     //     log("Fail to Set the const out put value!!");
        //     // }
        //     geOp.SetAttr("value", valueTensor);
            
        //     log("Set the const out put value " + std::to_string(constData));
        // }

        if (opType == "Const") {
            // TODO
            mlir::Attribute valueAttr = mlirOp.getAttr("value");
            llvm::errs() << "Const operatro. mlir::Attribute valueAttr  = " << valueAttr << "\n";
            if (auto denseAttr = mlir::dyn_cast<mlir::ElementsAttr>(valueAttr)) {
                ge::Tensor valueTensor;
                if(!ConvertToTensor(denseAttr, &valueTensor).ok()) {
                    llvm::errs() << "Fail to convert to tensor!!  \n";
                }
                ge::TensorDesc desc = valueTensor.GetTensorDesc();
                ge::DataType dt = desc.GetDataType();
                ge::Shape shape = desc.GetShape();
                std::vector<int64_t> dims =  shape.GetDims();

                std::ostringstream oss;
                oss << "Const ge tensor : "
                    << "DataType:[" << DataTypeNames.find(dt)->second << "]; Shape dims=[";
                for(auto dim : dims) {
                    oss << dim << ",";
                }
                oss << "]";
                llvm::errs() << oss.str() << "\n";

                geOp.SetAttr("value", valueTensor);
                geOp.UpdateOutputDesc("y", desc);
            } else {
                llvm::errs() << "Failed Const operatro. Not ElementsAttr \n";
            }

            

            // if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
            //     if (denseAttr.isSplat() && denseAttr.getElementType().isF32()) {
            //         auto v = denseAttr.getSplatValue<float>();
            //         auto shape = ge::Shape();
            //         ge::TensorDesc data_desc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
            //         uint32_t len = shape.GetShapeSize() * sizeof(float);
            //         len = 4;
            //         std::cout << "Const float len: " << len << std::endl;
            //         ge::Tensor valueTensor(data_desc, reinterpret_cast<uint8_t*>(&v), len);
            //         geOp.SetAttr("value", valueTensor);
            //     }
            //     // 如果是f64，转换为float
            //     if (denseAttr.isSplat() && denseAttr.getElementType().isInteger(64)) {
            //         auto v = denseAttr.getSplatValue<int64_t>();
            //         auto shape = ge::Shape({1,});
            //         ge::TensorDesc data_desc(shape, ge::FORMAT_ND, ge::DT_INT64);
            //         uint32_t len = shape.GetShapeSize() * sizeof(int64_t);
            //         std::cout << "Const int64 len: " << len << std::endl;
            //         ge::Tensor valueTensor(data_desc, reinterpret_cast<uint8_t*>(&v), len);
            //         geOp.SetAttr("value", valueTensor);
            //     }
            // }
        }

        // 建立映射
        symbolMap[mlirSymbol] = {
            .orgSymbol = mlirSymbol,
            .geOp = geOp,
            // .tensorDesc = data_desc
        };
        
        llvm::errs() << "  输出[" << i << "]: MLIR符号=" << mlirSymbol << "\n";
    }
    
    return true;
}

float Exporter::extractFloatValueSimple(mlir::Attribute valueAttr) {
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(valueAttr)) {
        // 直接转换为float
        return floatAttr.getValue().convertToFloat();
    }

    // 如果是DenseElementsAttr（标量张量）
    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
        if (denseAttr.isSplat() && denseAttr.getElementType().isF32()) {
            return denseAttr.getSplatValue<float>();
        }
        // 如果是f64，转换为float
        if (denseAttr.isSplat() && denseAttr.getElementType().isF64()) {
            return static_cast<float>(denseAttr.getSplatValue<double>());
        }
    }
    
    // 如果是IntegerAttr（整数转换为浮点）
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(valueAttr)) {
        return static_cast<float>(intAttr.getValue().getSExtValue());
    }
    
    llvm::errs() << "  Fail to get value from " << valueAttr << "\n";
    // 默认返回0.0
    return 0.0f;
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

const std::string* Exporter::getGeOpInputName(const std::string& opName, int index) {
    auto it = inputNameMap.find(opName);
    if (it == inputNameMap.end()) {
        return nullptr;  
    }

    const std::vector<std::string>& inputNames = it->second;
    if (inputNames.size() >= index+1) {
        return &inputNames[index]; 
    }

    return nullptr;
}

} // namespace mlir::air