#ifndef AIR_TRANSLATIONS_EXPORT_GRAPHDEF_H_
#define AIR_TRANSLATIONS_EXPORT_GRAPHDEF_H_

#include <unordered_map>
#include <iostream>
#include <string_view>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/IR/Location.h"                // from @llvm-project
#include "mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/Pass/Pass.h"                  // from @llvm-project
#include "mlir/Pass/PassManager.h"           // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"      // from @llvm-project
#include "xla/xla_npu/third_party/metadef/external/graph/graph.h"
#include "xla/xla_npu/third_party/metadef/external/graph/tensor.h"
#include "xla/xla_npu/third_party/metadef/external/graph/gnode.h"
#include "xla/xla_npu/third_party/metadef/external/graph/ascend_string.h"
#include "xla/xla_npu/third_party/metadef/external/graph/operator_factory.h"
#include "xla/xla_npu/third_party/metadef/external/graph/operator.h"
#include "xla/xla_npu/third_party/register/aclInterface.h"
#include "xla/xla_npu/third_party/air/external/ge/ge_api.h"

#include "./status.h"
// #include "translations/statusor.h"
// #include "utils/utils.h"

namespace mlir::air {

class Exporter {
public:
    
    Status Convert(mlir::ModuleOp module, std::unique_ptr<ge::Graph> &graph,
                        std::vector<ge::TensorDesc> &outputTensorDesc,
                        std::map<std::int64_t, std::int64_t> &inputOutputMap);
    explicit Exporter() {}


private:
    struct SymbolInfo {
        std::string orgSymbol;  // 原始符号，如"%arg0"
        std::string realSymbol;  // 真实符号，如"input_0", "op_1_out_0"
        ge::Operator geOp; // 对应的算子
        ge::TensorDesc tensorDesc;
    };

    static const std::unordered_map<std::string, std::vector<std::string>> inputNameMap;

    SymbolInfo processReturnOp(mlir::Operation& returnOp);
    bool processOperation(mlir::Operation& mlirOp, int opCounter);
    bool processOperationInputs(mlir::Operation& mlirOp, ge::Operator& geOp, std::string opType);
    bool processOperationOutputs(mlir::Operation& mlirOp, ge::Operator& geOp, int opCounter);

    std::string getMlirSymbol(mlir::Value value);
    int getOperationNumber(mlir::Operation* op);
    std::string generateOutputSymbol(int opIndex, int outputIndex);

    std::string_view getGeOpInputName(const std::string& opName, int index);

    std::unordered_map<std::string, SymbolInfo> symbolMap;
    std::vector<SymbolInfo> inputInfos;
};


} // namespace mlir::air


#endif // AIR_TRANSLATIONS_EXPORT_GRAPHDEF_H_