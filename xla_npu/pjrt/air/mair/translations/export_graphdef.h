/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
#include "xla/xla_npu/dependency/header/metadef/graph/graph.h"
#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "xla/xla_npu/dependency/header/metadef/graph/gnode.h"
#include "xla/xla_npu/dependency/header/metadef/graph/ascend_string.h"
#include "xla/xla_npu/dependency/header/metadef/graph/operator_factory.h"
#include "xla/xla_npu/dependency/header/metadef/graph/operator.h"
#include "xla/xla_npu/dependency/interface/acl_interface.h"
#include "xla/xla_npu/dependency/header/ge/ge_api.h"

#include "./status.h"
// #include "translations/statusor.h"
// #include "utils/utils.h"

namespace mlir::air {

class Exporter {
public:
    
    Status Convert(mlir::ModuleOp& module, ge::GraphPtr &graphPtr);
    explicit Exporter() {}


private:
    struct SymbolInfo {
        std::string orgSymbol;  // 原始符号，如"%arg0"
        ge::Operator geOp; // 对应的算子
        ge::TensorDesc tensorDesc;
    };

    static const std::unordered_map<std::string, std::vector<std::string>> inputNameMap;

    SymbolInfo processReturnOp(mlir::Operation& returnOp);
    bool processOperation(mlir::Operation& mlirOp, int opCounter, ge::GraphPtr &graphPtr);
    bool processOperationInputs(mlir::Operation& mlirOp, ge::Operator& geOp, std::string opType);
    bool processOperationOutputs(mlir::Operation& mlirOp, ge::Operator& geOp, int opCounter, std::string opType);

    std::string getMlirSymbol(mlir::Value value);
    int getOperationNumber(mlir::Operation* op);
    std::string generateOutputSymbol(int opIndex, int outputIndex);

    const std::string* getGeOpInputName(const std::string& opName, int index);

    float extractFloatValueSimple(mlir::Attribute valueAttr);
    void printSymbolInfo(const SymbolInfo& symbolInfo);
    const char * getOpName(const ge::Operator& geOp);

    std::unordered_map<std::string, SymbolInfo> symbolMap;
    std::vector<SymbolInfo> inputInfos;
};


} // namespace mlir::air


#endif // AIR_TRANSLATIONS_EXPORT_GRAPHDEF_H_