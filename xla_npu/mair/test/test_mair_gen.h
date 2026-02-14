#ifndef TEST_MAIR_GEN_H_
#define TEST_MAIR_GEN_H_

#include <iostream>
#include <vector>

#include "../ir/mair_ops.h"
#include "../translations/export_graphdef.h"
#include "../translations/status.h"
#include "../translations/utils.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::air {

class GraphGenTester {
   public:
    static Status Convert(mlir::ModuleOp& module, ge::GraphPtr& graphPtr);

    static void buildGeluModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildAddModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildAddMulModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildAEmptyModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
    static void buildConstModule(mlir::ModuleOp& module, mlir::MLIRContext& context);
};

}  // namespace mlir::air

#endif  // TEST_MAIR_GEN_H_
