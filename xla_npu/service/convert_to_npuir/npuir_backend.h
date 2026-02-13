#ifndef NPUIR_BACKEND_H_
#define NPUIR_BACKEND_H_

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/xla_npu/dependency/external/npuir/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusion.h"

namespace mlir::hfusion {

class GraphGenTester {
   public:
    static void test();
};

}  // namespace mlir::hfusion

#endif NPUIR_BACKEND_H_
