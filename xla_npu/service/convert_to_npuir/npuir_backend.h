#ifndef NPUIR_BACKEND_H_
#define NPUIR_BACKEND_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/CAPI/Registration.h"

#include "mlir-c/IR.h"

#include "xla/xla_npu/dependency/external/npuir/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusion.h"

namespace mlir::hfusion {

class GraphGenTester {
public:
    static void test();
};

}

#endif NPUIR_BACKEND_H_
