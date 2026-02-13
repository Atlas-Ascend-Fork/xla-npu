#include "npuir_backend.h"

using namespace mlir;
using namespace mlir::hfusion;

void GraphGenTester::test() {
    static mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    // context.getOrLoadDialect<mlir::hfusion::HFusionDialect>();
    // context.getOrLoadDialect<mlir::linalg::LinalgDialect>();

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
}
