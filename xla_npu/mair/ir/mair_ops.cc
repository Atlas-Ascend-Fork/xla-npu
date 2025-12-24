#include "mair_ops.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_OP_CLASSES
#include "ir/mair_ops.cc.inc"

namespace mlir {
namespace air {

void AirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ir/mair_ops.cc.inc"
  >();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

void GraphOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {}

}  // namespace air
}  // namespace mlir
