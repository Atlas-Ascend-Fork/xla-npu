#ifndef TRANSFORMS_MAIR_PASSES_H_
#define TRANSFORMS_MAIR_PASSES_H_
#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace air {
std::unique_ptr<Pass> createLegalizeStablehloToGePass();
}  // namespace air
}  // namespace mlir

#endif  // TRANSFORMS_MAIR_PASSES_H_
