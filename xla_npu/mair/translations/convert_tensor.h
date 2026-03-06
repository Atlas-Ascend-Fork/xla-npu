#ifndef AIR_UTILS_CONVERT_TENSOR_H_
#define AIR_UTILS_CONVERT_TENSOR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"    // from @llvm-project

#include "xla/xla_npu/dependency/header/metadef/graph/types.h"
#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "./status.h"

using namespace ge;

namespace mlir::air {

// Converts an MLIR elements attribute to a ascend tensor.
Status ConvertToTensor(mlir::ElementsAttr attr, Tensor *output_tensor);

}  // namespace air

#endif  // AIR_UTILS_CONVERT_TENSOR_H_
