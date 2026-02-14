#ifndef AIR_UTILS_CONVERT_TENSOR_H_
#define AIR_UTILS_CONVERT_TENSOR_H_

#include "./status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"    // from @llvm-project
#include "xla/xla_npu/third_party/metadef/external/graph/tensor.h"
#include "xla/xla_npu/third_party/metadef/external/graph/types.h"

using namespace ge;

namespace mlir::air {

// Converts an MLIR elements attribute to a ascend tensor.
Status ConvertToTensor(mlir::ElementsAttr attr, Tensor* output_tensor);

}  // namespace mlir::air

#endif  // AIR_UTILS_CONVERT_TENSOR_H_
