#ifndef AIR_UTILS_CONVERT_TYPE_H_
#define AIR_UTILS_CONVERT_TYPE_H_

#include "mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/IR/Types.h"         // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project

#include "xla/xla_npu/dependency/header/metadef/graph/types.h"
#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "./status.h"
#include "./utils.h"

using namespace ge;

namespace mlir::air {

// Converts a scalar MLIR type to a TensorFlow Datatype.
Status ConvertScalarTypeToDataType(mlir::Type type, DataType *dtype);

// Converts an MLIR type to TensorFlow DataType. If 'type' is a scalar type, it
// is converted directly. If it is a shaped type, the element type is converted.
Status ConvertToDataType(mlir::Type type, DataType *dtype);

Status ConvertToShape(mlir::ShapedType type, ge::Shape *shape);

}  // namespace air

#endif  // ASCEND_UTILS_CONVERT_TYPE_H_
