#include "./convert_type.h"

#include "./utils.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project

namespace mlir::air {

using mlir::Builder;
using mlir::ShapedType;
using mlir::Type;

Status ConvertScalarTypeToDataType(Type type, DataType *dtype) {
  if (type.isF16()) {
    *dtype = DT_FLOAT16;
    return Status::OK();
  } else if (type.isF32()) {
    *dtype = DT_FLOAT;
    return Status::OK();
  } else if (type.isF64()) {
    *dtype = DT_DOUBLE;
    return Status::OK();
  } else if (type.isBF16()) {
    *dtype = DT_BF16;
    return Status::OK();
  } else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (itype.getWidth()) {
      case 1:
        *dtype = DT_BOOL;
        return Status::OK();
      case 8:
        *dtype = itype.isUnsigned() ? DT_UINT8 : DT_INT8;
        return Status::OK();
      case 16:
        *dtype = itype.isUnsigned() ? DT_UINT16 : DT_INT16;
        return Status::OK();
      case 32:
        *dtype = itype.isUnsigned() ? DT_UINT32 : DT_INT32;
        return Status::OK();
      case 64:
        *dtype = itype.isUnsigned() ? DT_UINT64 : DT_INT64;
        return Status::OK();
      default:
        log("Converting " + debugString(type) + " to DataType. [IntegerType_NotFound]");
        return Status::OK();
    }
  } else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    auto etype = complex_type.getElementType();
    if (etype.isF32()) {
      *dtype = DT_COMPLEX64;
      return Status::OK();
    } else if (etype.isF64()) {
      *dtype = DT_COMPLEX128;
      return Status::OK();
    }
    log("Converting " + debugString(type) + " to DataType. [ComplextType_NotFound]");
    return Status::OK();
  }
  log("Converting " + debugString(type) + " to DataType. [NotFound]");
  return Status::OK();
}

Status ConvertToDataType(Type type, DataType *dtype) {
  if (auto stype = mlir::dyn_cast<ShapedType>(type)) {
    ConvertScalarTypeToDataType(stype.getElementType(), dtype);
  } else {
    ConvertScalarTypeToDataType(type, dtype);
  }
  return Status::OK();
}

Status ConvertToShape(mlir::ShapedType type, Shape *shape) {
  std::vector<int64_t> dims;
  if (type.hasRank()) {
    std::ostringstream oss;
    oss << "[";
    for (int64_t dim : type.getShape()) {
      dims.push_back(dim);
      oss << dim << ",";
    }
    oss << "]";
    log("ConvertToShape: dims " +  oss.str());
    *shape = ge::Shape(dims);
  } else {
    *shape = ge::Shape();
  }
  
  return Status::OK();
}

}  // namespace air
