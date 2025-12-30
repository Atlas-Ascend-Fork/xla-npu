#ifndef STABLEHLO2MAIR_CONVERTER_H
#define STABLEHLO2MAIR_CONVERTER_H

#include "mlir/IR/BuiltinOps.h"

namespace xla {
namespace npu {
namespace mair {

    // 方言转换对外接口
    int Stablehlo2MairConverter(mlir::OwningOpRef<mlir::ModuleOp> &module);

} // namespace mair
} // namespace npu
} // namespace xla

#endif // STABLEHLO2MAIR_CONVERTER_H
