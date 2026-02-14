#ifndef STABLEHLO2MAIR_CONVERTER_H
#define STABLEHLO2MAIR_CONVERTER_H

#include "mlir/IR/BuiltinOps.h"

namespace xla {
namespace mair {

// 方言转换对外接口
int Stablehlo2MairConverter(mlir::ModuleOp& module);

}  // namespace mair
}  // namespace xla

#endif  // STABLEHLO2MAIR_CONVERTER_H
