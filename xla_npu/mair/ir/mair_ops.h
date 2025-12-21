#ifndef MAIR_OPS_H_
#define MAIR_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

namespace mlir {
namespace air {

class AirDialect : public Dialect {
 public:
  explicit AirDialect(MLIRContext *context) : Dialect(getDialectNamespace(), context, TypeID::get<AirDialect>()) {
    initialize();
  }
  static StringRef getDialectNamespace() {
    return "air";
  }
  void initialize();
};

}  // end namespace air
}  // end namespace mlir

#define GET_OP_CLASSES
#include "ir/mair_ops.h.inc"

#endif  //  MAIR_OPS_H_
