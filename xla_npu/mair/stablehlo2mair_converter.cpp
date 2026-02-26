#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "transforms/mair_passes.h"
#include "stablehlo2mair_converter.h"

namespace xla {
namespace mair {
    int Stablehlo2MairConverter(mlir::ModuleOp &module) {
        mlir::PassManager pm(module.getContext());

        // 添加Stablehlo到Mair方言的转换Pass
        pm.addPass(mlir::air::createLegalizeStablehloToGePass());

        if (failed(pm.run(module)))
            return 4;

        return 0;
    }
}
}
