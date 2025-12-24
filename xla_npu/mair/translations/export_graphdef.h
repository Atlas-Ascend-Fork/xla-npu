#ifndef AIR_TRANSLATIONS_EXPORT_GRAPHDEF_H_
#define AIR_TRANSLATIONS_EXPORT_GRAPHDEF_H_

#include <unordered_set>
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"    // from @llvm-project

#include "./status.h"
// #include "translations/statusor.h"
// #include "utils/utils.h"

namespace mlir::air {

class Exporter {
public:
    static Status buildMlirAirGraphOp(mlir::ModuleOp module);

private:
    /* data */
    Exporter(/* args */);
    ~Exporter();
};


} // namespace mlir::air


#endif // AIR_TRANSLATIONS_EXPORT_GRAPHDEF_H_