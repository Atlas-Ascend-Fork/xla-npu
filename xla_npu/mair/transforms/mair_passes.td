include "mlir/Pass/PassBase.td"
//FuncOp  //ModuleOp
def LegalizeToMairPass : Pass<"stablehlo-legalize-to-mair", "mlir::shape::FuncOp"> {
  let summary = "Legalize from stableHLO dialect to mair dialect.";
  let constructor = "createLegalizeStablehloToGePass()";
}
