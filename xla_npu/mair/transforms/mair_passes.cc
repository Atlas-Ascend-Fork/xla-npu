#include "mair_passes.h"

#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#include <iostream>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"          // from @llvm-project
#include "mlir/IR/MLIRContext.h"                // from @llvm-project
#include "mlir/IR/PatternMatch.h"               // from @llvm-project
#include "mlir/Pass/Pass.h"                     // from @llvm-project
#include "mlir/Support/LLVM.h"                  // from @llvm-project
#include "mlir/Support/LogicalResult.h"         // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

#include "stablehlo/dialect/StablehloOps.h"

#include "../ir/mair_ops.h"

#include "transforms/mair_passes.h.inc"

namespace mlir {
namespace air {
namespace {
    class ConvertAddOp : public OpConversionPattern<stablehlo::AddOp> {
        public:
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(stablehlo::AddOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter) const final {
                rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getLhs(), op.getRhs());
                return success();
            };
    };
}

struct LegalizeToMairPass
    : PassWrapper<LegalizeToMairPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeToMairPass)
  StringRef getArgument() const override { return "stablehlo-to-mair"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AirDialect>();
  }
  void runOnOperation() final;
};

void LegalizeToMairPass::runOnOperation() {
  MLIRContext &context = getContext();

  ConversionTarget target(context);

  target.addLegalDialect<AirDialect>();
  target.addIllegalDialect<stablehlo::StablehloDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertAddOp>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    getOperation().emitError("stablehlo to mair legalization failed.");
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLegalizeStablehloToGePass() {
  return std::make_unique<LegalizeToMairPass>();
}

} // namespace air
} // end namespace mlir