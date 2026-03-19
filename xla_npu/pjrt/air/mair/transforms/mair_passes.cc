/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

#include "xla/xla_npu/pjrt/air/mair/transforms/mair_passes.h.inc"

namespace mlir {
namespace air {
namespace {
std::uint32_t getReduceTypeString(mlir::stablehlo::AllReduceOp allReduceOp) {
  // 获取 computation 区域
  mlir::Region& computationRegion = allReduceOp.getComputation();

  // 检查区域是否为空
  if (computationRegion.empty()) {
    return -1;
  }

  // 获取区域中的第一个块（通常只有一个块）
  mlir::Block& block = computationRegion.front();

  // 遍历块中的操作
  for (mlir::Operation& op : block) {
    // 检查是否是 add 操作
    if (auto addOp = mlir::dyn_cast<mlir::stablehlo::AddOp>(&op)) {
      return 0;
    }
    // 检查是否是 maximum 操作（reduce max）
    else if (auto maxOp = mlir::dyn_cast<mlir::stablehlo::MaxOp>(&op)) {
      return 1;
    }
    // 检查是否是 minimum 操作（reduce min）
    else if (auto minOp = mlir::dyn_cast<mlir::stablehlo::MinOp>(&op)) {
      return 2;
    }
    // 检查是否是 multiply 操作（reduce product）
    else if (auto mulOp = mlir::dyn_cast<mlir::stablehlo::MulOp>(&op)) {
      return 3;
    }
  }
  return -1;
}
std::int64_t getAllGatherRankSize(stablehlo::AllGatherOp allGatherOp) {
  auto replicaGroupsAttr = allGatherOp.getReplicaGroupsAttr();
  int64_t rankSize = -1;
  if (replicaGroupsAttr && replicaGroupsAttr.getType().getRank() >= 2) {
    auto shape = replicaGroupsAttr.getType().getShape();
    rankSize = shape[1];
  }
  return rankSize;
}
class ConvertAddOp : public OpConversionPattern<stablehlo::AddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertMulOp : public OpConversionPattern<stablehlo::MulOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};

// 暂未考虑BroadcastInDim包含转置的情况，转置需要新增transpose进行处理
class ConvertBroadcastInDimOp
    : public OpConversionPattern<stablehlo::BroadcastInDimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Value operand = op.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    auto broadcastDims = op.getBroadcastDimensions();
    auto operandShape = operandType.getShape();
    auto resultShape = resultType.getShape();
    // 维度减小，如果需要移除维度不为1，返回报错
    if (operandType.getRank() > resultType.getRank()) {
      llvm::SmallDenseSet<int64_t> operandDimsUsedInResult(
          broadcastDims.begin(), broadcastDims.end());
      SmallVector<int64_t> axesToSqueeze;
      for (int i = 0; i < operandType.getRank(); ++i) {
        if (operandDimsUsedInResult.count(i) == 0) {
          if (operandShape[i] != 1) {
            return failure();
          }
          axesToSqueeze.push_back(i);
        }
      }

      auto axesTensorType = RankedTensorType::get(
          {static_cast<int64_t>(axesToSqueeze.size())}, rewriter.getI64Type());
      auto axesAttr = DenseIntElementsAttr::get(axesTensorType, axesToSqueeze);
      auto axesConstOp = rewriter.create<air::ConstantOp>(
          op.getLoc(), axesTensorType, axesAttr);
      auto newOp = rewriter.create<air::SqueezeV3Op>(
          op.getLoc(), resultType, operand, axesConstOp.getResult());
      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }
    // 维度增加分两种情况, 如果新增维度为1直接增加，如果新增维度不为1, 先新增维度在执行BroadcastOp
    if (operandType.getRank() < resultType.getRank()) {
      llvm::SmallDenseSet<int64_t> resultDimsUsedInOperand(
          broadcastDims.begin(), broadcastDims.end());
      SmallVector<int64_t> axesToUnsqueeze;
      bool needBroadcastAfterUnsqueeze = false;
      for (int i = 0; i < resultType.getRank(); ++i) {
        if (resultDimsUsedInOperand.count(i) == 0) {
          if (resultShape[i] != 1) {
            needBroadcastAfterUnsqueeze = true;
          }
          axesToUnsqueeze.push_back(i);
        }
      }

      if (needBroadcastAfterUnsqueeze) {
        SmallVector<int64_t> intermediateShape;
        int operand_idx = 0;
        for (int i = 0; i < resultType.getRank(); ++i) {
          if (resultDimsUsedInOperand.count(i) > 0) {
            intermediateShape.push_back(operandShape[operand_idx]);
            operand_idx++;
          } else {
            intermediateShape.push_back(1);
          }
        }
        auto intermediateType = RankedTensorType::get(
            intermediateShape, resultType.getElementType());

        auto unsqueezeAxesTensorType = RankedTensorType::get(
            {static_cast<int64_t>(axesToUnsqueeze.size())},
            rewriter.getI64Type());
        auto unsqueezeAxesAttr =
            DenseIntElementsAttr::get(unsqueezeAxesTensorType, axesToUnsqueeze);
        auto unsqueezeAxesConstOp = rewriter.create<air::ConstantOp>(
            op.getLoc(), unsqueezeAxesTensorType, unsqueezeAxesAttr);

        auto unsqueezeOp = rewriter.create<air::UnsqueezeV3Op>(
            op.getLoc(), intermediateType, operand,
            unsqueezeAxesConstOp.getResult());
        Value intermediateValue = unsqueezeOp.getResult();

        auto targetShape = resultType.getShape();
        SmallVector<int64_t> shapeValues(targetShape.begin(),
                                         targetShape.end());
        auto shapeTensorType = RankedTensorType::get(
            {static_cast<int64_t>(shapeValues.size())}, rewriter.getI64Type());
        auto shapeAttr =
            DenseIntElementsAttr::get(shapeTensorType, shapeValues);
        auto shapeConstOp = rewriter.create<air::ConstantOp>(
            op.getLoc(), shapeTensorType, shapeAttr);
        auto newOp = rewriter.create<air::BroadcastOp>(
            op.getLoc(), resultType, intermediateValue,
            shapeConstOp.getResult());
        rewriter.replaceOp(op, newOp.getResult());
        return success();
      } else {
        auto axesTensorType = RankedTensorType::get(
            {static_cast<int64_t>(axesToUnsqueeze.size())},
            rewriter.getI64Type());
        auto axesAttr =
            DenseIntElementsAttr::get(axesTensorType, axesToUnsqueeze);
        auto axesConstOp = rewriter.create<air::ConstantOp>(
            op.getLoc(), axesTensorType, axesAttr);

        auto newOp = rewriter.create<air::UnsqueezeV3Op>(
            op.getLoc(), resultType, operand, axesConstOp.getResult());
        rewriter.replaceOp(op, newOp.getResult());
        return success();
      }
    }
    // 维度不变直接转换BroadcastOp
    auto target_shape = resultType.getShape();
    SmallVector<int64_t> shapeValues(target_shape.begin(), target_shape.end());

    auto shapeTensorType = RankedTensorType::get(
        {static_cast<int64_t>(shapeValues.size())}, rewriter.getI64Type());
    auto shapeAttr = DenseIntElementsAttr::get(shapeTensorType, shapeValues);
    auto shapeConstOp = rewriter.create<air::ConstantOp>(
        op.getLoc(), shapeTensorType, shapeAttr);
    auto newOp = rewriter.create<air::BroadcastOp>(
        op.getLoc(), resultType, operand, shapeConstOp.getResult());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  };
};
class ConvertConstantOp : public OpConversionPattern<stablehlo::ConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), op.getValue());
    return success();
  };
};
class ConvertTanhOp : public OpConversionPattern<stablehlo::TanhOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::TanhOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<TanhOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertAllReduceOp : public OpConversionPattern<stablehlo::AllReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::AllReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    std::uint32_t reduceType = getReduceTypeString(op);
    auto reduceTypeValue =
        rewriter
            .create<ConstantOp>(
                op.getLoc(), rewriter.getI32Type(),
                rewriter.getI32IntegerAttr(static_cast<int32_t>(reduceType)))
            .getResult();
    rewriter.replaceOpWithNewOp<HcomAllReduceOp>(
        op, op.getResult(0).getType(), op.getOperand(0), reduceTypeValue);
    return success();
  };
};
class ConvertAllGatherOp : public OpConversionPattern<stablehlo::AllGatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::AllGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    std::int64_t rankSize = getAllGatherRankSize(op);
    auto rankSizeValue =
        rewriter
            .create<ConstantOp>(op.getLoc(), rewriter.getI64Type(),
                                rewriter.getI64IntegerAttr(rankSize))
            .getResult();
    auto allGatherResult =
        rewriter
            .create<HcomAllGatherOp>(op.getLoc(), op.getResult(0).getType(),
                                     op.getOperand(0), rankSizeValue)
            .getResult();
    // 需要根据all_gather_dimension参数确定拼接维度
    std::uint64_t concatDim = op.getAllGatherDim();
    auto concatDimValue =
        rewriter
            .create<ConstantOp>(
                op.getLoc(), rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(static_cast<int64_t>(concatDim)))
            .getResult();
    auto concatResult =
        rewriter
            .create<ConcatV2Op>(op.getLoc(),
                                op.getResult(0).getType(),  // 最终结果类型
                                allGatherResult, concatDimValue)
            .getResult();

    rewriter.replaceOp(op, concatResult);
    return success();
  };
};
class ConvertConcatV2Op : public OpConversionPattern<stablehlo::ConcatenateOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ConcatenateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    mlir::ValueRange inputs = adaptor.getInputs();
    int64_t concatDim = op.getDimension();
    auto resultType = op.getType();
    auto indexType = rewriter.getI64Type();
    mlir::RankedTensorType scalarTensorType =
        mlir::RankedTensorType::get({}, indexType);
    mlir::DenseElementsAttr dimElementsAttr =
        mlir::DenseElementsAttr::get(scalarTensorType, concatDim);
    auto constantOp = rewriter.create<mlir::air::ConstantOp>(
        op.getLoc(), scalarTensorType, dimElementsAttr);
    mlir::Value concatDimValue = constantOp.getResult();
    auto newConcatOp = rewriter.create<mlir::air::ConcatV2Op>(
        op.getLoc(), resultType, inputs, concatDimValue,
        rewriter.getI64IntegerAttr(inputs.size()));
    rewriter.replaceOp(op, newConcatOp.getResult());
    return success();
  };
};
class ConvertDivOp : public OpConversionPattern<stablehlo::DivOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::DivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<DivOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertExpOp : public OpConversionPattern<stablehlo::ExpOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ExpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<ExpOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertMaxOp : public OpConversionPattern<stablehlo::MaxOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::MaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<MaxOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertNegOp : public OpConversionPattern<stablehlo::NegOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::NegOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<NegOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertSubOp : public OpConversionPattern<stablehlo::SubtractOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::SubtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<SubOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertMatMulOp : public OpConversionPattern<stablehlo::DotGeneralOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<MatMulOp>(op, op.getType(), op.getLhs(),
                                          op.getRhs(), nullptr, false, false);
    return success();
  };
};
class ConvertReshapeOp : public OpConversionPattern<stablehlo::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    mlir::Value inputTensor = adaptor.getOperand();
    auto resultType = dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "Result must be a ranked tensor");
    }
    auto targetShape = resultType.getShape();
    auto shapeAttr = rewriter.getI64TensorAttr(targetShape);
    auto shapeTensorType =
        mlir::RankedTensorType::get({static_cast<int64_t>(targetShape.size())},
                                    rewriter.getIntegerType(64));
    auto shapeTensor = rewriter
                           .create<mlir::air::ConstantOp>(
                               op.getLoc(), shapeTensorType, shapeAttr)
                           .getResult();
    auto newReshapeOp = rewriter.create<mlir::air::ReshapeOp>(
        op.getLoc(), op.getType(), inputTensor, shapeTensor
        // axis and num_axes use default values
    );
    rewriter.replaceOp(op, newReshapeOp.getResult());
    return success();
  };
};
class ConvertReduceOp : public OpConversionPattern<stablehlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto& body = op.getBody().front();
    auto it = body.begin();
    auto termIt = body.end();
    --termIt;

    if (it == termIt) {
      return rewriter.notifyMatchFailure(
          op, "Body must contain exactly one computation op.");
    }
    auto computeIt = it;
    auto afterComputeIt = std::next(computeIt);
    if (computeIt != body.begin() || afterComputeIt != termIt) {
      return rewriter.notifyMatchFailure(
          op, "Body must contain exactly one computation op.");
    }

    auto* singleComputeOp = &(*computeIt);

    bool isSum = isa<stablehlo::AddOp>(singleComputeOp);
    bool isMax = isa<stablehlo::MaxOp>(singleComputeOp);
    bool isMin = isa<stablehlo::MinOp>(singleComputeOp);

    if (!isSum && !isMax && !isMin) {
      return success();
    }

    auto inputsRange = adaptor.getInputs();
    if (inputsRange.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only single-input reduce is supported.");
    }
    mlir::Value inputs = inputsRange[0];

    auto dimensionsArrayRef = op.getDimensions();
    if (dimensionsArrayRef.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "Dimensions list cannot be empty.");
    }
    auto elementType = rewriter.getIntegerType(64);
    llvm::SmallVector<int64_t, 4> shapeVector;
    shapeVector.push_back(dimensionsArrayRef.size());
    auto tensorType = mlir::RankedTensorType::get(shapeVector, elementType);

    auto dimensionsAttr = rewriter.getI64TensorAttr(dimensionsArrayRef);

    auto axesValue = rewriter
                         .create<mlir::air::ConstantOp>(op.getLoc(), tensorType,
                                                        dimensionsAttr)
                         .getResult();

    bool keepDims = false;
    auto keepDimsAttr = rewriter.getBoolAttr(keepDims);

    mlir::Operation* newOp = nullptr;
    if (isSum) {
      newOp = rewriter.create<mlir::air::ReduceSumOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    } else if (isMax) {
      newOp = rewriter.create<mlir::air::ReduceMaxOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    } else if (isMin) {
      newOp = rewriter.create<mlir::air::ReduceMinOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    }
    if (newOp) {
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  };
};
class ConvertCastOp : public OpConversionPattern<stablehlo::ConvertOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ConvertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto inputValue = adaptor.getOperands()[0];
    auto dstType = op.getType();
    if (!dstType) {
      return rewriter.notifyMatchFailure(
          op, "failed to get destination type for convert op");
    }
    rewriter.replaceOpWithNewOp<CastOp>(op, dstType, inputValue,
                                        TypeAttr::get(dstType));
    return success();
  }
};
}  // namespace

struct LegalizeToMairPass
    : PassWrapper<LegalizeToMairPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeToMairPass)
  StringRef getArgument() const override { return "stablehlo-to-mair"; }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AirDialect>();
  }
  void runOnOperation() final;
};

void LegalizeToMairPass::runOnOperation() {
  MLIRContext& context = getContext();

  ConversionTarget target(context);

  target.addLegalDialect<AirDialect>();
  target.addIllegalDialect<stablehlo::StablehloDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertAddOp>(&getContext());
  patterns.add<ConvertMulOp>(&getContext());
  patterns.add<ConvertBroadcastInDimOp>(&getContext());
  patterns.add<ConvertConstantOp>(&getContext());
  patterns.add<ConvertTanhOp>(&getContext());
  patterns.add<ConvertAllReduceOp>(&getContext());
  patterns.add<ConvertAllGatherOp>(&getContext());
  patterns.add<ConvertDivOp>(&getContext());
  patterns.add<ConvertExpOp>(&getContext());
  patterns.add<ConvertMaxOp>(&getContext());
  patterns.add<ConvertNegOp>(&getContext());
  patterns.add<ConvertSubOp>(&getContext());
  patterns.add<ConvertMatMulOp>(&getContext());
  patterns.add<ConvertReshapeOp>(&getContext());
  patterns.add<ConvertReduceOp>(&getContext());
  patterns.add<ConvertConcatV2Op>(&getContext());
  patterns.add<ConvertCastOp>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("stablehlo to mair legalization failed.");
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLegalizeStablehloToGePass() {
  return std::make_unique<LegalizeToMairPass>();
}

}  // namespace air
}  // end namespace mlir