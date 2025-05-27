//===- CombToDatapath.cpp - Comb to Datapath Conversion Pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to Datapath Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToDatapath.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/PointerUnion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTODATAPATH
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

// A wrapper for comb::extractBits that returns a SmallVector<Value>.
static SmallVector<Value> extractBits(OpBuilder &builder, Value val) {
  SmallVector<Value> bits;
  comb::extractBits(builder, val, bits);
  return bits;
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

template <typename OpTy>
struct CombLowerVariadicOp : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = lowerFullyAssociativeOp(op, op.getOperands(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

  static Value lowerFullyAssociativeOp(OpTy op, OperandRange operands,
                                       ConversionPatternRewriter &rewriter) {
    Value lhs, rhs;
    switch (operands.size()) {
    case 0:
      assert(false && "cannot be called with empty operand range");
      break;
    case 1:
      return operands[0];
    case 2:
      lhs = operands[0];
      rhs = operands[1];
      return rewriter.create<OpTy>(op.getLoc(), ValueRange{lhs, rhs}, true);
    default:
      auto firstHalf = operands.size() / 2;
      lhs =
          lowerFullyAssociativeOp(op, operands.take_front(firstHalf), rewriter);
      rhs =
          lowerFullyAssociativeOp(op, operands.drop_front(firstHalf), rewriter);
      return rewriter.create<OpTy>(op.getLoc(), ValueRange{lhs, rhs}, true);
    }
  }
};


struct CombAddOpConversion : OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputs = adaptor.getInputs();
    // Lower only when there are two inputs.
    // Variadic operands must be lowered in a different pattern.
    if (inputs.size() <= 2)
      return failure();

    auto width = op.getType().getIntOrFloatBitWidth();
    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(), 0);
      return success();
    }

    auto results =
          rewriter.create<datapath::CompressOp>(op.getLoc(), op.getOperands(), 2);

    rewriter.replaceOpWithNewOp<comb::AddOp>(op, results.getResults(), true);
    return success();
  }
};

// struct CombSubOpConversion : OpConversionPattern<SubOp> {
//   using OpConversionPattern<SubOp>::OpConversionPattern;
//   LogicalResult
//   matchAndRewrite(SubOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto lhs = op.getLhs();
//     auto rhs = op.getRhs();
//     // Since `-rhs = ~rhs + 1` holds, rewrite `sub(lhs, rhs)` to:
//     // sub(lhs, rhs) => add(lhs, -rhs) => add(lhs, add(~rhs, 1))
//     // => add(lhs, ~rhs, 1)
//     auto notRhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), rhs,
//                                                       /*invert=*/true);
//     auto one = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(), 1);
//     rewriter.replaceOpWithNewOp<comb::AddOp>(op, ValueRange{lhs, notRhs, one},
//                                              true);
//     return success();
//   }
// };

struct CombMulOpConversion : OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<MulOp>::OpAdaptor;
  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 2)
      return failure();

    // FIXME: Currently it's lowered to a really naive implementation that
    // chains add operations.

    // a_{n}a_{n-1}...a_0 * b
    // = sum_{i=0}^{n} a_i * 2^i * b
    // = sum_{i=0}^{n} (a_i ? b : 0) << i
    int64_t width = op.getType().getIntOrFloatBitWidth();
    
    auto pp = rewriter.create<datapath::PartialProductOp>(
        op.getLoc(), op.getInputs(), width);

    rewriter.replaceOpWithNewOp<comb::AddOp>(op, pp.getResults(), true);
    return success();
  }
};


} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to AIG pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToDatapathPass
    : public impl::ConvertCombToDatapathBase<ConvertCombToDatapathPass> {
  void runOnOperation() override;
  using ConvertCombToDatapathBase<ConvertCombToDatapathPass>::ConvertCombToDatapathBase;
};
} // namespace

static void
populateCombToDatapathConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
      // Arithmetic Ops
      CombAddOpConversion, CombMulOpConversion
      >
      (patterns.getContext());
}

void ConvertCombToDatapathPass::runOnOperation() {
  ConversionTarget target(getContext());

  // Keep data movement operations like Extract, Concat and Replicate.
  target.addLegalOp<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
                    hw::BitcastOp, hw::ConstantOp>();

  // Datapath is target dialect.
  target.addLegalDialect<datapath::DatapathDialect, comb::CombDialect>();

  target.addIllegalOp<comb::AddOp, comb::MulOp>();

  target.addDynamicallyLegalOp<comb::AddOp>([](comb::AddOp op) {
    return op.getInputs().size() == 2;
  });

  RewritePatternSet patterns(&getContext());
  populateCombToDatapathConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
