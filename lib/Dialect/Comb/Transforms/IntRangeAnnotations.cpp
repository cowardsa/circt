//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Analysis/DataFlowFramework.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace circt::comb;
using namespace mlir;
using namespace mlir::dataflow;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_COMBINTRANGEANNOTATING
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

/// Gather ranges for all the values in `values`. Appends to the existing
/// vector.
static LogicalResult collectRanges(DataFlowSolver &solver, ValueRange values,
                                   SmallVectorImpl<ConstantIntRanges> &ranges) {
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<IntegerValueRangeLattice>(val);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return failure();

    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    ranges.push_back(inferredRange);
  }
  return success();
}

namespace {
template <typename CombOpTy>
struct CombOpAnnotate : public OpRewritePattern<CombOpTy> {
  CombOpAnnotate(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern<CombOpTy>(context), solver(s) {}

  LogicalResult matchAndRewrite(CombOpTy op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("comb.upper_bit_trunc"))
      return failure();

    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "Only support binary operations with one result");

    SmallVector<ConstantIntRanges> ranges;
    if (failed(collectRanges(solver, op->getOperands(), ranges)))
      return rewriter.notifyMatchFailure(op, "input without specified range");

    bool overflowed = false;
    auto a = ranges[0].umax();
    auto b = ranges[1].umax();
    if (isa<comb::AddOp>(op)) {
      a.uadd_ov(b, overflowed);
    } else if (isa<comb::MulOp>(op)) {
      a.umul_ov(b, overflowed);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported comb op");
    }

    op->setAttr("comb.upper_bit_trunc",
                BoolAttr::get(op->getContext(), overflowed));
    return success();
  }

private:
  DataFlowSolver &solver;
};

struct CombIntRangeAnnotatingPass
    : comb::impl::CombIntRangeAnnotatingBase<CombIntRangeAnnotatingPass> {

  using CombIntRangeAnnotatingBase::CombIntRangeAnnotatingBase;
  void runOnOperation() override;
};
} // namespace

void CombIntRangeAnnotatingPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  RewritePatternSet patterns(ctx);
  patterns.add<CombOpAnnotate<comb::AddOp>, CombOpAnnotate<comb::MulOp>,
               CombOpAnnotate<comb::SubOp>>(patterns.getContext(), solver);

  if (failed(applyPatternsGreedily(op, std::move(patterns))))
    signalPassFailure();
}
