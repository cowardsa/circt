//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DatapathToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTDATAPATHTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace datapath;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
struct CompressOpConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ValueRange operands = adaptor.getOperands();
    if (operands.size() < 3)
      return failure();

    ValueRange results = op.getResults();
    if (results.size() < 2)
      return failure();

    Value operandRunner = operands[0];
    for (Value operand : operands.drop_front())
      operandRunner =
          rewriter.create<smt::BVAddOp>(op.getLoc(), operandRunner, operand);

    SmallVector<Value, 2> newResults;
    Value resultRunner;
    for (Value result : results) {
      auto declareFunOp = rewriter.create<smt::DeclareFunOp>(
          op.getLoc(), typeConverter->convertType(result.getType()));
      newResults.push_back(declareFunOp.getResult());
      if (newResults.size() > 1)
        resultRunner = rewriter.create<smt::BVAddOp>(op.getLoc(), resultRunner,
                                                     declareFunOp);
      else
        resultRunner = declareFunOp;
    }

    auto premise =
        rewriter.create<smt::EqOp>(op.getLoc(), operandRunner, resultRunner);
    rewriter.create<smt::AssertOp>(op.getLoc(), premise);

    if (newResults.size() != results.size())
      return rewriter.notifyMatchFailure(op, "expected same number of results");

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Datapath to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertDatapathToSMTPass
    : public circt::impl::ConvertDatapathToSMTBase<ConvertDatapathToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateDatapathToSMTConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CompressOpConversion>(converter, patterns.getContext());
}

void ConvertDatapathToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<datapath::DatapathDialect>();
  target.addLegalDialect<smt::SMTDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateDatapathToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
