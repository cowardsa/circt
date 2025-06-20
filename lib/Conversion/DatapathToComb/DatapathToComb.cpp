//===- DatapathToComb.cpp--------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DatapathToComb.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/PointerUnion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTDATAPATHTOCOMB
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

// using namespace mlir;
using namespace circt;
using namespace datapath;

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
// Construct a full adder for three 1-bit inputs.
std::pair<Value, Value> fullAdder(ConversionPatternRewriter &rewriter,
                                  Location loc, Value a, Value b, Value c) {
  auto aXorB = rewriter.createOrFold<comb::XorOp>(loc, a, b, true);
  Value sum = rewriter.createOrFold<comb::XorOp>(loc, aXorB, c, true);

  auto carry = rewriter.createOrFold<comb::OrOp>(
      loc,
      ArrayRef<Value>{rewriter.createOrFold<comb::AndOp>(loc, a, b, true),
                      rewriter.createOrFold<comb::AndOp>(loc, aXorB, c, true)},
      true);

  return {sum, carry};
}

struct DatapathCompressOpConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();

    auto falseValue = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
    SmallVector<SmallVector<Value>> partialProducts;
    for (auto input : inputs) {
      partialProducts.push_back(
          extractBits(rewriter, input)); // Extract bits from each input
    } 

    // Wallace tree reduction
    rewriter.replaceOp(
        op,
        wallaceReduction(falseValue, width, rewriter, loc, partialProducts));
    return success();
}

private:
  // Perform Wallace tree reduction on partial products.
  // See https://en.wikipedia.org/wiki/Wallace_tree
  static SmallVector<Value>
  wallaceReduction(Value falseValue, size_t width,
                   ConversionPatternRewriter &rewriter, Location loc,
                   SmallVector<SmallVector<Value>> &partialProducts) {
    SmallVector<SmallVector<Value>> newPartialProducts;
    newPartialProducts.reserve(partialProducts.size());
    // Continue reduction until we have only two rows. The length of
    // `partialProducts` is reduced by 1/3 in each iteration.
    while (partialProducts.size() > 2) {
      newPartialProducts.clear();
      // Take three rows at a time and reduce to two rows(sum and carry).
      for (unsigned i = 0; i < partialProducts.size(); i += 3) {
        if (i + 2 < partialProducts.size()) {
          // We have three rows to reduce
          auto &row1 = partialProducts[i];
          auto &row2 = partialProducts[i + 1];
          auto &row3 = partialProducts[i + 2];

          assert(row1.size() == width && row2.size() == width &&
                 row3.size() == width);

          SmallVector<Value> sumRow, carryRow;
          sumRow.reserve(width);
          carryRow.reserve(width);
          carryRow.push_back(falseValue);

          // Process each bit position
          for (unsigned j = 0; j < width; ++j) {
            // Full adder logic
            auto [sum, carry] =
                fullAdder(rewriter, loc, row1[j], row2[j], row3[j]);
            sumRow.push_back(sum);
            if (j + 1 < width)
              carryRow.push_back(carry);
          }

          newPartialProducts.push_back(std::move(sumRow));
          newPartialProducts.push_back(std::move(carryRow));
        } else {
          // Add remaining rows as is
          newPartialProducts.append(partialProducts.begin() + i,
                                    partialProducts.end());
        }
      }

      std::swap(newPartialProducts, partialProducts);
    }

    assert(partialProducts.size() == 2);
    SmallVector<Value> carrySave;
    for (auto partialProduct : partialProducts) {
      // Reverse the order of the bits
      std::reverse(partialProduct.begin(), partialProduct.end());
      carrySave.push_back(rewriter.create<comb::ConcatOp>(loc, partialProduct));
    }
    // Use comb.add for the final addition.
    return carrySave;
  }
};                    


                    
struct DatapathPartialProductOpConversion : OpConversionPattern<PartialProductOp> {
  using OpConversionPattern<PartialProductOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PartialProductOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    

    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    Value a = inputs[0];
    Value b = inputs[1];
    unsigned width = a.getType().getIntOrFloatBitWidth();

    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(0), 0);
      return success();
    }

    // Extract individual bits from operands
    SmallVector<Value> aBits = extractBits(rewriter, a);
    SmallVector<Value> bBits = extractBits(rewriter, b);

    auto falseValue = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));

    // Generate partial products
    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      SmallVector<Value> row(i, falseValue);
      row.reserve(width);
      // Generate partial product bits
      for (unsigned j = 0; i + j < width; ++j)
        row.push_back(
            rewriter.createOrFold<comb::AndOp>(loc, aBits[j], bBits[i]));
      
      // Construct the concatentation which is the reverse of the vector order
      std::reverse(row.begin(), row.end());
      auto partialProductRow = rewriter.create<comb::ConcatOp>(loc,row);
      partialProducts.push_back(partialProductRow);
    }
    assert(partialProducts.size() == op.getNumResults() &&
           "Expected width number of partial products");

    rewriter.replaceOp(op, partialProducts);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Datapath pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertDatapathToCombPass
    : public impl::ConvertDatapathToCombBase<ConvertDatapathToCombPass> {
  void runOnOperation() override;
  using ConvertDatapathToCombBase<ConvertDatapathToCombPass>::ConvertDatapathToCombBase;
};
} // namespace

static void
populateDatapathToCombConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
      // Arithmetic Ops
      DatapathPartialProductOpConversion, DatapathCompressOpConversion
      >
      (patterns.getContext());
}

void ConvertDatapathToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
  target.addIllegalDialect<datapath::DatapathDialect>();

  RewritePatternSet patterns(&getContext());
  populateDatapathToCombConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
