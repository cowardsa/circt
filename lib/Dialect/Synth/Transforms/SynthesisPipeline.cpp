//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the default synthesis pipeline from core dialect to AIG.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/SynthesisPipeline.h"
#include "circt/Conversion/CombToDatapath.h"
#include "circt/Conversion/CombToSynth.h"
#include "circt/Conversion/DatapathToComb.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/Datapath/DatapathPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Support/SATSolver.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <memory>

using namespace mlir;
using namespace circt;
using namespace circt::synth;

//===----------------------------------------------------------------------===//
// Pipeline Implementation
//===----------------------------------------------------------------------===//

/// Helper function to populate additional legal ops for partial legalization.
template <typename... AllowedOpTy>
static void addOpName(SmallVectorImpl<std::string> &ops) {
  (ops.push_back(AllowedOpTy::getOperationName().str()), ...);
}
template <typename... OpToLowerTy>
static std::unique_ptr<Pass>
createLowerVariadicPass(bool timingAware, bool reuseSubsets = false) {
  LowerVariadicOptions options;
  addOpName<OpToLowerTy...>(options.opNames);
  options.timingAware = timingAware;
  options.reuseSubsets = reuseSubsets;
  return createLowerVariadic(options);
}
namespace {
/// Writes the whole module to a file and changes nothing.
///
/// Scheduled directly after the verified lowering so the file captures the
/// exact circuit the Lean proof covers. Everything downstream of this point
/// (CSE, canonicalisation, comb->AIG, mapping) is unverified, so this snapshot
/// is the reference an equivalence check compares the final output against.
/// Taking it inside the pipeline -- rather than replaying the early passes in
/// a separate process -- means the two sides provably come from one run.
struct SnapshotIRPass
    : public mlir::PassWrapper<SnapshotIRPass,
                               mlir::OperationPass<hw::HWModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SnapshotIRPass)

  SnapshotIRPass(StringRef filename)
      : filename(filename.str()),
        writtenOnce(std::make_shared<std::atomic_flag>()) {}

  StringRef getArgument() const override { return "synth-snapshot-ir"; }

  void runOnOperation() override {
    // The comb lowering pipeline runs nested under each hw.module, so write
    // one file per module. With a single top -- the usual case -- this is just
    // the requested path; otherwise the module name disambiguates.
    auto module = getOperation();
    SmallString<128> path(filename);
    if (!writtenOnce->test_and_set()) {
      // First module keeps the exact requested filename.
    } else {
      llvm::sys::path::replace_extension(
          path, "." + module.getModuleName().str() + ".mlir");
    }

    std::error_code ec;
    llvm::ToolOutputFile out(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError() << "cannot open snapshot file '" << path
                         << "': " << ec.message();
      return signalPassFailure();
    }
    // Print the module inside a top-level container so the result parses
    // standalone, which is what an equivalence checker needs.
    out.os() << "module {\n";
    module.print(out.os());
    out.os() << "\n}\n";
    out.keep();
  }

  std::string filename;
  // Shared so pass clones agree on which module claimed the exact filename.
  std::shared_ptr<std::atomic_flag> writtenOnce;
};
} // namespace

void circt::synth::buildCombLoweringPipeline(
    OpPassManager &pm, const CombLoweringPipelineOptions &options) {
  {
    // The verified lowering replaces the Datapath dialect flow, so it takes
    // priority over it when requested.
    if (options.verifiedDatapath) {
      // Lower variadic Mul into a binary op since the verified lowering only
      // handles two-input multiplies.
      pm.addPass(createLowerVariadicPass<comb::MulOp>(options.timingAware));
      comb::VerifiedDatapathOptions verifiedOptions;
      verifiedOptions.leanExe = options.verifiedDatapathLeanExe;
      pm.addPass(comb::createVerifiedDatapath(verifiedOptions));
      // Capture the proof boundary before any unverified pass runs.
      if (!options.verifiedDatapathSnapshot.empty())
        pm.addPass(
            std::make_unique<SnapshotIRPass>(options.verifiedDatapathSnapshot));
      pm.addPass(createSimpleCanonicalizerPass());
    } else if (!options.disableDatapath) {
      // Lower variadic Mul into a binary op to enable datapath lowering.
      pm.addPass(createLowerVariadicPass<comb::MulOp>(options.timingAware));
      pm.addPass(createConvertCombToDatapath());
      pm.addPass(createSimpleCanonicalizerPass());
      if (options.synthesisStrategy == OptimizationStrategyTiming)
        pm.addPass(datapath::createDatapathReduceDelay());
      circt::ConvertDatapathToCombOptions datapathOptions;
      datapathOptions.timingAware = options.timingAware;
      pm.addPass(createConvertDatapathToComb(datapathOptions));
    }
    pm.addPass(createCSEPass());
    pm.addPass(createSimpleCanonicalizerPass());
    // Partially legalize Comb, then run CSE and canonicalization.
    circt::ConvertCombToSynthOptions convOptions;
    addOpName<comb::AndOp, comb::OrOp, comb::XorOp, comb::MuxOp, comb::ICmpOp,
              hw::ArrayGetOp, hw::ArraySliceOp, hw::ArrayCreateOp,
              hw::ArrayConcatOp, hw::AggregateConstantOp>(
        convOptions.additionalLegalOps);
    pm.addPass(circt::createConvertCombToSynth(convOptions));
  }
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());
  // Balance mux chains. For area oriented flow, we want to keep the mux chains
  // unless they are very deep.
  comb::BalanceMuxOptions balanceOptions{OptimizationStrategyTiming ? 16 : 64};
  pm.addPass(comb::createBalanceMux(balanceOptions));

  // Lower variadic ops before running full lowering to target IR.
  // For AIG, lower variadic XoR since AIG cannot keep variadic
  // representation.
  pm.addPass(createLowerVariadicPass<comb::XorOp>(
      options.timingAware,
      options.synthesisStrategy == OptimizationStrategyArea));

  pm.addPass(circt::hw::createHWAggregateToComb());
  pm.addPass(circt::createConvertCombToSynth());
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void circt::synth::buildSynthOptimizationPipeline(
    OpPassManager &pm, const SynthOptimizationPipelineOptions &options) {
  // LowerWordToBits may not be scalable for large designs so conditionally
  // disable it. It's also worth considering keeping word-level representation
  // for faster synthesis.
  if (!options.disableWordToBits)
    pm.addPass(synth::createLowerWordToBits());
  pm.addPass(createCSEPass());
  // Run after LowerWordToBits for more precise timing-info & scalability.
  pm.addPass(createLowerVariadicPass(options.timingAware));
  pm.addPass(createStructuralHash());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.addPass(synth::createMaximumAndCover());
  pm.addPass(createLowerVariadicPass(options.timingAware));
  pm.addPass(createStructuralHash());

  // SOP balancing.
  if (!options.disableSOPBalancing) {
    SOPBalancingOptions sopOptions;
    // FIXME: The following is very small compared to the default value of ABC
    // (6/8) and mockturtle(4/25) due to inefficient implementation of
    // CutRewriter.
    sopOptions.maxCutInputSize = 4;
    sopOptions.maxCutsPerRoot = 4;
    pm.addPass(synth::createSOPBalancing(sopOptions));
    pm.addPass(createStructuralHash());
  }

  if (!options.disableFunctionalReduction && hasIncrementalSATSolverBackend()) {
    FunctionalReductionOptions functionalReductionOptions;
    functionalReductionOptions.conflictLimit =
        options.functionalReductionConflictLimit;
    pm.addPass(createFunctionalReduction(functionalReductionOptions));
  }

  if (!options.abcCommands.empty()) {
    synth::ABCRunnerOptions abcOptions;
    abcOptions.abcPath = options.abcPath;
    abcOptions.abcCommands.assign(options.abcCommands.begin(),
                                  options.abcCommands.end());
    abcOptions.continueOnFailure = options.ignoreAbcFailures;
    pm.addPass(synth::createABCRunner(abcOptions));
  }
  // TODO: Add more balancing and rewriting passes.
}

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

void circt::synth::registerSynthesisPipeline() {
  PassPipelineRegistration<CombLoweringPipelineOptions>(
      "synth-comb-lowering-pipeline",
      "The default pipeline for until Comb lowering",
      buildCombLoweringPipeline);
  PassPipelineRegistration<SynthOptimizationPipelineOptions>(
      "synth-optimization-pipeline",
      "The default pipeline for AIG optimization pipeline",
      buildSynthOptimizationPipeline);
}
