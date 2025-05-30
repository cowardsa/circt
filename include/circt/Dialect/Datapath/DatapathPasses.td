//===-- DatapathPasses.td - Datapath pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_PASSES_TD
#define CIRCT_DIALECT_DATAPATH_PASSES_TD

include "mlir/Pass/PassBase.td"

def DatapathReplicateAdds : Pass<"datapath-replicate-adds"> {
  let summary = "Replicate add operations to fold into compressor trees.";
  let description = [{
    This pass replicates add operations to allow them to be folded into
    compressor trees. This is useful for optimizing the datapath logic by
    reducing the number of adders needed.
  }];
  let constructor = "circt::datapath::createDatapathReplicateAddsPass()";
  let dependentDialects = ["datapath::DatapathDialect"];
}

#endif // CIRCT_DIALECT_DATAPATH_PASSES_TD
