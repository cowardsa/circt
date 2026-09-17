// The two snapshots bracket the datapath lowering, so an equivalence checker
// can split the pipeline into three separately checkable segments instead of
// one end-to-end obligation.
// RUN: rm -rf %t && mkdir %t
// RUN: circt-synth %s --datapath-frontend-snapshot=%t/frontend.mlir \
// RUN:   --datapath-snapshot=%t/datapath.mlir -o /dev/null
// RUN: FileCheck --check-prefix=FRONTEND %s < %t/frontend.mlir
// RUN: FileCheck --check-prefix=DATAPATH %s < %t/datapath.mlir

// The front-end snapshot is taken after the n-ary mul split but before the
// lowering, so the multiply is still a comb.mul.
// FRONTEND: hw.module @mul
// FRONTEND: comb.mul

// The datapath snapshot is taken after the lowering, so the multiply is gone
// and no datapath op survives to the downstream passes.
// DATAPATH: hw.module @mul
// DATAPATH-NOT: comb.mul
// DATAPATH-NOT: datapath.

hw.module @mul(in %a: i4, in %b: i4, out mul: i4) {
  %0 = comb.mul %a, %b : i4
  hw.output %0 : i4
}
