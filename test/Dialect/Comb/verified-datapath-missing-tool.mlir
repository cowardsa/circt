// RUN: circt-opt --verify-diagnostics --comb-verified-datapath %s

// Without the lean-exe option the pass must fail when it encounters a
// multiplier.

// expected-error @below {{comb-verified-datapath requires the 'lean-exe' option}}
module {
  hw.module @missing_tool(in %a : i3, in %b : i3, out out : i3) {
    %0 = comb.mul %a, %b : i3
    hw.output %0 : i3
  }
}
