// UNSUPPORTED: system-windows
// RUN: circt-opt --verify-diagnostics --comb-verified-datapath="lean-exe=%S/does-not-exist" %s

// A tool that cannot be executed must fail the pass loudly rather than
// silently keeping the multiplier.

hw.module @tool_failure(in %a : i4, in %b : i4, out out : i4) {
  // expected-error @below {{verified datapath tool}}
  %0 = comb.mul %a, %b : i4
  hw.output %0 : i4
}
