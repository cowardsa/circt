// RUN: circt-opt --comb-verified-datapath="lean-exe=/home/oy229/datapath-verification/.lake/build/bin/datapath-cli" %s | FileCheck %s
// RUN: circt-opt --comb-verified-datapath="lean-exe=/home/oy229/datapath-verification/.lake/build/bin/datapath-cli max-heap-bits=8" %s | FileCheck %s --check-prefix=BUDGET


// `datapath-cli mul 3 3 3` performs the Dadda compression of the 3-bit
// partial-product heap, whose column 2 needs a single half adder (sum g5 =
// xor, carry dropped by truncation).

// CHECK-LABEL: hw.module @mul3
hw.module @mul3(in %a : i3, in %b : i3, out out : i3) {
  // CHECK:      %[[A0:.+]] = comb.extract %a from 0 : (i3) -> i1
  // CHECK-NEXT: %[[B0:.+]] = comb.extract %b from 0 : (i3) -> i1
  // CHECK-NEXT: %[[G0:.+]] = comb.and bin %[[A0]], %[[B0]] : i1
  // CHECK-NEXT: %[[B1:.+]] = comb.extract %b from 1 : (i3) -> i1
  // CHECK-NEXT: %[[G1:.+]] = comb.and bin %[[A0]], %[[B1]] : i1
  // CHECK-NEXT: %[[A1:.+]] = comb.extract %a from 1 : (i3) -> i1
  // CHECK-NEXT: %[[G2:.+]] = comb.and bin %[[A1]], %[[B0]] : i1
  // CHECK-NEXT: %[[A2:.+]] = comb.extract %a from 2 : (i3) -> i1
  // CHECK-NEXT: %[[G3:.+]] = comb.and bin %[[A2]], %[[B0]] : i1
  // CHECK-NEXT: %[[B2:.+]] = comb.extract %b from 2 : (i3) -> i1
  // CHECK-NEXT: %[[G4:.+]] = comb.and bin %[[A0]], %[[B2]] : i1
  // CHECK-NEXT: %[[G5:.+]] = comb.xor bin %[[G3]], %[[G4]] : i1
  // CHECK-NEXT: %[[G6:.+]] = comb.and bin %[[A1]], %[[B1]] : i1
  // CHECK-NEXT: %[[ROW0:.+]] = comb.concat %[[G5]], %[[G1]], %[[G0]] : i1, i1, i1
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %[[ROW1:.+]] = comb.concat %[[G6]], %[[G2]], %false : i1, i1, i1
  // CHECK-NEXT: %[[SUM:.+]] = comb.add bin %[[ROW0]], %[[ROW1]] : i3
  // CHECK-NEXT: hw.output %[[SUM]] : i3
  %0 = comb.mul %a, %b : i3
  hw.output %0 : i3
}

// The netlist tool runs once per width; a second 3-bit multiplier reuses the
// cached netlist.
// CHECK-LABEL: hw.module @mul3_again
hw.module @mul3_again(in %x : i3, in %y : i3, out out : i3) {
  // CHECK: comb.add bin
  // CHECK-NOT: comb.mul
  %0 = comb.mul %x, %y : i3
  hw.output %0 : i3
}

// Multipliers with more than two operands are left untouched.
// CHECK-LABEL: hw.module @variadic_mul
hw.module @variadic_mul(in %a : i3, in %b : i3, in %c : i3, out out : i3) {
  // CHECK: comb.mul %a, %b, %c : i3
  %0 = comb.mul %a, %b, %c : i3
  hw.output %0 : i3
}

// A 3-operand addition compresses each column of stacked operand bits (bit k
// of operands a/b/c) down to two rows: one FA in columns 1 and 2 (carry as
// `ab | (a^b)c`), one HA in column 0 (its sum is g0; the b0 bit rides along in
// row1).
// CHECK-LABEL: hw.module @add3x3
hw.module @add3x3(in %a : i3, in %b : i3, in %c : i3, out out : i3) {
  // CHECK:      %[[B0:.+]] = comb.extract %b from 0 : (i3) -> i1
  // CHECK-NEXT: %[[C0:.+]] = comb.extract %c from 0 : (i3) -> i1
  // CHECK-NEXT: %[[G0:.+]] = comb.xor bin %[[B0]], %[[C0]] : i1
  // CHECK-NEXT: %[[A1:.+]] = comb.extract %a from 1 : (i3) -> i1
  // CHECK-NEXT: %[[B1:.+]] = comb.extract %b from 1 : (i3) -> i1
  // CHECK-NEXT: %[[G1:.+]] = comb.xor bin %[[A1]], %[[B1]] : i1
  // CHECK-NEXT: %[[C1:.+]] = comb.extract %c from 1 : (i3) -> i1
  // CHECK-NEXT: %[[G2:.+]] = comb.xor bin %[[G1]], %[[C1]] : i1
  // CHECK-NEXT: %[[G3:.+]] = comb.and bin %[[B0]], %[[C0]] : i1
  // CHECK-NEXT: %[[A2:.+]] = comb.extract %a from 2 : (i3) -> i1
  // CHECK-NEXT: %[[B2:.+]] = comb.extract %b from 2 : (i3) -> i1
  // CHECK-NEXT: %[[G4:.+]] = comb.xor bin %[[A2]], %[[B2]] : i1
  // CHECK-NEXT: %[[C2:.+]] = comb.extract %c from 2 : (i3) -> i1
  // CHECK-NEXT: %[[G5:.+]] = comb.xor bin %[[G4]], %[[C2]] : i1
  // CHECK-NEXT: %[[G6:.+]] = comb.and bin %[[A1]], %[[B1]] : i1
  // CHECK-NEXT: %[[G7:.+]] = comb.and bin %[[G1]], %[[C1]] : i1
  // CHECK-NEXT: %[[G8:.+]] = comb.or bin %[[G6]], %[[G7]] : i1
  // CHECK-NEXT: %[[ROW0:.+]] = comb.concat %[[G5]], %[[G2]], %[[G0]] : i1, i1, i1
  // CHECK-NEXT: %[[A0:.+]] = comb.extract %a from 0 : (i3) -> i1
  // CHECK-NEXT: %[[ROW1:.+]] = comb.concat %[[G8]], %[[G3]], %[[A0]] : i1, i1, i1
  // CHECK-NEXT: %[[SUM:.+]] = comb.add bin %[[ROW0]], %[[ROW1]] : i3
  // CHECK-NEXT: hw.output %[[SUM]] : i3
  %0 = comb.add %a, %b, %c : i3
  hw.output %0 : i3
}

// 2-operand additions are already final carry-propagate adders and are left
// untouched.
// CHECK-LABEL: hw.module @add2_untouched
hw.module @add2_untouched(in %a : i3, in %b : i3, out out : i3) {
  // CHECK: comb.add %a, %b : i3
  // CHECK-NOT: comb.extract
  %0 = comb.add %a, %b : i3
  hw.output %0 : i3
}

// Zero-extension awareness: known-bits analysis reports each operand's live
// width (5), so the tool is invoked as `mul 10 5 5` and the heap holds only
// the 5x5 = 25 real partial products (Dadda: 8 FAs + 4 HAs, 89 gates instead
// of the 231 a blind 10-bit multiply needs) — bits 5-9 of the extended
// operands are never extracted, and column 9 stays empty (constant false);
// product bit 9 comes solely from the final adder's carry.
// CHECK-LABEL: hw.module @mul_zext_full
hw.module @mul_zext_full(in %a : i5, in %b : i5, out out : i10) {
  // CHECK-NOT: comb.extract %{{.+}} from 5
  // CHECK-NOT: comb.extract %{{.+}} from 6
  // CHECK-NOT: comb.extract %{{.+}} from 7
  // CHECK-NOT: comb.extract %{{.+}} from 8
  // CHECK-NOT: comb.extract %{{.+}} from 9
  // CHECK:      %false = hw.constant false
  // CHECK-NEXT: %[[ROW0:.+]] = comb.concat %false, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  // CHECK-NEXT: %[[ROW1:.+]] = comb.concat %false, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %false : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  // CHECK-NEXT: %[[SUM:.+]] = comb.add bin %[[ROW0]], %[[ROW1]] : i10
  // CHECK-NEXT: hw.output %[[SUM]] : i10
  %c0_i5 = hw.constant 0 : i5
  %ax = comb.concat %c0_i5, %a : i5, i5
  %bx = comb.concat %c0_i5, %b : i5, i5
  %0 = comb.mul %ax, %bx : i10
  hw.output %0 : i10
}

// Sign-extension awareness: comb spells a sign extension as
// `concat(replicate(extract(x, n-1)), x)`, which carries no known bits, so
// the structural `m_Sext` match is what reports the live width (5) here. The
// tool is invoked as `mul 10 5s 5s`, and the heap holds the 5x5 real partial
// products plus the sign-bit copies the extension implies — 149 gates against
// the 189 of a blind 10-bit multiply, with bits 5-9 of the extended operands
// never extracted.
// CHECK-LABEL: hw.module @mul_sext_full
hw.module @mul_sext_full(in %a : i5, in %b : i5, out out : i10) {
  // CHECK-NOT: comb.extract %{{.+}} from 5 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 6 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 7 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 8 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 9 : (i10) -> i1
  // CHECK: %[[SUM:.+]] = comb.add bin %{{.+}}, %{{.+}} : i10
  // CHECK-NEXT: hw.output %[[SUM]] : i10
  %sa = comb.extract %a from 4 : (i5) -> i1
  %ea = comb.replicate %sa : (i1) -> i5
  %ax = comb.concat %ea, %a : i5, i5
  %sb = comb.extract %b from 4 : (i5) -> i1
  %eb = comb.replicate %sb : (i1) -> i5
  %bx = comb.concat %eb, %b : i5, i5
  %0 = comb.mul %ax, %bx : i10
  hw.output %0 : i10
}

// A single sign bit is added without a `comb.replicate`; `m_Sext` matches that
// shape too, so the third operand is compressed as `3s` (`add 4 3 4 4 3s`).
// CHECK-LABEL: hw.module @add_sext
hw.module @add_sext(in %a : i4, in %b : i4, in %c : i3, out out : i4) {
  // The extended operand's bit 3 is never extracted; its sign bit (bit 2) is
  // extracted once and feeds both of the columns above the live width.
  // CHECK:      %[[CX:.+]] = comb.concat %{{.+}}, %c : i1, i3
  // CHECK-NOT:  comb.extract %[[CX]] from 3
  // CHECK:      %[[SIGN:.+]] = comb.extract %[[CX]] from 2 : (i4) -> i1
  // CHECK-NOT:  comb.extract %[[CX]] from 3
  // CHECK:      %[[SUM:.+]] = comb.add bin %{{.+}}, %{{.+}} : i4
  // CHECK-NEXT: hw.output %[[SUM]] : i4
  %sc = comb.extract %c from 2 : (i3) -> i1
  %cx = comb.concat %sc, %c : i1, i3
  %0 = comb.add %a, %b, %cx : i4
  hw.output %0 : i4
}

// When both models apply, zero extension is the tighter one: the sign bit of
// the extended value is itself known zero here, so known-bits reports a live
// width of 5 and the operand is compressed as `5`, not as the `9s` the
// structural match would give (`mul 10 5 5`, 73 gates).
// CHECK-LABEL: hw.module @sext_of_zext_prefers_zext
hw.module @sext_of_zext_prefers_zext(in %a : i5, in %b : i5, out out : i10) {
  // CHECK-NOT: comb.extract %{{.+}} from 5 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 6 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 7 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 8 : (i10) -> i1
  // CHECK-NOT: comb.extract %{{.+}} from 9 : (i10) -> i1
  // CHECK: %[[SUM:.+]] = comb.add bin %{{.+}}, %{{.+}} : i10
  // CHECK-NEXT: hw.output %[[SUM]] : i10
  %c0_i4 = hw.constant 0 : i4
  %av = comb.concat %c0_i4, %a : i4, i5
  %sa = comb.extract %av from 8 : (i9) -> i1
  %ax = comb.concat %sa, %av : i1, i9
  %bv = comb.concat %c0_i4, %b : i4, i5
  %sb = comb.extract %bv from 8 : (i9) -> i1
  %bx = comb.concat %sb, %bv : i1, i9
  %0 = comb.mul %ax, %bx : i10
  hw.output %0 : i10
}

// A fused multiply-add is one expression, so `c` joins the multiply's partial
// products in a single bit heap and a single compressor tree: `expr 3 3 add2
// mul 0.3 1.3 2.3`. Lowering the multiply on its own would instead leave `+ c`
// behind as a second adder, which costs delay and leaves arithmetic behind at
// the proof boundary.
// CHECK-LABEL: hw.module @fma
hw.module @fma(in %a : i3, in %b : i3, in %c : i3, out out : i3) {
  // The addend's bits are extracted, so they are in the heap...
  // CHECK-DAG: comb.extract %c from 0 : (i3) -> i1
  // CHECK-DAG: comb.extract %c from 1 : (i3) -> i1
  // CHECK-DAG: comb.extract %c from 2 : (i3) -> i1
  // ...and exactly one adder is left, the final carry-propagate adder.
  // CHECK: %[[SUM:.+]] = comb.add bin %{{.+}}, %{{.+}} : i3
  // CHECK-NEXT: hw.output %[[SUM]] : i3
  %0 = comb.mul %a, %b : i3
  %1 = comb.add %0, %c : i3
  hw.output %1 : i3
}

// A dot product fuses both multiplies and the addend into one heap.
// CHECK-LABEL: hw.module @dot_product
hw.module @dot_product(in %a : i3, in %b : i3, in %c : i3, in %d : i3, out out : i3) {
  // CHECK: %[[SUM:.+]] = comb.add bin %{{.+}}, %{{.+}} : i3
  // CHECK-NEXT: hw.output %[[SUM]] : i3
  %0 = comb.mul %a, %b : i3
  %1 = comb.mul %c, %d : i3
  %2 = comb.add %0, %1 : i3
  hw.output %2 : i3
}

// A subtracted addend joins the heap too: `comb.sub` is rewritten as
// `add(lhs, ~rhs, 1)` first, so a signed fused multiply-add — which reaches
// the pass as a subtraction — fuses like the unsigned one, carry-in included.
// CHECK-LABEL: hw.module @sub_fma
hw.module @sub_fma(in %a : i3, in %b : i3, in %c : i3, out out : i3) {
  // CHECK-NOT: comb.sub
  // CHECK: %[[SUM:.+]] = comb.add bin %{{.+}}, %{{.+}} : i3
  // CHECK-NEXT: hw.output %[[SUM]] : i3
  %0 = comb.mul %a, %b : i3
  %1 = comb.sub %0, %c : i3
  hw.output %1 : i3
}

// A multiply with more than one use is a leaf rather than being duplicated
// into the heap of every expression that reads it, so it keeps its own
// compressor tree and the addition stays a plain 2-input adder.
// CHECK-LABEL: hw.module @shared_mul_not_absorbed
hw.module @shared_mul_not_absorbed(in %a : i3, in %b : i3, in %c : i3, out x : i3, out y : i3) {
  // CHECK: %[[PROD:.+]] = comb.add bin %{{.+}}, %{{.+}} : i3
  // CHECK-NEXT: %[[SUM:.+]] = comb.add %[[PROD]], %c : i3
  // CHECK-NEXT: hw.output %[[PROD]], %[[SUM]] : i3, i3
  %0 = comb.mul %a, %b : i3
  %1 = comb.add %0, %c : i3
  hw.output %0, %1 : i3, i3
}

// On its own, `add(a, b, 1)` is a carry-propagate adder with a carry-in, which
// is cheaper than a compressor tree feeding one, so it is left untouched — as
// the Datapath dialect's conversion also leaves it.
// CHECK-LABEL: hw.module @carry_in_add_untouched
hw.module @carry_in_add_untouched(in %a : i3, in %b : i3, out out : i3) {
  // CHECK: comb.add %a, %b, %c1_i3 : i3
  // CHECK-NOT: comb.extract
  // CHECK: hw.output
  %one = hw.constant 1 : i3
  %0 = comb.add %a, %b, %one : i3
  hw.output %0 : i3
}

// Each multiply an expression absorbs multiplies the heap out, so fusing stops
// once the estimated heap exceeds the budget and the root is lowered on its
// own instead. Here the fused heap would be 3*3 + 3 = 12 bits: over the budget,
// so the multiply keeps its own compressor tree and the addition is left as a
// 2-input adder — exactly what an unfused lowering gives, never worse.
// BUDGET-LABEL: hw.module @over_budget
hw.module @over_budget(in %a : i3, in %b : i3, in %c : i3, out out : i3) {
  // BUDGET-NOT: comb.extract %c
  // BUDGET: %[[PROD:.+]] = comb.add bin %{{.+}}, %{{.+}} : i3
  // BUDGET-NEXT: %[[SUM:.+]] = comb.add %[[PROD]], %c : i3
  // BUDGET-NEXT: hw.output %[[SUM]] : i3
  %0 = comb.mul %a, %b : i3
  %1 = comb.add %0, %c : i3
  hw.output %1 : i3
}
