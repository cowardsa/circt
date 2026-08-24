// RUN: circt-opt --comb-verified-datapath="lean-exe=/home/oy229/datapath-verification/.lake/build/bin/datapath-cli" %s | FileCheck %s


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
// of operands a/b/c) down to two rows: one FA in columns 1 and 2, one HA in
// column 0 (its sum is g0; the b0 bit rides along in row1).
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
  // CHECK-NEXT: %[[G7:.+]] = comb.and bin %[[A1]], %[[C1]] : i1
  // CHECK-NEXT: %[[G8:.+]] = comb.or bin %[[G6]], %[[G7]] : i1
  // CHECK-NEXT: %[[G9:.+]] = comb.and bin %[[B1]], %[[C1]] : i1
  // CHECK-NEXT: %[[G10:.+]] = comb.or bin %[[G8]], %[[G9]] : i1
  // CHECK-NEXT: %[[ROW0:.+]] = comb.concat %[[G5]], %[[G2]], %[[G0]] : i1, i1, i1
  // CHECK-NEXT: %[[A0:.+]] = comb.extract %a from 0 : (i3) -> i1
  // CHECK-NEXT: %[[ROW1:.+]] = comb.concat %[[G10]], %[[G3]], %[[A0]] : i1, i1, i1
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
