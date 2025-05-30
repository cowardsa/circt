// RUN: circt-opt --pass-pipeline='builtin.module(kanagawa.design(kanagawa.class(kanagawa.method(kanagawa-reblock))))' %s | FileCheck %s

// CHECK-LABEL:   kanagawa.class sym @Reblock {
// CHECK:           kanagawa.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:             %[[VAL_3:.*]] = kanagawa.sblock () -> i32 {
// CHECK:               %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:               kanagawa.sblock.return %[[VAL_4]] : i32
// CHECK:             }
// CHECK:             %[[VAL_5:.*]] = kanagawa.sblock () -> i32 attributes {maxThreads = 2 : i64} {
// CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_3]], %[[VAL_2]] : i32
// CHECK:               %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_2]] : i32
// CHECK:               kanagawa.sblock.return %[[VAL_7]] : i32
// CHECK:             }
// CHECK:             %[[VAL_8:.*]] = kanagawa.sblock () -> i32 {
// CHECK:               %[[VAL_9:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:               kanagawa.sblock.return %[[VAL_9]] : i32
// CHECK:             }
// CHECK:             cf.br ^bb1(%[[VAL_8]] : i32)
// CHECK:           ^bb1(%[[VAL_10:.*]]: i32):
// CHECK:             %[[VAL_11:.*]] = kanagawa.sblock () -> i32 {
// CHECK:               %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_8]] : i32
// CHECK:               kanagawa.sblock.return %[[VAL_12]] : i32
// CHECK:             }
// CHECK:             kanagawa.return %[[VAL_11]] : i32
// CHECK:           }
// CHECK:         }

kanagawa.design @foo {
kanagawa.class sym @Reblock {

  kanagawa.method @foo(%arg0 : i32, %arg1 : i32) -> i32 {
      %0 = arith.addi %arg0, %arg1 : i32
      kanagawa.sblock.inline.begin {maxThreads = 2}
      %inner = arith.addi %0, %arg1 : i32
      %1 = arith.addi %inner, %arg1 : i32
      kanagawa.sblock.inline.end
      %2 = arith.addi %1, %arg1 : i32
      // %0 is used within the same MLIR block but outside the scope.
      cf.br ^bb1(%2 : i32)
    ^bb1(%barg : i32):
      %3 = arith.addi %barg, %2 : i32
      // %1 is used in a different MLIR block (dominated by the sblock parent block).
      kanagawa.return %3 : i32
  }
}
}
