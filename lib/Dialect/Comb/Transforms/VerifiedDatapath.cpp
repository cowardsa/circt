//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VerifiedDatapath pass, which lowers arithmetic
// expressions built from `comb.mul` and `comb.add` through externally
// generated, formally verified compressor trees.
//
// The pass fuses a whole addition tree over multiplies and leaf values -- an
// FMA `a * b + c`, a dot product `a * b + c * d`, a 3-input addition -- into
// one expression, so that every partial product and every addend lands in a
// single bit heap and is compressed by a single tree. Compressing `a * b` on
// its own and leaving `+ c` behind as a separate adder, as an interface
// limited to one multiply or one addition forces, both costs delay and leaves
// more arithmetic for downstream equivalence checking.
//
// The pass invokes the Lean `datapath-cli` tool (from the
// datapath-verification project) as
// `datapath-cli expr <width> <numOperands> <token>...`, where the tokens spell
// the expression in prefix notation:
//
//   `mul`             a binary multiply, followed by its two operands
//   `add<n>`          an n-ary addition, followed by its n operands
//   `<index>.<spec>`  a leaf: operand `<index>` with the given spec
//
// An operand spec is `<live>`: the operand's low `<live>` bits are its real
// bits and the bits above them are constant 0. So `a * b + c` over three
// 8-bit-live operands in 16 bits is `expr 16 3 add2 mul 0.8 1.8 2.8`.
//
// The tool builds one bit heap for the whole expression (partial products for
// each multiply, stacked operand bits for each addend), compresses it with a
// Dadda tree, and replays the resulting full/half adder chain through the
// formally verified `applyChainSafe` checker before printing a gate netlist:
//
//   ok expr <width> <numOperands> <token>...
//   gate g0 and b0 b4
//   gate g1 xor g0 b2
//   ...
//   row0 b0 g0 g3 -
//   row1 - g1 g2 g4
//
// References are `b<i>` (input bit `i % width` of operand `i / width`),
// `g<i>` (gate outputs, defined before use), or `c0`/`c1` (constants). The
// two `row` lines list one reference per column (LSB first); `-` denotes a
// constant-0 position. This pass translates each gate 1:1 into a comb
// bit-level op, concatenates the two rows into two width-bit values, and
// replaces the expression's root with `comb.add(row0, row1)` -- the final
// carry-propagate adder, left for downstream lowering.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

using namespace circt;
using namespace comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_VERIFIEDDATAPATH
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {

//===----------------------------------------------------------------------===//
// Operand extension analysis
//===----------------------------------------------------------------------===//

/// How many of an operand's low bits are its real ("live") bits: the bits at
/// or above `live` are all constant 0. The Lean flow models the operand
/// accordingly and never puts those extension bits in the heap as independent
/// bits.
struct OperandSpec {
  unsigned live;

  /// The protocol token for this operand: `<live>`.
  std::string token() const { return std::to_string(live); }
};

/// Determine the live width of `operand`, an operand of a `width`-bit op.
/// Zero extension shows up in the known-bits lattice as leading known-zero
/// bits.
static OperandSpec computeOperandSpec(Value operand, unsigned width) {
  KnownBits known = comb::computeKnownBits(operand);
  return OperandSpec{width - known.Zero.countLeadingOnes()};
}

//===----------------------------------------------------------------------===//
// Expression fusion
//===----------------------------------------------------------------------===//

/// True if `op` is an operation the fused expression grammar has a node for.
static bool isArithNode(Operation *op) {
  if (auto mulOp = dyn_cast<MulOp>(op))
    return mulOp.getNumOperands() == 2;
  if (auto addOp = dyn_cast<AddOp>(op))
    return addOp.getNumOperands() >= 2;
  return false;
}

/// True if `v`'s definition can be pulled into the bit heap of the addition
/// that consumes it, rather than being summed separately first.
///
/// Only additions absorb: a multiply's operands always stay leaves, so that
/// the heap holds exactly one level of partial products, matching the Datapath
/// dialect's `datapath.partial_product` + `datapath.compress` pair. Absorbing
/// also requires a single use, since a shared subexpression would otherwise be
/// duplicated into the heap of every expression that reads it.
static bool canAbsorb(Value v) {
  Operation *def = v.getDefiningOp();
  return def && isArithNode(def) && v.hasOneUse();
}

/// Rewrite every `comb.sub(lhs, rhs)` under `root` as `comb.add(lhs, ~rhs, 1)`,
/// the same rewrite `comb::convertSubToAdd` applies for the Datapath dialect's
/// conversion. Without it a subtracted term can never join the bit heap of the
/// expression it belongs to: a signed fused multiply-add reaches this pass as
/// `sub(a * b, c)`, whose subtraction would otherwise be left behind as a
/// separate adder.
static void lowerSubToAdd(Operation *root) {
  SmallVector<SubOp> subs;
  root->walk([&](SubOp subOp) { subs.push_back(subOp); });
  for (SubOp subOp : subs) {
    OpBuilder builder(subOp);
    Location loc = subOp.getLoc();
    // -rhs == ~rhs + 1, so sub(lhs, rhs) == add(lhs, ~rhs, 1).
    Value notRhs =
        createOrFoldNot(builder, loc, subOp.getRhs(), subOp.getTwoState());
    Value one = hw::ConstantOp::create(builder, loc, subOp.getType(), 1);
    Value sum =
        AddOp::create(builder, loc, ValueRange{subOp.getLhs(), notRhs, one},
                      subOp.getTwoState());
    subOp.getResult().replaceAllUsesWith(sum);
    subOp.erase();
  }
}

/// True if `op` is `add(a, b, 1)`, the shape a `comb.sub` lowers to. On its own
/// that is a carry-propagate adder with a carry-in, which is cheaper than a
/// compressor tree feeding one; the Datapath dialect's conversion leaves it
/// alone for the same reason. It is still worth compressing once it has
/// absorbed a multiply or a further addend.
static bool isCarryInAdd(Operation *op) {
  auto addOp = dyn_cast<AddOp>(op);
  if (!addOp || addOp.getNumOperands() != 3)
    return false;
  auto constOp = addOp.getOperand(2).getDefiningOp<hw::ConstantOp>();
  return constOp && constOp.getValue().isOne();
}

/// A fused arithmetic expression: an addition tree over multiplies and leaf
/// values, in the shape the Lean `expr` protocol accepts. Node 0 is the root.
struct Expr {
  struct Node {
    enum class Kind { Leaf, Add, Mul } kind;
    /// Kind::Leaf: index into `leaves`/`specs`.
    unsigned leaf = 0;
    /// Kind::Add / Kind::Mul: child node indices.
    SmallVector<unsigned> kids;
  };

  unsigned width = 0;
  SmallVector<Node> nodes;
  /// The expression's input operands, in the order the protocol numbers them.
  SmallVector<Value> leaves;
  /// Parallel to `leaves`.
  SmallVector<OperandSpec> specs;
  /// Ops absorbed into this expression, in pre-order: each has a single use,
  /// by its parent, so erasing them front to back after the root is safe.
  SmallVector<Operation *> interior;
  unsigned numMuls = 0;

  /// The prefix-notation tokens for the whole expression.
  SmallVector<std::string> tokens() const {
    SmallVector<std::string> out;
    emit(0, out);
    return out;
  }

  /// An upper bound on the number of bits the expression puts in the heap.
  /// Addition stacks its operands' bits; multiplication is a convolution, so
  /// its operands' bit counts multiply.
  uint64_t estimateHeapBits() const { return estimate(0); }

private:
  void emit(unsigned n, SmallVectorImpl<std::string> &out) const {
    const Node &node = nodes[n];
    switch (node.kind) {
    case Node::Kind::Leaf:
      out.push_back(std::to_string(node.leaf) + "." + specs[node.leaf].token());
      return;
    case Node::Kind::Mul:
      out.push_back("mul");
      break;
    case Node::Kind::Add:
      out.push_back("add" + std::to_string(node.kids.size()));
      break;
    }
    for (unsigned kid : node.kids)
      emit(kid, out);
  }

  uint64_t estimate(unsigned n) const {
    const Node &node = nodes[n];
    switch (node.kind) {
    case Node::Kind::Leaf:
      return specs[node.leaf].live;
    case Node::Kind::Add: {
      uint64_t sum = 0;
      for (unsigned kid : node.kids)
        sum += estimate(kid);
      return sum;
    }
    case Node::Kind::Mul:
      return estimate(node.kids[0]) * estimate(node.kids[1]);
    }
    llvm_unreachable("covered switch");
  }
};

/// Builds an `Expr` by descending from a root operation through the additions
/// it can absorb.
class ExprBuilder {
public:
  ExprBuilder(unsigned width, bool allowAbsorb) : allowAbsorb(allowAbsorb) {
    expr.width = width;
  }

  Expr build(Operation *root) && {
    buildNode(root);
    return std::move(expr);
  }

private:
  unsigned buildNode(Operation *op) {
    // Reserve this node's slot first: the recursive calls below append to
    // `nodes`, so the reference would not survive them.
    unsigned self = expr.nodes.size();
    expr.nodes.push_back(Expr::Node{});

    if (isa<MulOp>(op)) {
      ++expr.numMuls;
      SmallVector<unsigned> kids = {buildLeaf(op->getOperand(0)),
                                    buildLeaf(op->getOperand(1))};
      expr.nodes[self].kind = Expr::Node::Kind::Mul;
      expr.nodes[self].kids = std::move(kids);
      return self;
    }

    SmallVector<unsigned> kids;
    for (Value operand : op->getOperands()) {
      if (allowAbsorb && canAbsorb(operand)) {
        // Pre-order, so that erasing the list front to back always removes a
        // parent before the child whose only use it is.
        expr.interior.push_back(operand.getDefiningOp());
        kids.push_back(buildNode(operand.getDefiningOp()));
        continue;
      }
      kids.push_back(buildLeaf(operand));
    }
    expr.nodes[self].kind = Expr::Node::Kind::Add;
    expr.nodes[self].kids = std::move(kids);
    return self;
  }

  unsigned buildLeaf(Value v) {
    unsigned self = expr.nodes.size();
    expr.nodes.push_back(
        Expr::Node{Expr::Node::Kind::Leaf, (unsigned)expr.leaves.size(), {}});
    expr.leaves.push_back(v);
    expr.specs.push_back(computeOperandSpec(v, expr.width));
    return self;
  }

  Expr expr;
  bool allowAbsorb;
};

//===----------------------------------------------------------------------===//
// Netlist representation and parsing
//===----------------------------------------------------------------------===//

/// A reference to a signal in the netlist.
struct NetlistRef {
  enum class Kind { InputBit, Gate, Const, Zero } kind;
  // InputBit: global input bit index; Gate: gate index; Const: 0 or 1.
  unsigned index = 0;
};

struct NetlistGate {
  enum class Kind { And, Or, Xor, Nand } kind;
  NetlistRef lhs, rhs;
};

/// A parsed compressor netlist for one operation shape.
struct Netlist {
  unsigned width;
  unsigned numOperands;
  SmallVector<NetlistGate> gates;
  // Exactly `width` entries per row, LSB first.
  SmallVector<NetlistRef> row0, row1;
};

static FailureOr<NetlistRef> parseRef(StringRef token, unsigned width,
                                      ArrayRef<OperandSpec> specs,
                                      unsigned numGatesSoFar) {
  if (token == "-")
    return NetlistRef{NetlistRef::Kind::Zero, 0};
  if (token == "c0")
    return NetlistRef{NetlistRef::Kind::Const, 0};
  if (token == "c1")
    return NetlistRef{NetlistRef::Kind::Const, 1};
  unsigned index;
  if (token.consume_front("b")) {
    // Input bits must address a live bit of an existing operand: bits at or
    // above an operand's live width are constant 0, and a valid netlist never
    // references them.
    if (token.getAsInteger(10, index) || index / width >= specs.size() ||
        index % width >= specs[index / width].live)
      return failure();
    return NetlistRef{NetlistRef::Kind::InputBit, index};
  }
  if (token.consume_front("g")) {
    // Gates may only reference previously defined gates.
    if (token.getAsInteger(10, index) || index >= numGatesSoFar)
      return failure();
    return NetlistRef{NetlistRef::Kind::Gate, index};
  }
  return failure();
}

/// Parse the netlist protocol output; `header` is the expected first line
/// (e.g. "ok mul 8 8 8" or "ok add 8 3 8 4s 2"). `specs` gives each operand's
/// live width and extension kind.
static FailureOr<Netlist> parseNetlist(StringRef output, StringRef header,
                                       unsigned width,
                                       ArrayRef<OperandSpec> specs) {
  Netlist netlist;
  netlist.width = width;
  netlist.numOperands = specs.size();

  SmallVector<StringRef> lines;
  output.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (lines.empty() || lines[0].trim() != header)
    return failure();

  bool sawRow0 = false, sawRow1 = false;
  for (StringRef line : ArrayRef(lines).drop_front()) {
    SmallVector<StringRef> tokens;
    line.split(tokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (tokens.empty())
      continue;

    if (tokens[0] == "gate") {
      // gate g<N> <op> <ref> <ref>, with N the next gate index.
      if (sawRow0 || sawRow1 || tokens.size() != 5)
        return failure();
      if (tokens[1] != ("g" + Twine(netlist.gates.size())).str())
        return failure();
      NetlistGate gate;
      if (tokens[2] == "and")
        gate.kind = NetlistGate::Kind::And;
      else if (tokens[2] == "or")
        gate.kind = NetlistGate::Kind::Or;
      else if (tokens[2] == "xor")
        gate.kind = NetlistGate::Kind::Xor;
      else if (tokens[2] == "nand")
        gate.kind = NetlistGate::Kind::Nand;
      else
        return failure();
      auto lhs = parseRef(tokens[3], width, specs, netlist.gates.size());
      auto rhs = parseRef(tokens[4], width, specs, netlist.gates.size());
      if (failed(lhs) || failed(rhs) || lhs->kind == NetlistRef::Kind::Zero ||
          rhs->kind == NetlistRef::Kind::Zero)
        return failure();
      gate.lhs = *lhs;
      gate.rhs = *rhs;
      netlist.gates.push_back(gate);
      continue;
    }

    if (tokens[0] == "row0" || tokens[0] == "row1") {
      if (tokens.size() != width + 1)
        return failure();
      SmallVector<NetlistRef> &row =
          tokens[0] == "row0" ? netlist.row0 : netlist.row1;
      bool &seen = tokens[0] == "row0" ? sawRow0 : sawRow1;
      if (seen)
        return failure();
      seen = true;
      for (StringRef token : ArrayRef(tokens).drop_front()) {
        auto ref = parseRef(token, width, specs, netlist.gates.size());
        if (failed(ref))
          return failure();
        row.push_back(*ref);
      }
      continue;
    }

    return failure();
  }

  if (!sawRow0 || !sawRow1)
    return failure();
  return netlist;
}

//===----------------------------------------------------------------------===//
// External tool invocation
//===----------------------------------------------------------------------===//

/// Run `<leanExe> <toolArgs>...` and parse its stdout. `specs` describes the
/// expression's operands, in the order the protocol numbers them. Emits
/// diagnostics on `op` when anything goes wrong.
static FailureOr<Netlist> runNetlistTool(StringRef leanExe,
                                         ArrayRef<std::string> toolArgs,
                                         unsigned width,
                                         ArrayRef<OperandSpec> specs,
                                         Operation *op) {
  SmallString<128> outPath;
  if (llvm::sys::fs::createTemporaryFile("datapath-netlist", "txt", outPath))
    return op->emitError("failed to create temporary file for netlist output"),
           failure();
  llvm::FileRemover outRemover(outPath);

  // The tool echoes its arguments back in its header line.
  SmallVector<StringRef> args = {leanExe};
  std::string header = "ok";
  for (const std::string &arg : toolArgs) {
    args.push_back(arg);
    header += " " + arg;
  }

  std::optional<StringRef> redirects[3] = {/*stdin=*/StringRef(""),
                                           /*stdout=*/outPath.str(),
                                           /*stderr=*/std::nullopt};
  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(leanExe, args,
                                         /*Env=*/std::nullopt, redirects,
                                         /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                                         &errMsg);
  if (result != 0)
    return op->emitError("verified datapath tool '")
               << leanExe << " " << llvm::join(toolArgs, " ") << "' failed"
               << (errMsg.empty() ? "" : ": ") << errMsg,
           failure();

  auto buffer = llvm::MemoryBuffer::getFile(outPath);
  if (!buffer)
    return op->emitError("failed to read netlist output file"), failure();

  auto netlist = parseNetlist(buffer.get()->getBuffer(), header, width, specs);
  if (failed(netlist))
    return op->emitError(
               "malformed netlist from verified datapath tool for '")
               << llvm::join(toolArgs, " ") << "'",
           failure();
  return netlist;
}

//===----------------------------------------------------------------------===//
// Netlist materialization
//===----------------------------------------------------------------------===//

/// Materialize the netlist as comb ops and return the two packed rows.
static std::pair<Value, Value>
buildNetlist(const Netlist &netlist, Operation *op, ValueRange operands) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  unsigned width = netlist.width;

  // Lazily created constants and input bit extracts.
  Value constants[2];
  auto getConstant = [&](bool value) -> Value {
    if (!constants[value])
      constants[value] = hw::ConstantOp::create(builder, loc, APInt(1, value));
    return constants[value];
  };
  SmallVector<Value> inputBits(netlist.numOperands * width);
  auto getInputBit = [&](unsigned index) -> Value {
    if (!inputBits[index])
      inputBits[index] = builder.createOrFold<ExtractOp>(
          loc, operands[index / width], index % width, 1);
    return inputBits[index];
  };

  SmallVector<Value> gateValues;
  auto getRef = [&](NetlistRef ref) -> Value {
    switch (ref.kind) {
    case NetlistRef::Kind::Zero:
      return getConstant(false);
    case NetlistRef::Kind::Const:
      return getConstant(ref.index);
    case NetlistRef::Kind::InputBit:
      return getInputBit(ref.index);
    case NetlistRef::Kind::Gate:
      return gateValues[ref.index];
    }
    llvm_unreachable("covered switch");
  };

  for (const NetlistGate &gate : netlist.gates) {
    Value lhs = getRef(gate.lhs), rhs = getRef(gate.rhs);
    Value result;
    switch (gate.kind) {
    case NetlistGate::Kind::And:
      result = builder.createOrFold<AndOp>(loc, lhs, rhs, /*twoState=*/true);
      break;
    case NetlistGate::Kind::Or:
      result = builder.createOrFold<OrOp>(loc, lhs, rhs, /*twoState=*/true);
      break;
    case NetlistGate::Kind::Xor:
      result = builder.createOrFold<XorOp>(loc, lhs, rhs, /*twoState=*/true);
      break;
    case NetlistGate::Kind::Nand:
      result = builder.createOrFold<AndOp>(loc, lhs, rhs, /*twoState=*/true);
      result = builder.createOrFold<XorOp>(loc, result, getConstant(true),
                                           /*twoState=*/true);
      break;
    }
    gateValues.push_back(result);
  }

  // Pack a row into a width-bit value. Row entries are LSB first, but
  // comb.concat takes operands MSB first.
  auto packRow = [&](ArrayRef<NetlistRef> row) -> Value {
    SmallVector<Value> bits;
    for (NetlistRef ref : llvm::reverse(row))
      bits.push_back(getRef(ref));
    if (bits.size() == 1)
      return bits[0];
    return builder.createOrFold<ConcatOp>(loc, bits);
  };
  return {packRow(netlist.row0), packRow(netlist.row1)};
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

class VerifiedDatapathPass
    : public impl::VerifiedDatapathBase<VerifiedDatapathPass> {
public:
  using VerifiedDatapathBase::VerifiedDatapathBase;
  void runOnOperation() override;
};

} // namespace

void VerifiedDatapathPass::runOnOperation() {
  // An operation this pass can rewrite, or turn into one it can: `comb.sub`
  // becomes an addition below.
  auto isCandidate = [](Operation *op) {
    return (isArithNode(op) || isa<SubOp>(op)) &&
           op->getResult(0).getType().getIntOrFloatBitWidth() != 0;
  };

  // Leave a module with no arithmetic alone, tool or no tool.
  auto found = getOperation()->walk([&](Operation *op) {
    return isCandidate(op) ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (!found.wasInterrupted())
    return;

  if (leanExe.empty()) {
    getOperation()->emitError(
        "comb-verified-datapath requires the 'lean-exe' option to point at "
        "the datapath-cli executable");
    return signalPassFailure();
  }

  lowerSubToAdd(getOperation());

  // Collect the candidate operations before rewriting any of them; each
  // rewrite inserts many ops.
  SmallVector<Operation *> candidates;
  getOperation()->walk([&](Operation *op) {
    if (isCandidate(op))
      candidates.push_back(op);
  });

  // Operations absorbed into an expression that has already been rewritten.
  // Walk order lists an operation before its users, so visiting the candidates
  // backwards reaches an expression's root before the operations it might
  // absorb, and whether a candidate is absorbed is settled by the time it is
  // reached: absorbed ones are in here and erased, while any the root turned
  // out not to take — because the heap budget or its use count ruled it out —
  // are still standing, and root expressions of their own.
  DenseSet<Operation *> absorbed;

  // Netlists only depend on the expression's shape — the operation kinds, the
  // width, and each leaf's live width and extension kind — all of which the
  // protocol tokens spell out, so the token string is the cache key and the
  // tool runs once per distinct shape.
  llvm::StringMap<Netlist> netlistCache;
  for (Operation *op : llvm::reverse(candidates)) {
    if (absorbed.contains(op))
      continue;
    unsigned width = op->getResult(0).getType().getIntOrFloatBitWidth();

    // A leaf's live width excludes its extension bits — the leading
    // known-zero bits of a zero-extended value `concat(c0, x)`. The Lean flow
    // then keeps those bits out of the bit heap as independent bits entirely,
    // shrinking the compressor.
    Expr expr = ExprBuilder(width, /*allowAbsorb=*/true).build(op);
    // Fusing an expression is what lets an addend share the multiply's
    // compressor tree, but each multiply it pulls in multiplies the heap out,
    // so fall back to rewriting the root alone when the heap grows past the
    // budget. That is never worse than not fusing at all.
    if (maxHeapBits != 0 && expr.estimateHeapBits() > maxHeapBits)
      expr = ExprBuilder(width, /*allowAbsorb=*/false).build(op);

    // A 2-input addition is already a carry-propagate adder, and `add(a, b, 1)`
    // is one with a carry-in: there is nothing to compress unless the
    // expression pulled in a multiply or a further addend.
    if (expr.numMuls == 0 &&
        (expr.leaves.size() < 3 ||
         (expr.leaves.size() == 3 && isCarryInAdd(op))))
      continue;

    SmallVector<std::string> toolArgs = {"expr", std::to_string(width),
                                         std::to_string(expr.leaves.size())};
    llvm::append_range(toolArgs, expr.tokens());

    std::string key = llvm::join(toolArgs, " ");
    auto cached = netlistCache.find(key);
    if (cached == netlistCache.end()) {
      auto netlist = runNetlistTool(leanExe, toolArgs, width, expr.specs, op);
      if (failed(netlist))
        return signalPassFailure();
      cached = netlistCache.try_emplace(key, std::move(*netlist)).first;
    }

    auto [row0, row1] = buildNetlist(cached->second, op, expr.leaves);
    OpBuilder builder(op);
    Value sum = AddOp::create(builder, op->getLoc(), ValueRange{row0, row1},
                              /*twoState=*/true);
    op->getResult(0).replaceAllUsesWith(sum);
    op->erase();
    // `interior` is in pre-order and every entry has a single use, by its
    // parent, so each is dead by the time it is reached.
    for (Operation *interior : expr.interior) {
      absorbed.insert(interior);
      interior->erase();
    }
  }
}
