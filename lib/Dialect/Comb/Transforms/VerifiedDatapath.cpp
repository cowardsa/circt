//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VerifiedDatapath pass, which lowers 2-input
// comb.mul and 3+-input comb.add operations through externally generated,
// formally verified compressor trees.
//
// The pass invokes the Lean `datapath-cli` tool (from the
// datapath-verification project) as `datapath-cli mul <width> <specs...>` or
// `datapath-cli add <width> <numOperands> <specs...>`. An operand spec is
// `<live>` or `<live>s`: the operand's low `<live>` bits are its real bits
// and the bits above them are either constant 0 (`<live>`, zero extension)
// or copies of bit `<live>-1` (`<live>s`, sign extension). The tool builds
// the bit heap (partial products for mul, stacked operand bits for add) from
// that operand model, compresses it with a Dadda tree, and replays the
// resulting full/half adder chain through the formally verified
// `applyChainSafe` checker before printing a gate netlist:
//
//   ok mul <width> <specs...>   (or: ok add <width> <numOperands> <specs...>)
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
// replaces the original op with `comb.add(row0, row1)` — the final
// carry-propagate adder, left for downstream lowering.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
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

/// How an operand's `live` low bits extend to the full result width: the bits
/// at or above `live` are all constant 0 (zero extension) or all copies of bit
/// `live - 1` (sign extension). The Lean flow models the operand accordingly
/// and never puts the extension bits in the heap as independent bits.
struct OperandSpec {
  unsigned live;
  bool isSigned;

  /// The protocol token for this operand: `<live>` or `<live>s`.
  std::string token() const {
    return std::to_string(live) + (isSigned ? "s" : "");
  }
};

/// Determine the extension shape of `operand`, an operand of a `width`-bit op.
///
/// Zero extension shows up in the known-bits lattice as leading known-zero
/// bits. Sign extension has no known bits at all, but comb spells it
/// structurally as
///
///   %sign = comb.extract %x from <n-1> : (i<n>) -> i1
///   %ext  = comb.replicate %sign : (i1) -> i<width-n>
///   %res  = comb.concat %ext, %x : i<width-n>, i<n>
///
/// (with the `comb.replicate` absent when only a single bit is added), which
/// `comb::m_Sext` matches.
static OperandSpec computeOperandSpec(Value operand, unsigned width) {
  KnownBits known = comb::computeKnownBits(operand);
  OperandSpec spec{width - known.Zero.countLeadingOnes(), /*isSigned=*/false};

  Value base;
  if (mlir::matchPattern(operand, comb::m_Sext(mlir::matchers::m_Any(&base)))) {
    unsigned signedLive = base.getType().getIntOrFloatBitWidth();
    // A zero-extended operand contributes nothing above its live width, while
    // a sign-extended one still replicates its sign bit into every column, so
    // zero extension wins whenever it is at least as tight.
    if (signedLive > 0 && signedLive < spec.live)
      spec = {signedLive, /*isSigned=*/true};
  }
  return spec;
}

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
    // above an operand's live width are a constant 0 or a copy of the sign
    // bit, and a valid netlist references neither directly.
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

/// Run `<leanExe> mul <width> <specA> <specB>` or
/// `<leanExe> add <width> <numOperands> <spec0> ...` and parse its stdout.
/// Emits diagnostics on `op` when anything goes wrong.
static FailureOr<Netlist> runNetlistTool(StringRef leanExe, StringRef kind,
                                         unsigned width,
                                         ArrayRef<OperandSpec> specs,
                                         Operation *op) {
  SmallString<128> outPath;
  if (llvm::sys::fs::createTemporaryFile("datapath-netlist", "txt", outPath))
    return op->emitError("failed to create temporary file for netlist output"),
           failure();
  llvm::FileRemover outRemover(outPath);

  // The tool echoes the arguments (sans the operand count for "mul") back in
  // its header line.
  SmallVector<std::string> argStorage = {std::to_string(width)};
  if (kind == "add")
    argStorage.push_back(std::to_string(specs.size()));
  for (const OperandSpec &spec : specs)
    argStorage.push_back(spec.token());

  SmallVector<StringRef> args = {leanExe, kind};
  std::string header = ("ok " + kind).str();
  for (const std::string &arg : argStorage) {
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
               << leanExe << " "
               << llvm::join(args.begin() + 1, args.end(), " ") << "' failed"
               << (errMsg.empty() ? "" : ": ") << errMsg,
           failure();

  auto buffer = llvm::MemoryBuffer::getFile(outPath);
  if (!buffer)
    return op->emitError("failed to read netlist output file"), failure();

  auto netlist = parseNetlist(buffer.get()->getBuffer(), header, width, specs);
  if (failed(netlist))
    return op->emitError("malformed netlist from verified datapath tool for '")
               << kind << " " << width << "'",
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
  // Collect candidates first; the rewrite inserts many ops. 2-input
  // multipliers and 3+-input adders are compressed; 2-input adders are
  // already final carry-propagate adders and stay untouched.
  SmallVector<Operation *> candidates;
  getOperation()->walk([&](Operation *op) {
    if (auto mulOp = dyn_cast<MulOp>(op)) {
      if (mulOp.getNumOperands() == 2 &&
          mulOp.getType().getIntOrFloatBitWidth() > 0)
        candidates.push_back(op);
      return;
    }
    if (auto addOp = dyn_cast<AddOp>(op)) {
      if (addOp.getNumOperands() >= 3 &&
          addOp.getType().getIntOrFloatBitWidth() > 0)
        candidates.push_back(op);
    }
  });
  if (candidates.empty())
    return;

  if (leanExe.empty()) {
    getOperation()->emitError(
        "comb-verified-datapath requires the 'lean-exe' option to point at "
        "the datapath-cli executable");
    return signalPassFailure();
  }

  // Netlists only depend on the operation kind, width, and per-operand live
  // widths, so run the tool once per distinct shape.
  llvm::StringMap<Netlist> netlistCache;
  for (Operation *op : candidates) {
    bool isMul = isa<MulOp>(op);
    StringRef kind = isMul ? "mul" : "add";
    unsigned width = op->getResult(0).getType().getIntOrFloatBitWidth();

    // An operand's live width excludes its extension bits — the leading
    // known-zero bits of a zero-extended value `concat(c0, x)`, or the
    // replicated sign bit of a sign-extended one `concat(replicate(x[n-1]),
    // x)`. The Lean flow then keeps those bits out of the bit heap as
    // independent bits entirely, shrinking the compressor.
    SmallVector<OperandSpec> specs;
    for (Value operand : op->getOperands())
      specs.push_back(computeOperandSpec(operand, width));

    std::string key = (kind + Twine(" ") + Twine(width)).str();
    for (const OperandSpec &spec : specs)
      key += " " + spec.token();
    auto cached = netlistCache.find(key);
    if (cached == netlistCache.end()) {
      auto netlist = runNetlistTool(leanExe, kind, width, specs, op);
      if (failed(netlist))
        return signalPassFailure();
      cached = netlistCache.try_emplace(key, std::move(*netlist)).first;
    }

    auto [row0, row1] = buildNetlist(cached->second, op, op->getOperands());
    OpBuilder builder(op);
    Value sum = AddOp::create(builder, op->getLoc(), ValueRange{row0, row1},
                              /*twoState=*/true);
    op->getResult(0).replaceAllUsesWith(sum);
    op->erase();
  }
}
