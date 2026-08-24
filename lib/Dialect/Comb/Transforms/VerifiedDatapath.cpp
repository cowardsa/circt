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
// datapath-verification project) as `datapath-cli mul <width>` or
// `datapath-cli add <width> <numOperands>`. The tool builds the bit heap
// (partial products for mul, stacked operand bits for add), compresses it
// with a Dadda tree, and replays the resulting full/half adder chain through
// the formally verified `applyChainSafe` checker before printing a gate
// netlist:
//
//   ok mul <width>          (or: ok add <width> <numOperands>)
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
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
                                      ArrayRef<unsigned> liveWidths,
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
    // above an operand's live width are known zero and a valid netlist never
    // references them.
    if (token.getAsInteger(10, index) || index / width >= liveWidths.size() ||
        index % width >= liveWidths[index / width])
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
/// (e.g. "ok mul 8 8 8" or "ok add 8 3 8 4 2"). `liveWidths` gives each
/// operand's live width (bits above it are known zero).
static FailureOr<Netlist> parseNetlist(StringRef output, StringRef header,
                                       unsigned width,
                                       ArrayRef<unsigned> liveWidths) {
  Netlist netlist;
  netlist.width = width;
  netlist.numOperands = liveWidths.size();

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
      auto lhs = parseRef(tokens[3], width, liveWidths, netlist.gates.size());
      auto rhs = parseRef(tokens[4], width, liveWidths, netlist.gates.size());
      if (failed(lhs) || failed(rhs) ||
          lhs->kind == NetlistRef::Kind::Zero ||
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
        auto ref = parseRef(token, width, liveWidths, netlist.gates.size());
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

/// Run `<leanExe> mul <width> <liveA> <liveB>` or
/// `<leanExe> add <width> <numOperands> <live0> ...` and parse its stdout.
/// Emits diagnostics on `op` when anything goes wrong.
static FailureOr<Netlist> runNetlistTool(StringRef leanExe, StringRef kind,
                                         unsigned width,
                                         ArrayRef<unsigned> liveWidths,
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
    argStorage.push_back(std::to_string(liveWidths.size()));
  for (unsigned live : liveWidths)
    argStorage.push_back(std::to_string(live));

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
  int result =
      llvm::sys::ExecuteAndWait(leanExe, args,
                                /*Env=*/std::nullopt, redirects,
                                /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);
  if (result != 0)
    return op->emitError("verified datapath tool '")
               << leanExe << " " << llvm::join(args.begin() + 1, args.end(), " ")
               << "' failed" << (errMsg.empty() ? "" : ": ") << errMsg,
           failure();

  auto buffer = llvm::MemoryBuffer::getFile(outPath);
  if (!buffer)
    return op->emitError("failed to read netlist output file"), failure();

  auto netlist =
      parseNetlist(buffer.get()->getBuffer(), header, width, liveWidths);
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
static std::pair<Value, Value> buildNetlist(const Netlist &netlist,
                                            Operation *op,
                                            ValueRange operands) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  unsigned width = netlist.width;

  // Lazily created constants and input bit extracts.
  Value constants[2];
  auto getConstant = [&](bool value) -> Value {
    if (!constants[value])
      constants[value] =
          hw::ConstantOp::create(builder, loc, APInt(1, value));
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

    // An operand's live width excludes its leading known-zero bits — e.g. a
    // zero-extended value `concat(c0, x)`. The Lean flow then keeps those
    // constant-0 bits out of the bit heap entirely, shrinking the compressor.
    SmallVector<unsigned> liveWidths;
    for (Value operand : op->getOperands()) {
      KnownBits known = comb::computeKnownBits(operand);
      liveWidths.push_back(width - known.Zero.countLeadingOnes());
    }

    std::string key = (kind + Twine(" ") + Twine(width)).str();
    for (unsigned live : liveWidths)
      key += " " + std::to_string(live);
    auto cached = netlistCache.find(key);
    if (cached == netlistCache.end()) {
      auto netlist = runNetlistTool(leanExe, kind, width, liveWidths, op);
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
