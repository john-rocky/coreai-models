// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// SpecModel — one DENSE decode bundle driven through owned MTLBuffers, with the
// verify shape (`[1, q, vocab]`, all q rows readable) exposed to the host.
//
// The shipped decode bundles run the head on EVERY position (no last-row slice),
// so feeding q tokens in one forward returns q per-position logit rows, each
// conditioned on the prefix up to and including that position — exactly what
// speculative verification needs. Nothing here re-exports the graph.
//
// Rollback is trivial on a dense (KV-only) model and that is why this file is
// short next to the GDN-hybrid precedent (SpecDecodeEngine.swift in the zoo):
//
//   * the graph writes KV rows at `offset = position_ids.count - input_ids.count`,
//     i.e. at `processed`, and attends `[0, processed + q)`;
//   * so rows past the accepted prefix are stale but NEVER read — the next forward
//     binds a shorter `position_ids` and overwrites them;
//   * therefore "reject" == "do not advance `processed`". No state snapshot, no
//     re-anchor forward, no window discipline.
//
// `forward()` deliberately does NOT advance `processed`; the caller commits the
// accepted count. Every forward is drained (`currentWorkCompleted`) because the
// loop is strictly sequential — the next round cannot draft until it has read
// this round's logits.

import CoreAI
import CoreAIShared
import Foundation
import Metal

struct SpecDecodeError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { self.description = message }
}

final class SpecModel {
    let name: String
    let vocab: Int
    let kvCapacity: Int
    let maxQuery: Int

    /// Committed KV length — the number of tokens the cache is authoritative for.
    private(set) var processed = 0
    private(set) var forwards = 0
    /// Rows written by the last forward (== that forward's query length).
    private(set) var lastRows = 0
    /// Wall time of the last forward (encode + GPU drain), milliseconds.
    private(set) var lastForwardMs = 0.0

    private let model: AIModel  // retains the mapped weights
    private let fn: InferenceFunction
    private let inName: String
    private let posName: String
    private let keyName: String
    private let valueName: String
    private let logitsName: String

    private let inDesc: NDArrayDescriptor
    private let posDesc: NDArrayDescriptor
    private let logitsDesc: NDArrayDescriptor

    private let device: MTLDevice
    // ComputeStream is a non-Sendable final class. The spec loop awaits each forward
    // to completion before starting the next, so it is never touched concurrently.
    nonisolated(unsafe) private let computeStream: ComputeStream

    private let idBuffer: MTLBuffer  // [1, maxQuery]    int32
    private let posBuffer: MTLBuffer  // [1, kvCapacity]  int32, pre-filled 0..<kvCapacity
    private let logitsBuffer: MTLBuffer  // [1, maxQuery, vocab] float16
    private let logitsElementCapacity: Int
    /// Element stride between logit rows for the LAST forward's resolved shape.
    private var lastRowStride = 0

    private struct StateBinding {
        let name: String
        let buffer: MTLBuffer
        let scalarType: NDArray.ScalarType
        let shape: [Int]
        let strides: [Int]
        let byteCount: Int
    }
    private let keyState: StateBinding
    private let valueState: StateBinding

    // MARK: - Init

    init(assetURL: URL, kvCapacity: Int, maxQuery: Int) async throws {
        self.name = assetURL.deletingPathExtension().lastPathComponent
        self.kvCapacity = kvCapacity
        self.maxQuery = maxQuery

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SpecDecodeError("no Metal device")
        }
        self.device = device

        // PreparedModel derives the specialization options from the probed structure:
        // a dynamic ("main") graph gets GPU + expectFrequentReshapes, which is what a
        // multi-query-length loop needs (one specialization per S, closed set 1...K+1).
        let prepared = try await PreparedModel.prepare(at: assetURL)
        self.model = prepared.model

        guard let descriptor = prepared.model.functionDescriptor(for: "main") else {
            throw SpecDecodeError("\(assetURL.lastPathComponent): no 'main' function")
        }
        guard descriptor.inputNames.count >= 2 else {
            throw SpecDecodeError("expected input_ids + position_ids, got \(descriptor.inputNames)")
        }
        guard descriptor.stateNames.count == 2 else {
            throw SpecDecodeError(
                "expected exactly 2 states (KV pair) on a dense bundle, got \(descriptor.stateNames) "
                    + "— a hybrid/SSM bundle needs the snapshot-and-replay rollback instead")
        }
        guard descriptor.outputNames.count >= 1 else {
            throw SpecDecodeError("expected a logits output")
        }
        inName = descriptor.inputNames[0]
        posName = descriptor.inputNames[1]
        keyName = descriptor.stateNames[0]
        valueName = descriptor.stateNames[1]
        logitsName = descriptor.outputNames[0]

        guard case .ndArray(let inputIdsDesc) = descriptor.inputDescriptor(of: inName) else {
            throw SpecDecodeError("no descriptor for '\(inName)'")
        }
        guard case .ndArray(let posIdsDesc) = descriptor.inputDescriptor(of: posName) else {
            throw SpecDecodeError("no descriptor for '\(posName)'")
        }
        guard case .ndArray(let logDesc) = descriptor.outputDescriptor(of: logitsName) else {
            throw SpecDecodeError("no descriptor for '\(logitsName)'")
        }
        guard logDesc.scalarType == .float16 else {
            throw SpecDecodeError("expected float16 logits, got \(logDesc.scalarType)")
        }
        // The verify shape must be dynamic in the query dimension: a `--static-ids`
        // bundle is pinned to S=1 and cannot verify a draft at all.
        guard inputIdsDesc.shape.count == 2, inputIdsDesc.shape[1] <= 0 else {
            throw SpecDecodeError(
                "input_ids is static \(inputIdsDesc.shape) — this bundle can only run S=1. "
                    + "Speculative verify needs the dynamic-ids export (no --static-ids).")
        }
        guard let v = logDesc.shape.last, v > 0 else {
            throw SpecDecodeError("logits descriptor has no vocab dimension: \(logDesc.shape)")
        }
        vocab = v
        inDesc = inputIdsDesc
        posDesc = posIdsDesc
        logitsDesc = logDesc

        guard case .ndArray(let keyDesc) = descriptor.stateDescriptor(of: keyName),
            case .ndArray(let valueDesc) = descriptor.stateDescriptor(of: valueName)
        else {
            throw SpecDecodeError("cannot read KV cache state descriptors")
        }

        guard let queue = device.makeCommandQueue() else {
            throw SpecDecodeError("no Metal command queue")
        }
        queue.label = "spec-decode.\(name)"
        computeStream = ComputeStream(commandQueue: queue)

        // Inputs / outputs, sized once for the largest query this run will feed.
        let idBytes = inputIdsDesc.resolvingDynamicDimensions([1, maxQuery]).minimumByteCount
        let posBytes = posIdsDesc.resolvingDynamicDimensions([1, kvCapacity]).minimumByteCount
        let logitsResolved = logDesc.resolvingDynamicDimensions([1, maxQuery, v])
        let logitsBytes = logitsResolved.minimumByteCount
        guard let idBuf = device.makeBuffer(length: idBytes, options: .storageModeShared),
            let posBuf = device.makeBuffer(length: posBytes, options: .storageModeShared),
            let logitsBuf = device.makeBuffer(length: logitsBytes, options: .storageModeShared)
        else {
            throw SpecDecodeError("input/logits buffer allocation failed")
        }
        idBuffer = idBuf
        posBuffer = posBuf
        logitsBuffer = logitsBuf
        logitsElementCapacity = logitsBytes / MemoryLayout<Float16>.size

        // position_ids is the constant ramp 0,1,2,… — written once, bound as a prefix.
        let posPtr = posBuf.contents().bindMemory(to: Int32.self, capacity: kvCapacity)
        for i in 0..<kvCapacity { posPtr[i] = Int32(i) }

        // KV cache states: [layers, batch, kv_heads, seq, head_dim], seq dim replaced
        // by the requested capacity (it is dynamic in the export).
        func makeState(_ stateName: String, _ desc: NDArrayDescriptor) throws -> StateBinding {
            let seqDim = KVCacheSeqDim.detect(shape: desc.shape)
            var shape = desc.shape
            shape[seqDim] = kvCapacity
            guard !shape.contains(where: { $0 < 0 }) else {
                throw SpecDecodeError("state '\(stateName)' has unresolved dims \(shape)")
            }
            let resolved = desc.resolvingDynamicDimensions(shape)
            let byteCount = resolved.minimumByteCount
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)
            else {
                throw SpecDecodeError("state '\(stateName)' allocation failed (\(byteCount) bytes)")
            }
            memset(buffer.contents(), 0, byteCount)
            return StateBinding(
                name: stateName, buffer: buffer, scalarType: desc.scalarType,
                shape: shape, strides: resolved.preferredStrides, byteCount: byteCount)
        }
        keyState = try makeState(keyName, keyDesc)
        valueState = try makeState(valueName, valueDesc)

        guard let loaded = try prepared.model.loadFunction(named: "main") else {
            throw SpecDecodeError("cannot load function 'main'")
        }
        fn = loaded
    }

    var kvByteCount: Int { keyState.byteCount + valueState.byteCount }

    // MARK: - Stream state

    /// Fresh context: zeroed KV, `processed` back to 0.
    func reset() {
        memset(keyState.buffer.contents(), 0, keyState.byteCount)
        memset(valueState.buffer.contents(), 0, valueState.byteCount)
        processed = 0
        lastRows = 0
    }

    /// Accept `count` of the tokens fed by the last forward. Rejected rows keep their
    /// stale KV entries, which the next forward overwrites before anything reads them.
    func commit(_ count: Int) {
        precondition(count >= 0 && count <= lastRows, "commit \(count) of \(lastRows) rows")
        processed += count
    }

    var room: Int { kvCapacity - processed }

    // MARK: - Forward

    /// Run one forward over `tokens` at the current `processed` offset. All
    /// `tokens.count` logit rows are left in the owned buffer for `argmax(row:)`;
    /// `processed` is unchanged until `commit(_:)`.
    func forward(_ tokens: [Int32]) async throws {
        guard !tokens.isEmpty else { throw SpecDecodeError("empty forward") }
        guard tokens.count <= maxQuery else {
            throw SpecDecodeError("query \(tokens.count) exceeds maxQuery \(maxQuery)")
        }
        guard processed + tokens.count <= kvCapacity else {
            throw SpecDecodeError(
                "context \(processed) + \(tokens.count) exceeds KV capacity \(kvCapacity)")
        }

        let queryLength = tokens.count
        let positions = processed + queryLength

        let idPtr = idBuffer.contents().bindMemory(to: Int32.self, capacity: queryLength)
        for i in 0..<queryLength { idPtr[i] = tokens[i] }

        let idShape = [1, queryLength]
        let idStrides = try resolvedStrides(descriptor: inDesc, shape: idShape)
        let idValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: idBuffer, byteOffset: 0,
            scalarType: inDesc.scalarType, shape: idShape, strides: idStrides)

        let posShape = [1, positions]
        let posStrides = try resolvedStrides(descriptor: posDesc, shape: posShape)
        let posValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: posBuffer, byteOffset: 0,
            scalarType: posDesc.scalarType, shape: posShape, strides: posStrides)

        var keyValue = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: keyState.buffer, byteOffset: 0, scalarType: keyState.scalarType,
            shape: keyState.shape, strides: keyState.strides)
        var valueValue = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: valueState.buffer, byteOffset: 0, scalarType: valueState.scalarType,
            shape: valueState.shape, strides: valueState.strides)
        var states = InferenceFunction.AsyncMutableViews()
        states.insert(&keyValue, for: keyState.name)
        states.insert(&valueValue, for: valueState.name)

        let logitsShape = [1, queryLength, vocab]
        let logitsStrides = try resolvedStrides(descriptor: logitsDesc, shape: logitsShape)
        var logitsValue = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: logitsBuffer, byteOffset: 0, scalarType: .float16,
            shape: logitsShape, strides: logitsStrides)
        var outputs = InferenceFunction.AsyncMutableViews()
        outputs.insert(&logitsValue, for: logitsName)

        let start = ContinuousClock.now
        let _ = try fn.encode(
            inputs: [inName: idValue, posName: posValue],
            states: consume states, outputViews: consume outputs, to: computeStream)
        await computeStream.currentWorkCompleted()
        lastForwardMs = milliseconds(since: start)

        lastRows = queryLength
        lastRowStride = logitsStrides[1]
        forwards += 1
    }

    // MARK: - Logits

    /// Greedy argmax over ONE logits row, read straight from the fp16 output buffer.
    /// Flattening `[1, S, vocab]` to `[Float]` on the CPU was HALF the decode wall in
    /// the 27B spec-decode work — never do it here.
    func argmax(row: Int) -> Int32 {
        argmaxWithValue(row: row).token
    }

    func argmaxWithValue(row: Int) -> (token: Int32, value: Float) {
        precondition(row >= 0 && row < lastRows, "row \(row) of \(lastRows)")
        let ptr = logitsBuffer.contents().bindMemory(
            to: Float16.self, capacity: logitsElementCapacity)
        let base = row * lastRowStride
        var bestIndex = 0
        var bestValue = ptr[base]
        for j in 1..<vocab where ptr[base + j] > bestValue {
            bestValue = ptr[base + j]
            bestIndex = j
        }
        return (Int32(bestIndex), Float(bestValue))
    }
}

// MARK: - Helpers

enum KVCacheSeqDim {
    /// [L, B, H, S, D] → 3 · [B, H, S, D] → 2 (same rule as the pipelined engine).
    static func detect(shape: [Int]) -> Int { shape.count == 5 ? 3 : 2 }
}

func milliseconds(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) * 1000
        + Double(d.components.attoseconds) / 1e15
}

func seconds(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}
