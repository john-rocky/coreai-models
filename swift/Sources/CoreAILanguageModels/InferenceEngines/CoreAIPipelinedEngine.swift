// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Metal
import MetalPerformanceShaders
import Synchronization
import os

// MARK: - Timing

private func milliseconds(since start: ContinuousClock.Instant) -> Double {
    let duration = ContinuousClock.now - start
    let (secs, attoseconds) = duration.components
    return (Double(secs) + Double(attoseconds) / 1e18) * 1000.0
}

// MARK: - Constants

/// Maximum number of in-flight pipeline stages. Shared by the backpressure gate
/// and all buffer rotation logic to guarantee no two concurrent stages alias
/// the same memory.
private let pipelineDepth = 3
private let averageExpectedPromptSize = 256
private let temperatureTolerance: Double = 0.001

/// MPSNDArray enforces 64-byte row-stride alignment on backing buffers.
private let minimumMPSNDArrayBufferSize = 64

// MARK: - Core AI Pipelined Engine (Public Wrapper)

/// GPU-pipelined inference engine using Core AI's encode API.
///
/// Key features:
/// - Non-blocking GPU encoding via `InferenceFunction.encode`
/// - GPU-direct token sampling (argmax/topK) via MPSGraph compute shaders
/// - Pipeline-depth-matched buffer rotation for CPU/GPU overlap
/// - Growing KV cache with pipelined expansion
/// - All tensors are owned MTLBuffers — Core AI never allocates/frees them
final class CoreAIPipelinedEngine: InferenceEngine, Sendable {
    typealias ConfigType = ModelConfig

    nonisolated(unsafe) private var engine: EngineImpl

    /// Token history for implicit prefix caching. Marked nonisolated(unsafe) because
    /// mutations are serialized by the generation lifecycle: generate() awaits any prior
    /// Task before starting, and the forwarding `async let` only appends tokens while
    /// runCompletion holds the engine lock. No concurrent writes are possible when the
    /// cancel-and-await contract is upheld.
    nonisolated(unsafe) private var history = TokenHistory()
    nonisolated(unsafe) private(set) var lastPrefixHitCount: Int = 0
    private let engineInUse = Atomic<Bool>(false)
    let config: ModelConfig

    // Generation lifecycle
    private let _activeToken = Mutex<GenerationToken?>(nil)
    private let _generationTask = Mutex<Task<Void, Never>?>(nil)

    var isBusy: Bool { _activeToken.withLock { $0 != nil } }

    var processedTokenCount: Int { engine.processedTokenCount }

    init(
        config: ModelConfig,
        preparedModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        let engine = try await EngineImpl(
            config: config, preparedModel: preparedModel, options: options)
        self.engine = engine
        self.config = config
    }

    /// Atomically claim exclusive use of `engine`.
    ///
    /// Traps on contention. Callers must guarantee single-ownership.
    private func acquireEngine() {
        let (exchanged, _) = engineInUse.compareExchange(
            expected: false,
            desired: true,
            ordering: .acquiring
        )
        guard exchanged else {
            fatalError("Trying to acquire engine when it's already in use")
        }
    }

    /// Try to claim exclusive use of `engine` without trapping.
    ///
    /// Returns `true` if the caller now holds it (and must call `releaseEngine`), `false` if
    /// another caller holds it.
    private func tryAcquireEngine() -> Bool {
        let (exchanged, _) = engineInUse.compareExchange(
            expected: false,
            desired: true,
            ordering: .acquiring
        )
        return exchanged
    }

    private func releaseEngine() {
        engineInUse.store(false, ordering: .releasing)
    }

    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        if inferenceOptions.includeLogits {
            throw InferenceRuntimeError.invalidArgument(
                "CoreAI pipelined engine does not support logits (GPU-side sampling). "
                    + "Use a sequential engine for constrained generation or evaluation."
            )
        }
        if inferenceOptions.forcedContinuation != nil {
            throw InferenceRuntimeError.invalidArgument(
                "CoreAI pipelined engine does not support forcedContinuation (GPU-side sampling). "
                    + "Use a sequential engine for evaluation."
            )
        }

        // Serialize: if a prior generation is still winding down (GPU drain),
        // cancel it and wait for the engine slot to be released.
        if let priorTask = _generationTask.withLock({ $0 }) {
            _activeToken.withLock { $0?.cancel() }
            await priorTask.value
        }

        let maxTokens = inferenceOptions.maxTokens
        let stopReasonStore = StopReasonStore()
        let (base, outputContinuation) =
            AsyncThrowingStream<InferenceOutput, any Error>.makeStream()

        let token = GenerationToken()
        _activeToken.withLock { $0 = token }

        let task = Task {
            self.acquireEngine()
            defer {
                self.releaseEngine()
                // Only clear if this generation still owns both slots
                if self._activeToken.withLock({ $0 === token }) {
                    self._activeToken.withLock { $0 = nil }
                    self._generationTask.withLock { $0 = nil }
                }
            }
            do {
                let (tokenStream, tokenContinuation) =
                    AsyncThrowingStream<InferenceEngine.TokenId, any Error>.makeStream()

                outputContinuation.onTermination = { @Sendable _ in
                    tokenContinuation.finish()
                }

                // Implicit prefix caching: resolve input against history
                var (commonPrefix, resolvedNewTokens) = self.history.resolve(input: input)
                self.lastPrefixHitCount = commonPrefix

                // Detect TRUE divergence before backup (tokens actually differ)
                let isDivergence = commonPrefix < input.count && commonPrefix < self.history.count

                // Ensure at least 1 token for prefill (seeds the decode loop).
                // Back up by 1 if the entire input is cached.
                if resolvedNewTokens.isEmpty && commonPrefix > 0 {
                    commonPrefix -= 1
                    resolvedNewTokens = input[commonPrefix...]
                }

                // A rewind (processedTokenCount going backwards) replays positions whose
                // KV entries are simply overwritten — but non-truncatable recurrent states
                // (hybrid GDN/SSM) hold a running scan that cannot be rewound, so any
                // rewind on those models must become a full reset and replay.
                let needsRewind = commonPrefix < self.engine.processedTokenCount
                if isDivergence || (needsRewind && self.engine.hasNonTruncatableStates) {
                    // Tokens differ — full reset (partial rewind corrupts buffer rotation)
                    await self.engine.computeStream.currentWorkCompleted()
                    self.engine.reset()
                    self.history.clear()
                    resolvedNewTokens = input[...]
                    commonPrefix = 0
                    self.lastPrefixHitCount = 0
                } else {
                    if needsRewind {
                        // Pure extension — partial rewind (buffer phase preserved)
                        await self.engine.computeStream.currentWorkCompleted()
                        self.engine.processedTokenCount = commonPrefix
                        self.engine.step = commonPrefix
                        self.engine.lastSampledToken = nil
                    }
                    // Re-establish the invariant history == input[0..<commonPrefix] before
                    // the prefill append below. Covers the backup-by-1 branch on an exact
                    // replay (no rewind needed, but the backed-up token is re-appended —
                    // without this truncate it would duplicate and the NEXT turn would
                    // read as divergence and full-reset).
                    self.history.truncate(to: commonPrefix)
                }

                let newTokens = Array(resolvedNewTokens)

                async let forwarding: Void = {
                    do {
                        for try await token in tokenStream {
                            // Yield BEFORE recording: a token the consumer never receives
                            // must not enter history — the next turn's prompt extends only
                            // what the consumer saw, so a post-break token recorded here
                            // would read as divergence and force a full re-prefill. The KV
                            // entry such a token leaves behind is trimmed by the rewind
                            // above on the next generate().
                            let result = outputContinuation.yield(InferenceOutput(tokenId: token))
                            if case .terminated = result {
                                tokenContinuation.finish()
                                break
                            }
                            self.history.append(token)
                        }
                    } catch {
                        outputContinuation.finish(throwing: error)
                    }
                }()

                // Track prefill tokens BEFORE runCompletion — the forwarding loop
                // concurrently appends generated tokens, so prefill must come first.
                if !newTokens.isEmpty {
                    self.history.append(contentsOf: newTokens[...])
                }

                try await self.engine.runCompletion(
                    prompt: newTokens,
                    sampler: samplingConfiguration,
                    maxTokens: maxTokens,
                    yieldingTo: tokenContinuation
                )

                tokenContinuation.finish()
                await forwarding
                stopReasonStore.setIfUnset(.maxTokens)
                outputContinuation.finish()
            } catch is CancellationError {
                stopReasonStore.set(.cancelled)
                outputContinuation.finish()
            } catch {
                stopReasonStore.set(.error)
                outputContinuation.finish(throwing: error)
            }
        }
        _generationTask.withLock { $0 = task }
        return GenerationSequence(base: base, stopReasonStore: stopReasonStore)
    }

    /// Wait for any in-flight generate() Task to return the engine.
    private func drain() {
        var attempts = 0
        while engineInUse.load(ordering: .acquiring) {
            attempts += 1
            if attempts > 5000 {
                fatalError("Engine not returned after drain() — tokenSequence Task stuck?")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    func cancel() async throws {
        let task: Task<Void, Never>? = _generationTask.withLock { task in
            task?.cancel()
            defer { task = nil }
            return task
        }
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }
        await task?.value
    }

    func reset(to tokenIndex: Int) async throws {
        precondition(
            tokenIndex >= 0 && tokenIndex <= processedTokenCount,
            "reset(to: \(tokenIndex)) out of range [0, \(processedTokenCount)]")
        if tokenIndex == 0 {
            // Full reset: cancel + drain + clear everything
            _activeToken.withLock {
                $0?.cancel()
                $0 = nil
            }
            _generationTask.withLock {
                $0?.cancel()
                $0 = nil
            }
            drain()
            await engine.computeStream.currentWorkCompleted()
            guard tryAcquireEngine() else { return }
            defer { releaseEngine() }
            engine.reset()
            history.clear()
        } else {
            // Partial reset: wait for generation to finish naturally, then rewind counter.
            // Do NOT cancel — cancelling corrupts the pipeline's double-buffer state.
            // The KV cache is valid up to processedTokenCount after natural completion.
            if engine.hasNonTruncatableStates {
                throw InferenceRuntimeError.invalidState(
                    "Partial reset is not supported for hybrid models with recurrent state. "
                        + "Use reset(to: 0) and replay the prefix.")
            }
            drain()
            await engine.computeStream.currentWorkCompleted()
            guard tryAcquireEngine() else { return }
            defer { releaseEngine() }
            engine.processedTokenCount = tokenIndex
            engine.step = tokenIndex
            engine.lastSampledToken = nil
            history.truncate(to: tokenIndex)
        }
    }

    func cleanup() async throws {
        let cleanupSpan = InstrumentsProfiler.beginCleanup(engine: "CoreAI-Pipelined")
        if tryAcquireEngine() {
            let stream = engine.computeStream
            releaseEngine()
            await stream.currentWorkCompleted()
        }
        cleanupSpan.end()
    }

    func validateSamplingStrategy(_ config: SamplingConfiguration) throws {
        // All sampling configurations are now supported by the GPU sampler:
        // greedy, temperature, topK, topP, and minP.
    }

    func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {
        acquireEngine()
        defer { releaseEngine() }
        try await engine.performWarmup(queryLength: queryLength, samplingConfig: sampling)
    }
}

// MARK: - Pipeline Depth Gate

/// Bounds in-flight encode calls so MPSGraph's per-encode scratch
/// (sized by the graph's max shape — multiple GB on large models) can't accumulate.
///
/// Without this, the decode loop submits encodes (~220/s) faster than the
/// sampler callback drains them (~70/s); depth grows until
/// `MPSCommandBufferImageCache` fails to allocate another private MTLBuffer.
///
/// Capacity matches `pipelineDepth` — covers {logits encode + sampler commit + optional KV-cache grow};
/// deeper queues only cost memory.
///
/// Class, not actor: `release()` runs synchronously from the Metal callback —
/// an actor would force `Task { await release() }` with ordering ambiguity.
/// `internal` (not `private`) so `PipelineGateTests` can reach it.
final class PipelineGate: Sendable {
    private struct State: Sendable {
        var inFlight: Int = 0
        var waiters: [CheckedContinuation<Void, Never>] = []
    }

    private let capacity: Int
    private let state = OSAllocatedUnfairLock<State>(initialState: State())

    init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    /// Take a slot; suspend if all slots are busy.
    func acquire() async {
        // Fast path: take a slot without suspending.
        let takenImmediately = state.withLock { state -> Bool in
            guard state.inFlight < capacity else { return false }
            state.inFlight += 1
            return true
        }
        if takenImmediately { return }

        // Slow path: enqueue a waiter. Re-check under the lock in case a slot
        // opened between the fast path and now.
        await withCheckedContinuation { cont in
            let runImmediately = state.withLock { state -> Bool in
                if state.inFlight < capacity {
                    state.inFlight += 1
                    return true
                }
                state.waiters.append(cont)
                return false
            }
            if runImmediately { cont.resume() }
        }
    }

    /// Give back a slot. Called from the sampler's GPU-completion callback on a
    /// Metal callback thread; resumes any pending waiter (slot transferred
    /// directly without decrementing `inFlight`) or decrements the count.
    ///
    /// The waiter is resumed *outside* the lock so a rescheduled task can't
    /// re-enter `acquire` while we still hold it.
    func release() {
        let waiter = state.withLock { state -> CheckedContinuation<Void, Never>? in
            if !state.waiters.isEmpty {
                // Slot transferred to the woken waiter — inFlight count unchanged.
                return state.waiters.removeFirst()
            }
            state.inFlight -= 1
            return nil
        }
        waiter?.resume()
    }

    // Test-only introspection. Kept as underscored names to discourage
    // production use; exercised by PipelineGateTests.

    var _inFlightForTesting: Int {
        state.withLock { $0.inFlight }
    }

    var _waitersForTesting: Int {
        state.withLock { $0.waiters.count }
    }
}

// MARK: - Per-Token Inputs

/// A model input beyond `input_ids`/`position_ids` whose value depends on the token id of
/// the step being encoded — e.g. Gemma's per-layer-embedding rows, gathered by token id from
/// a host-side mmap table too large to live in the graph. The engine owns one buffer holding
/// `maxContextLength` per-step slots (slot index = token position, so in-flight prefill steps
/// each read a distinct region and host writes never race the GPU), fills the step's slot via
/// `EngineOptions.perTokenInputProvider`, and binds it as an additional input on every encode.
private struct PipelinedPerTokenInput {
    let name: String
    let buffer: MTLBuffer
    let scalarType: NDArray.ScalarType
    let shape: [Int]
    let strides: [Int]
    let stepByteCount: Int
}

// MARK: - Static Inputs

/// A model input bound to the SAME host buffer on every encode — e.g. a giant quantized
/// embedding table mmap'd from disk that the graph gathers from in-graph by token id
/// (Gemma 4's per-layer-embedding table as `ple_table`/`ple_scale` inputs). The buffer is
/// supplied by `EngineOptions.staticInputBuffers`, is never written, and imposes no per-step
/// host work — unlike per-token inputs there is no S=1 constraint and no decode-loop wait on
/// the sampled token, so the full pipeline depth survives.
private struct PipelinedStaticInput {
    let name: String
    let buffer: MTLBuffer
    let scalarType: NDArray.ScalarType
    let shape: [Int]
    let strides: [Int]
}

/// One-shot rendezvous between the sampler's GPU-completion callback (which learns the
/// sampled token on a Metal callback thread) and the decode loop (which must know that token
/// BEFORE it can gather per-token inputs for the next step). Either side may arrive first;
/// strict deliver/take alternation lets one instance be reused across steps.
final class TokenRendezvous: Sendable {
    private enum State {
        case idle
        case token(Int32)
        case waiter(CheckedContinuation<Int32, Never>)
    }

    private let state = OSAllocatedUnfairLock<State>(initialState: .idle)

    /// Called from the sampler completion callback with the sampled token.
    func deliver(_ token: Int32) {
        let waiter = state.withLock { state -> CheckedContinuation<Int32, Never>? in
            switch state {
            case .waiter(let continuation):
                state = .idle
                return continuation
            default:
                state = .token(token)
                return nil
            }
        }
        waiter?.resume(returning: token)
    }

    /// Awaited by the encode loop after submitting the sampler for a step.
    func take() async -> Int32 {
        await withCheckedContinuation { continuation in
            let ready = state.withLock { state -> Int32? in
                switch state {
                case .token(let token):
                    state = .idle
                    return token
                default:
                    state = .waiter(continuation)
                    return nil
                }
            }
            if let token = ready { continuation.resume(returning: token) }
        }
    }
}

// MARK: - Engine Implementation

private struct EngineImpl: ~Copyable {
    var vocabSize: Int { config.vocabSize }

    let config: ModelConfig
    let options: EngineOptions
    let function: InferenceFunction
    let pipelineQueue: MTLCommandQueue
    let computeStream: ComputeStream
    let device: MTLDevice

    // Descriptor metadata
    let inputIdsName: String
    let positionIdsName: String
    let keyCacheName: String
    let valueCacheName: String
    let logitsOutputName: String
    let keyCacheScalarType: NDArray.ScalarType
    let valueCacheScalarType: NDArray.ScalarType

    // Base descriptors for shape resolution (preferredStrides, not contiguous)
    let inputIdsBaseDesc: NDArrayDescriptor
    let positionIdsBaseDesc: NDArrayDescriptor
    let logitsBaseDesc: NDArrayDescriptor

    // Owned MTLBuffers
    var inputTokensBuffer: MTLBuffer
    var cachePositionBuffers: [MTLBuffer]
    var decodeOutputBuffers: [MTLBuffer]
    var decodeLogitsBuffers: [MTLBuffer]

    // KV cache — reuses CoreAIKVCache protocol from KVCache+CoreAI.swift
    var kvCache: any CoreAIKVCache

    // Linear attention state bindings for hybrid models (nil for pure transformer models).
    // States 0/1 are KV cache; additional states handled by handler.
    var additionalStates: FixedMTLBufferState?
    var hasNonTruncatableStates: Bool

    // Per-token inputs beyond input_ids/position_ids (host-gathered, e.g. Gemma PLE rows)
    let perTokenInputs: [PipelinedPerTokenInput]
    let perTokenInputProvider: PerTokenInputProvider?
    let sampledTokenRendezvous = TokenRendezvous()

    // Static inputs beyond input_ids/position_ids (same buffer every encode, e.g. mmap'd
    // gather tables — see PipelinedStaticInput)
    let staticInputs: [PipelinedStaticInput]

    // Optional static-chunk prefill function (multifunction bundles: "main" = S=1
    // decode, "prefill" = S=C chunk, weights shared). Static S sidesteps the
    // MPSGraph GPURegionRuntime crash on dynamic-ids graphs while keeping the
    // externalized composite kernels; position/KV dims stay dynamic as in the
    // S=1 ship contract.
    let prefillFunction: InferenceFunction?
    let prefillChunkLength: Int
    let prefillInputIdsBaseDesc: NDArrayDescriptor?
    let prefillPositionIdsBaseDesc: NDArrayDescriptor?
    let prefillLogitsBaseDesc: NDArrayDescriptor?
    let prefillLogitsBuffer: MTLBuffer?

    // Logits — reuses GrowingLogitsBuffer from TensorStorage+CoreAI.swift
    var logits: GrowingLogitsBuffer

    // GPU sampler — reuses MPSGraphSampler from MPSGraphSamplers.swift
    var cachedSampler: (any MPSGraphSampler)?
    var cachedSamplerTemperature: Double?

    // State
    var processedTokenCount: Int = 0
    var step: Int = 0
    // Last GPU-sampled token, mirrored to the CPU only when per-token inputs need it
    // (the decode loop must gather the next step's rows for this token).
    var lastSampledToken: Int32? = nil

    // Backpressure gate — see PipelineGate doc-comment for the failure mode it prevents.
    // Capacity matches pipeline depth: {encode logits + sampler commit + optional KV-cache grow} in flight.
    let inFlightGate = PipelineGate(capacity: pipelineDepth)

    // MARK: - Init

    init(
        config: ModelConfig,
        preparedModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw InferenceRuntimeError.genericError("Failed to create Metal device")
        }

        let model = preparedModel.model

        // Get function descriptor
        guard let descriptor = model.functionDescriptor(for: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot find function '\(config.function)' in model")
        }

        // Validate: 2+ inputs (input_ids, position_ids, plus optional static or
        // host-gathered per-token inputs), 1+ output, 2–4 states
        guard descriptor.inputNames.count >= 2 else {
            throw InferenceRuntimeError.invalidInputType(
                "Expected at least 2 inputs, got \(descriptor.inputNames.count): \(descriptor.inputNames)")
        }
        // Inputs beyond the first two are static (a buffer was supplied for the name in
        // EngineOptions.staticInputBuffers) or per-token (filled by the provider per step).
        let extraInputNames = Array(descriptor.inputNames.dropFirst(2))
        let staticInputNames = extraInputNames.filter { options.staticInputBuffers[$0] != nil }
        let perTokenInputNames = extraInputNames.filter { options.staticInputBuffers[$0] == nil }
        guard perTokenInputNames.count <= Self.maxPerTokenInputs else {
            throw InferenceRuntimeError.invalidInputType(
                "At most \(Self.maxPerTokenInputs) per-token inputs beyond input_ids/position_ids "
                    + "are supported, got \(perTokenInputNames.count): \(perTokenInputNames) "
                    + "(constant inputs can ride EngineOptions.staticInputBuffers instead)")
        }
        guard staticInputNames.count <= Self.maxStaticInputs else {
            throw InferenceRuntimeError.invalidInputType(
                "At most \(Self.maxStaticInputs) static inputs are supported, "
                    + "got \(staticInputNames.count): \(staticInputNames)")
        }
        guard descriptor.outputNames.count >= 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got \(descriptor.outputNames.count)")
        }
        guard descriptor.stateNames.count >= 2 && descriptor.stateNames.count <= 4 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected 2–4 states, got \(descriptor.stateNames.count): \(descriptor.stateNames)"
            )
        }

        // Classify states using the shared factory logic
        let classified = StateHandlerFactory.classifyStates(
            descriptor: descriptor, stateKinds: nil, verbose: descriptor.stateNames.count > 2)

        // Find the growing KV pair (first two states with .kvCache kind)
        let growingNames = classified.filter { $0.kind == .kvCache }.map(\.name)
        guard growingNames.count >= 2 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 2 growing KV cache states, found \(growingNames.count) "
                    + "in: \(classified.map { "\($0.name)=\($0.kind.rawValue)" })")
        }
        let keyCacheName = growingNames[0]
        let valueCacheName = growingNames[1]

        // Fixed states: everything that isn't the primary growing KV pair
        let fixedNames =
            classified
            .filter { $0.kind == .slidingCache || $0.kind == .fixed }
            .map(\.name)
        // Additional growing states beyond the primary pair
        let extraGrowingNames = Array(growingNames.dropFirst(2))

        // Extract names
        let inputIdsName = descriptor.inputNames[0]
        let positionIdsName = descriptor.inputNames[1]
        let logitsOutputName = descriptor.outputNames[0]

        // Extract state descriptors for KV cache shape/type
        guard case .ndArray(let keyCacheDesc) = descriptor.stateDescriptor(of: keyCacheName),
            case .ndArray(let valueCacheDesc) = descriptor.stateDescriptor(of: valueCacheName)
        else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get KV cache state descriptors")
        }

        // Extract input descriptors
        guard case .ndArray(let inputIdsDesc) = descriptor.inputDescriptor(of: inputIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(inputIdsName)'")
        }
        guard case .ndArray(let posIdsDesc) = descriptor.inputDescriptor(of: positionIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(positionIdsName)'")
        }

        // Extract logits descriptor
        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsOutputName) else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get descriptor for '\(logitsOutputName)'")
        }
        guard logitsDesc.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Only float16 logits supported, got \(logitsDesc.scalarType)")
        }

        // Static inputs: the caller-supplied buffer is bound unchanged on every encode.
        var staticInputsLocal: [PipelinedStaticInput] = []
        for name in staticInputNames {
            guard case .ndArray(let desc) = descriptor.inputDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidInputType(
                    "Cannot get descriptor for static input '\(name)'")
            }
            guard !desc.shape.contains(where: { $0 < 0 }) else {
                throw InferenceRuntimeError.invalidInputType(
                    "Static input '\(name)' has dynamic dims \(desc.shape) — static inputs "
                        + "must be fixed-shape")
            }
            let resolved = desc.resolvingDynamicDimensions(desc.shape)
            let buffer = options.staticInputBuffers[name]!.buffer
            guard buffer.length >= resolved.minimumByteCount else {
                throw InferenceRuntimeError.invalidInputType(
                    "Static input '\(name)' needs \(resolved.minimumByteCount) bytes but the "
                        + "supplied buffer holds \(buffer.length)")
            }
            staticInputsLocal.append(
                PipelinedStaticInput(
                    name: name,
                    buffer: buffer,
                    scalarType: desc.scalarType,
                    shape: desc.shape,
                    strides: resolved.preferredStrides
                ))
        }
        if !staticInputsLocal.isEmpty {
            let fmt = ByteCountFormatter()
            fmt.countStyle = .memory
            let total = staticInputsLocal.reduce(0) { $0 + $1.buffer.length }
            CLILogger.log(
                "Pipelined engine carrying \(staticInputsLocal.count) static input(s): "
                    + staticInputsLocal.map(\.name).joined(separator: ", ")
                    + " (\(fmt.string(fromByteCount: Int64(total))) bound per encode)")
        }

        // Per-token inputs: fixed-shape, filled by the host provider once per step.
        // One owned buffer holds maxContextLength slots (slot = token position) so
        // in-flight steps read disjoint regions.
        var perTokenInputsLocal: [PipelinedPerTokenInput] = []
        for name in perTokenInputNames {
            guard case .ndArray(let desc) = descriptor.inputDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidInputType(
                    "Cannot get descriptor for per-token input '\(name)'")
            }
            guard !desc.shape.contains(where: { $0 < 0 }) else {
                throw InferenceRuntimeError.invalidInputType(
                    "Per-token input '\(name)' has dynamic dims \(desc.shape) — per-token inputs "
                        + "must be fixed-shape (S=1)")
            }
            let resolved = desc.resolvingDynamicDimensions(desc.shape)
            let stepByteCount = resolved.minimumByteCount
            let byteCount = config.maxContextLength * stepByteCount
            guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed(
                    "per-token input '\(name)' (\(byteCount) bytes)")
            }
            memset(buf.contents(), 0, byteCount)
            perTokenInputsLocal.append(
                PipelinedPerTokenInput(
                    name: name,
                    buffer: buf,
                    scalarType: desc.scalarType,
                    shape: desc.shape,
                    strides: resolved.preferredStrides,
                    stepByteCount: stepByteCount
                ))
        }
        if !perTokenInputsLocal.isEmpty {
            guard options.perTokenInputProvider != nil else {
                throw InferenceRuntimeError.invalidInputType(
                    "Model declares per-token input(s) "
                        + perTokenInputsLocal.map(\.name).joined(separator: ", ")
                        + " but EngineOptions.perTokenInputProvider is nil — set a provider "
                        + "that gathers the rows for each token id")
            }
            let fmt = ByteCountFormatter()
            fmt.countStyle = .memory
            let total = perTokenInputsLocal.reduce(0) { $0 + $1.buffer.length }
            CLILogger.log(
                "Pipelined engine carrying \(perTokenInputsLocal.count) per-token input(s): "
                    + perTokenInputsLocal.map(\.name).joined(separator: ", ")
                    + " (\(fmt.string(fromByteCount: Int64(total))) slots)")
        }

        // Allocate inputTokens MTLBuffer
        let inputTokensByteCount = config.maxContextLength * inputIdsDesc.scalarType.byteSize
        guard let inputTokensBuf = device.makeBuffer(length: inputTokensByteCount, options: .storageModeShared) else {
            throw InferenceRuntimeError.bufferAllocationFailed("inputTokens (\(inputTokensByteCount) bytes)")
        }

        // Allocate pipeline-depth-matched cache position buffers
        let cachePosSize = config.maxContextLength * posIdsDesc.scalarType.byteSize
        var cachePosBuffers: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            guard let buf = device.makeBuffer(length: cachePosSize, options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed("cachePositions (\(cachePosSize) bytes)")
            }
            cachePosBuffers.append(buf)
        }

        // Pre-populate cache positions with [0, 1, ..., maxCtx-1]
        for buf in cachePosBuffers {
            let ptr = buf.contents().bindMemory(to: Int32.self, capacity: config.maxContextLength)
            for i in 0..<config.maxContextLength {
                ptr[i] = Int32(i)
            }
        }

        // Allocate pipeline-depth-matched decode output buffers (sampler writes next token)
        var decodeOutBuffers: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            let decodeOutSize = max(minimumMPSNDArrayBufferSize, MemoryLayout<Int32>.size)
            guard let buf = device.makeBuffer(length: decodeOutSize, options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed(
                    "decodeOutputBuffer (\(decodeOutSize) bytes)")
            }
            decodeOutBuffers.append(buf)
        }

        // Allocate pipeline-depth-matched decode logits buffers (inference writes logits for decode)
        let decodeLogitsSize = config.vocabSize * MemoryLayout<UInt16>.size
        var decodeLogBufs: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            guard let buf = device.makeBuffer(length: decodeLogitsSize, options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed("decodeLogitsBuffer (\(decodeLogitsSize) bytes)")
            }
            decodeLogBufs.append(buf)
        }

        // Create KV cache using factory — pass original descriptors (with -1 dynamic dims intact)
        // so the factory can correctly detect growing vs static support via isDynamicKVCache().
        let kvCacheLocal = try KVCacheFactory.make(
            options: options,
            device: device,
            keyReqs: keyCacheDesc,
            valueReqs: valueCacheDesc,
            maxContextLength: config.maxContextLength
        )

        let resolvedSize = options.resolvedKVCacheSize(maxContextLength: config.maxContextLength)
        CLILogger.log("Created \(options.kvCacheStrategy) KV cache with size \(resolvedSize, default: "nil")")

        // Allocate fixed-size buffers for additional persistent states (sliding caches, hybrid states).
        var additionalStatesLocal: FixedMTLBufferState? = nil
        let allFixedNames = fixedNames + extraGrowingNames  // extra growing get resolved to max size
        if !allFixedNames.isEmpty {
            var extraStates: [(name: String, descriptor: NDArrayDescriptor)] = []
            for name in allFixedNames {
                guard case .ndArray(let desc) = descriptor.stateDescriptor(of: name) else {
                    throw InferenceRuntimeError.invalidOutputType(
                        "Cannot get descriptor for persistent state '\(name)'")
                }
                // Resolve dynamic dims to max for any extra growing states
                let resolved =
                    desc.shape.contains(where: { $0 < 0 })
                    ? desc.resolvingDynamicDimensions(desc.shape.map { $0 < 0 ? config.maxContextLength : $0 })
                    : desc
                extraStates.append((name, resolved))
            }
            additionalStatesLocal = try FixedMTLBufferState(states: extraStates, device: device)
            CLILogger.log(
                "Pipelined additional states: \(allFixedNames.joined(separator: ", "))")
        }

        // Create growing logits buffer (reuses TensorStorage+CoreAI.swift).
        // A fully static logits output (e.g. a decode-only S=1 graph: [1, 1, vocab])
        // can't be resolved at a larger capacity — size the buffer to its static
        // sequence length instead of the prompt-sized default.
        let logitsSeqIsStatic = logitsDesc.shape.count >= 2 && logitsDesc.shape[1] > 0
        let logitsRef = try GrowingLogitsBuffer(
            device: device,
            descriptor: descriptor,
            name: logitsOutputName,
            vocabSize: config.vocabSize,
            maxCapacity: logitsSeqIsStatic ? logitsDesc.shape[1] : config.maxContextLength,
            initialCapacity: logitsSeqIsStatic ? logitsDesc.shape[1] : averageExpectedPromptSize
        )

        // Load inference function
        guard let fn = try model.loadFunction(named: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot load function '\(config.function)'")
        }

        // Optional "prefill" function (static S=C chunk graph). Present only in
        // multifunction bundles; absence leaves behavior unchanged.
        var prefillFnLocal: InferenceFunction? = nil
        var prefillChunkLenLocal = 0
        var prefillIdsDescLocal: NDArrayDescriptor? = nil
        var prefillPosDescLocal: NDArrayDescriptor? = nil
        var prefillLogitsDescLocal: NDArrayDescriptor? = nil
        var prefillLogitsBufLocal: MTLBuffer? = nil
        if config.function == "main",
            let pfDesc = model.functionDescriptor(for: "prefill"),
            case .ndArray(let pfIds) = pfDesc.inputDescriptor(of: inputIdsName),
            case .ndArray(let pfPos) = pfDesc.inputDescriptor(of: positionIdsName),
            case .ndArray(let pfLogits) = pfDesc.outputDescriptor(of: logitsOutputName),
            pfIds.shape.count == 2, pfIds.shape[1] > 1
        {
            let chunkLen = pfIds.shape[1]
            let pfLogitsResolved = pfLogits.resolvingDynamicDimensions(
                [1, chunkLen, config.vocabSize])
            guard
                let pfLogitsBuf = device.makeBuffer(
                    // MPSNDArray enforces a 64-byte minimum backing-buffer length.
                    length: max(64, pfLogitsResolved.minimumByteCount),
                    options: .storageModeShared)
            else {
                throw InferenceRuntimeError.bufferAllocationFailed(
                    "prefill logits (\(pfLogitsResolved.minimumByteCount) bytes)")
            }
            prefillFnLocal = try model.loadFunction(named: "prefill")
            prefillChunkLenLocal = chunkLen
            prefillIdsDescLocal = pfIds
            prefillPosDescLocal = pfPos
            prefillLogitsDescLocal = pfLogits
            prefillLogitsBufLocal = pfLogitsBuf
            CLILogger.log(
                "Pipelined engine loaded 'prefill' function: static chunk S=\(chunkLen)")
        }

        guard let pipelineQueue = device.makeCommandQueue() else {
            throw InferenceRuntimeError.invalidState(
                "Failed to allocate MTLCommandQueue for CoreAIPipelinedEngine")
        }
        pipelineQueue.label = "CoreAIPipelinedEngine.queue"
        let computeStream = ComputeStream(commandQueue: pipelineQueue)

        // Assign
        self.config = config
        self.options = options
        self.function = fn
        self.pipelineQueue = pipelineQueue
        self.computeStream = computeStream
        self.device = device
        self.inputIdsName = inputIdsName
        self.positionIdsName = positionIdsName
        self.keyCacheName = keyCacheName
        self.valueCacheName = valueCacheName
        self.logitsOutputName = logitsOutputName
        self.keyCacheScalarType = keyCacheDesc.scalarType
        self.valueCacheScalarType = valueCacheDesc.scalarType
        self.inputIdsBaseDesc = inputIdsDesc
        self.positionIdsBaseDesc = posIdsDesc
        self.logitsBaseDesc = logitsDesc
        self.inputTokensBuffer = inputTokensBuf
        self.cachePositionBuffers = cachePosBuffers
        self.decodeOutputBuffers = decodeOutBuffers
        self.decodeLogitsBuffers = decodeLogBufs
        self.kvCache = kvCacheLocal
        self.additionalStates = additionalStatesLocal
        self.hasNonTruncatableStates = classified.contains(where: { $0.kind == .fixed })
        self.perTokenInputs = perTokenInputsLocal
        self.perTokenInputProvider = options.perTokenInputProvider
        self.staticInputs = staticInputsLocal
        self.logits = logitsRef
        self.prefillFunction = prefillFnLocal
        self.prefillChunkLength = prefillChunkLenLocal
        self.prefillInputIdsBaseDesc = prefillIdsDescLocal
        self.prefillPositionIdsBaseDesc = prefillPosDescLocal
        self.prefillLogitsBaseDesc = prefillLogitsDescLocal
        self.prefillLogitsBuffer = prefillLogitsBufLocal
        self.cachedSampler = nil
        self.cachedSamplerTemperature = nil

        CLILogger.log("CoreAI pipelined engine initialized — Vocab: \(config.vocabSize)")
    }

    // MARK: - Per-Token Input Binding

    /// Maximum number of per-token inputs beyond input_ids/position_ids.
    static let maxPerTokenInputs = 2

    /// Maximum number of static inputs (gather tables and the like).
    static let maxStaticInputs = 4

    /// Fill each per-token input's slot for `position` with rows for `token` (via the
    /// provider) and merge the slot bindings into `inputs`. Slot = position keeps in-flight
    /// steps on disjoint buffer regions.
    private func bindPerTokenInputs(
        token: Int32, position: Int,
        into inputs: inout [String: InferenceFunction.AsyncValue]
    ) throws {
        guard let provider = perTokenInputProvider else {
            throw InferenceRuntimeError.invalidState(
                "Per-token inputs present but no provider — engine init should have rejected this")
        }
        for perToken in perTokenInputs {
            let byteOffset = position * perToken.stepByteCount
            provider(
                perToken.name, token, position,
                perToken.buffer.contents() + byteOffset, perToken.stepByteCount)
            inputs[perToken.name] = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: perToken.buffer,
                byteOffset: byteOffset,
                scalarType: perToken.scalarType,
                shape: perToken.shape,
                strides: perToken.strides
            )
        }
    }

    /// Per-token inputs constrain the engine to S=1 steps: each step's rows are gathered for
    /// exactly one token. Run with `COREAI_CHUNK_THRESHOLD=1` so prefill chunks are S=1 too.
    private func requireSingleTokenStep(_ queryLength: Int) throws {
        guard queryLength == 1 else {
            throw InferenceRuntimeError.invalidArgument(
                "Model has per-token inputs — only S=1 steps are supported, got query length "
                    + "\(queryLength). Set COREAI_CHUNK_THRESHOLD=1 so prefill runs as S=1 steps.")
        }
    }

    // MARK: - Sampler

    private mutating func getOrCreateSampler(for config: SamplingConfiguration) throws -> any MPSGraphSampler {
        let config = config.normalized()
        let temperature = config.temperature

        if let existingSampler = cachedSampler, let existingTemp = cachedSamplerTemperature {
            let existingIsGreedy = existingTemp == 0
            let requestedIsGreedy = temperature == 0

            if existingIsGreedy != requestedIsGreedy {
                throw InferenceRuntimeError.genericError(
                    "Sampling configuration changed mid-generation. Call reset() first.")
            }
            if !existingIsGreedy && !requestedIsGreedy
                && abs(existingTemp - temperature) > temperatureTolerance
            {
                throw InferenceRuntimeError.genericError(
                    "Temperature changed mid-generation (\(existingTemp) -> \(temperature)). Call reset() first.")
            }
            return existingSampler
        }

        let newSampler = try MPSGraphSamplerFactory.makeSampler(
            device: device,
            vocabSize: self.config.vocabSize,
            config: config
        )
        cachedSampler = newSampler
        cachedSamplerTemperature = temperature
        return newSampler
    }

    // MARK: - Core Encode Step

    /// Encodes inference + GPU sampling for one step.
    ///
    /// 1. Construct RawView/MutableRawView from MTLBuffers with current shapes
    /// 2. Encode to ComputeStream (non-blocking)
    /// 3. withMetal3Queue: encode GPU argmax/topK (writes to rotating decodeOutputBuffers)
    /// 4. Callback yields token
    private mutating func _encodeNextStepGPU(
        tokens: some Collection<Int32>,
        gpuSampler: any MPSGraphSampler,
        yieldingTo continuation: AsyncThrowingStream<InferenceEngine.TokenId, Error>.Continuation
    ) async throws {
        let currentStep = processedTokenCount

        let actualTokenCount = tokens.isEmpty ? 1 : tokens.count
        let queryLength = actualTokenCount

        // Per-token inputs: resolve this step's token id (prompt token during prefill;
        // the previous GPU-sampled token during decode, mirrored via the rendezvous).
        var perTokenStepToken: Int32? = nil
        if !perTokenInputs.isEmpty {
            try requireSingleTokenStep(queryLength)
            if let promptToken = tokens.first {
                perTokenStepToken = promptToken
            } else if let sampled = lastSampledToken {
                perTokenStepToken = sampled
            } else {
                throw InferenceRuntimeError.invalidState(
                    "Decode step with per-token inputs before any sampled token")
            }
        }

        defer {
            processedTokenCount += actualTokenCount
            step += 1
        }

        let encodeStepID = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIPipelinedEncodeNextStep",
            details: "step=\(currentStep) qLen=\(queryLength)"
        )

        // PrepareStep: write tokens + build views
        let prepareSpan = InstrumentsProfiler.beginPrepareStep(
            step: currentStep, operation: "write+build", engine: "CoreAI-Pipelined")

        // Prefill: write tokens at their natural position so this step's region is disjoint
        // from any prior chunk's region still in-flight on the GPU (encode holds a live
        // MTLBuffer reference; no encodeWriteOperands serialization available in Core AI).
        // Decode: token is in the previous step's decodeOutputBuffer — no CPU write needed.
        let tokenByteOffset = processedTokenCount * MemoryLayout<Int32>.size
        if !tokens.isEmpty {
            let ptr = inputTokensBuffer.contents().bindMemory(
                to: Int32.self, capacity: processedTokenCount + queryLength)
            for (i, token) in tokens.enumerated() {
                ptr[processedTokenCount + i] = token
            }
        }

        // Select cache position buffer for this step (pipeline-depth-matched rotation)
        let cachePosBuffer = cachePositionBuffers[step % pipelineDepth]
        let posLength = processedTokenCount + queryLength

        // Build Inputs as AsyncValue (from MTLBuffers)
        let tokenShape = [1, queryLength]
        let tokenStrides = try resolvedStrides(descriptor: inputIdsBaseDesc, shape: tokenShape)
        let tokenValue: InferenceFunction.AsyncValue
        if tokens.isEmpty {
            // Decode: read input token from previous step's decode output buffer
            tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: decodeOutputBuffers[(step + pipelineDepth - 1) % pipelineDepth],
                byteOffset: 0,
                scalarType: .int32,
                shape: tokenShape,
                strides: tokenStrides
            )
        } else {
            // Prefill: read from inputTokensBuffer at natural position
            tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: inputTokensBuffer,
                byteOffset: tokenByteOffset,
                scalarType: .int32,
                shape: tokenShape,
                strides: tokenStrides
            )
        }
        let posShape = [1, posLength]
        let posStrides = try resolvedStrides(descriptor: positionIdsBaseDesc, shape: posShape)
        let posValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: cachePosBuffer,
            byteOffset: 0,
            scalarType: .int32,
            shape: posShape,
            strides: posStrides
        )

        var asyncInputs: [String: InferenceFunction.AsyncValue] = [
            inputIdsName: tokenValue,
            positionIdsName: posValue,
        ]
        if let stepToken = perTokenStepToken {
            try bindPerTokenInputs(token: stepToken, position: currentStep, into: &asyncInputs)
        }
        // Static inputs: same caller-owned buffer every step, nothing to fill.
        for staticInput in staticInputs {
            asyncInputs[staticInput.name] = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: staticInput.buffer,
                byteOffset: 0,
                scalarType: staticInput.scalarType,
                shape: staticInput.shape,
                strides: staticInput.strides
            )
        }

        // Build States as AsyncMutableValue (KV cache, in-place update)
        let keyBuffer = kvCache.keyBinding.metalBuffer
        let keyShape = kvCache.keyBinding.layout.shape
        let keyStrides = kvCache.keyBinding.layout.strides
        var keyState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: keyBuffer,
            byteOffset: 0,
            scalarType: keyCacheScalarType,
            shape: keyShape,
            strides: keyStrides
        )
        let valBuffer = kvCache.valueBinding.metalBuffer
        let valShape = kvCache.valueBinding.layout.shape
        let valStrides = kvCache.valueBinding.layout.strides
        var valState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: valBuffer,
            byteOffset: 0,
            scalarType: valueCacheScalarType,
            shape: valShape,
            strides: valStrides
        )

        // Build Output as AsyncMutableValue (logits)
        // Decode uses per-step rotating buffer; prefill uses the shared growing buffer.
        let logitsOutputBuffer = tokens.isEmpty ? decodeLogitsBuffers[step % pipelineDepth] : logits.metalBuffer
        let logitsShape = [1, queryLength, vocabSize]
        let logitsStrides = try resolvedStrides(descriptor: logitsBaseDesc, shape: logitsShape)

        prepareSpan.end()

        // Backpressure: cap outstanding encode calls
        await inFlightGate.acquire()

        // Encode inference using the public encode() API.
        // This commits + uses runAfterSyncPoint (no stream wait) — enables true pipelining.
        let logitsSpan = InstrumentsProfiler.beginLogitsInference(
            step: currentStep, tokens: queryLength, engine: "CoreAI-Pipelined")

        // Swift 6 lifetime safety: AsyncMutableViews uses @lifetime(self: &mutableValue)
        // on insert(), so all inserts + consume must be in the same scope without branching.
        try encodeWithStates(
            function: function, inputs: asyncInputs,
            keyState: &keyState, keyCacheName: keyCacheName,
            valState: &valState, valueCacheName: valueCacheName,
            additionalStates: additionalStates,
            logitsBuffer: logitsOutputBuffer, logitsName: logitsOutputName,
            logitsShape: logitsShape, logitsStrides: logitsStrides,
            computeStream: computeStream)
        logitsSpan.end()

        // GPU sampling via Metal queue
        let localGPUSampler = gpuSampler
        let outputBuffer = decodeOutputBuffers[step % pipelineDepth]
        let samplerLogitsBuffer = tokens.isEmpty ? decodeLogitsBuffers[step % pipelineDepth] : logits.metalBuffer
        let logitsOffset = (actualTokenCount - 1) * vocabSize * MemoryLayout<UInt16>.size
        let samplerStrategy = gpuSampler is MPSGraphArgmaxSampler ? "GPU-argmax" : "GPU-composite"
        let samplerTemperature = cachedSamplerTemperature ?? 0.0

        let sampleSpan = InstrumentsProfiler.beginSampleEncoding(
            step: currentStep, strategy: samplerStrategy, temperature: samplerTemperature)

        do {
            let queue = pipelineQueue
            let localInFlightGate = inFlightGate
            let localRendezvous = perTokenInputs.isEmpty ? nil : sampledTokenRendezvous
            let completionCallback: (Int32) -> Void = { nextToken in
                // Release the pipeline slot acquired before encode. Happens on
                // Metal's callback thread — PipelineGate.release() is thread-safe.
                localInFlightGate.release()
                // Mirror the sampled token to the CPU so the next step can gather
                // its per-token inputs (no-op for models without them).
                localRendezvous?.deliver(nextToken)
                InstrumentsProfiler.endCustomInterval(
                    name: "CoreAIPipelinedEncodeNextStep",
                    signpostID: encodeStepID,
                    details: "token=\(nextToken)"
                )
                continuation.yield(nextToken)
            }

            // Order the sampler behind this step's logits write. The model's encode work
            // is committed to the SHARED Metal queue from the stream's internal task
            // machinery, asynchronously — a command buffer committed directly from here
            // can jump ahead of a still-open stream batch and read a not-yet-written
            // logits row (observed: bogus sampled token from a zeroed row at every
            // post-drain boundary). The stream's only public ordering primitive is this
            // full drain; it costs ~30% pipelined decode throughput. (Tried: silgen-
            // shimming the internal appendTask to commit the sampler from the stream's
            // task queue — its execution model reorders against subsequent encodes and
            // scrambles decode entirely. A real fix needs an upstream/SDK ordering API.)
            await computeStream.currentWorkCompleted()
            if queryLength == 1 {
                localGPUSampler.encode(
                    to: queue,
                    logitsBuffer: samplerLogitsBuffer,
                    logitsOffset: logitsOffset,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: completionCallback
                )
            } else {
                localGPUSampler.encodeWithSlice(
                    to: queue,
                    logitsBuffer: samplerLogitsBuffer,
                    queryLength: actualTokenCount,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: completionCallback
                )
            }
        }

        sampleSpan.end()

        // With per-token inputs, the next step's gather needs THIS step's sampled token on
        // the CPU — wait for the sampler completion here. This serializes the GPU pipeline
        // (the win over a hand-rolled loop is the on-GPU argmax + on-device KV, not depth).
        if !perTokenInputs.isEmpty {
            lastSampledToken = await sampledTokenRendezvous.take()
        }
    }

    // MARK: - Token Generation

    private mutating func generateTokenBatch(
        count: Int,
        gpuSampler: any MPSGraphSampler,
        yieldingTo continuation: AsyncThrowingStream<InferenceEngine.TokenId, Error>.Continuation,
        isCancelled: borrowing Atomic<Bool>
    ) async throws {
        for _ in 0..<count {
            guard !isCancelled.load(ordering: .relaxed) else { return }
            try await _encodeNextStepGPU(
                tokens: [],
                gpuSampler: gpuSampler,
                yieldingTo: continuation
            )
        }
    }

    // MARK: - KV Cache Growth

    private mutating func growKVCacheAndRebind(neededCapacity: Int) async throws {
        let cacheSpan = InstrumentsProfiler.beginCacheManagement(
            step: processedTokenCount, operation: "grow", engine: "CoreAI-Pipelined")

        do {
            do {
                let queue = pipelineQueue
                guard let cmdBuf = queue.makeCommandBuffer() else {
                    throw KVCacheError.allocationFailed(0)
                }

                if (try kvCache.encodePipelinedExpansion(
                    forContextLength: neededCapacity,
                    commandBuffer: cmdBuf)) != nil
                {
                    CLILogger.log("KV cache grew (pipelined) to \(kvCache.currentCapacity)")
                } else {
                    throw KVCacheError.capacityExceeded(
                        needed: neededCapacity, available: kvCache.currentCapacity)
                }
            }
        } catch {
            cacheSpan.end()
            throw error
        }
        cacheSpan.end()
    }

    // MARK: - Run Completion

    mutating func runCompletion(
        prompt: [InferenceEngine.TokenId],
        sampler: SamplingConfiguration,
        maxTokens: Int?,
        yieldingTo continuation: AsyncThrowingStream<InferenceEngine.TokenId, Error>.Continuation
    ) async throws {
        let gpuSampler = try getOrCreateSampler(for: sampler)

        let isCancelled = Atomic<Bool>(false)
        continuation.onTermination = { _ in
            isCancelled.store(true, ordering: .relaxed)
        }

        let contextLeftAfterPrompt = config.maxContextLength - processedTokenCount - prompt.count
        guard contextLeftAfterPrompt >= 1 else {
            throw InferenceRuntimeError.contextLengthExceeded(
                processedTokenCount, config.maxContextLength)
        }
        var totalMaxTokens = min(maxTokens ?? Int.max, contextLeftAfterPrompt)

        // iOS + dynamically-sized KV cache: the on-device compiler miscompiles this
        // graph class once the bound KV state's seq dim reaches 2048 — output is
        // corrupt from the first token (capacity <=1024 is correct; macOS is correct
        // at every size; a cache evict does not help). Cap the pre-grow target at
        // 1024 until the compiler fix lands, and fail loudly when even that budget
        // is exhausted. Repro + bisect: apple/coreai-models#124.
        #if os(iOS)
        if kvCache is GrowingKVCache {
            let iosDynamicKVCapacityCap = 1024
            let budgetLeft = iosDynamicKVCapacityCap - processedTokenCount - prompt.count
            guard budgetLeft >= 1 else {
                throw InferenceRuntimeError.contextLengthExceeded(
                    processedTokenCount, iosDynamicKVCapacityCap)
            }
            if totalMaxTokens > budgetLeft {
                CLILogger.log(
                    "iOS dynamic-KV guard: capping maxTokens \(totalMaxTokens) -> \(budgetLeft) "
                        + "(KV capacity limited to \(iosDynamicKVCapacityCap); apple/coreai-models#124)"
                )
                totalMaxTokens = budgetLeft
            }
        }
        #endif

        // Pre-grow KV cache for prompt
        let promptCapacityNeeded = min(
            processedTokenCount + prompt.count + totalMaxTokens, config.maxContextLength)
        if promptCapacityNeeded > kvCache.currentCapacity {
            do {
                let queue = pipelineQueue
                let grew = try kvCache.ensureCapacity(
                    forContextLength: promptCapacityNeeded, queue: queue)
                if grew {
                    CLILogger.log(
                        "KV cache grew to \(kvCache.currentCapacity) for prompt (\(prompt.count) tokens)"
                    )
                }
            }
        }

        // Split prompt into chunks when it exceeds the chunk threshold. A bundle
        // with a static-chunk "prefill" function always chunks (its main graph
        // is S=1-static, so multi-token prompts can't ride a single encode).
        // Skip prefill entirely if prompt is empty (prefix-cached continuation).
        if !prompt.isEmpty {
            let prefillTokens: ArraySlice<Int32>
            if prompt.count > config.chunkThreshold || (prefillFunction != nil && prompt.count > 1) {
                prefillTokens = try await processChunkedInput(tokens: prompt)
            } else {
                let prefillCapacity = max(1, prompt.count)
                if try logits.ensureCapacity(forContextLength: prefillCapacity) {
                    let fmt = ByteCountFormatter()
                    fmt.countStyle = .memory
                    CLILogger.log(
                        "Logits buffer grew to capacity \(logits.currentCapacity) (\(fmt.string(fromByteCount: Int64(logits.currentByteCount))))"
                    )
                }
                prefillTokens = prompt[...]
            }

            // Process prompt with sampling
            try await _encodeNextStepGPU(
                tokens: prefillTokens,
                gpuSampler: gpuSampler,
                yieldingTo: continuation
            )
        }

        // Generate-Grow-Continue loop
        var remainingTokens = totalMaxTokens - 1

        while remainingTokens > 0 {
            guard !isCancelled.load(ordering: .relaxed) else { break }

            let availableSlots = kvCache.currentCapacity - processedTokenCount
            let tokensThisRound = min(remainingTokens, availableSlots)

            if tokensThisRound > 0 {
                try await generateTokenBatch(
                    count: tokensThisRound,
                    gpuSampler: gpuSampler,
                    yieldingTo: continuation,
                    isCancelled: isCancelled
                )
                remainingTokens -= tokensThisRound
            }

            if remainingTokens > 0 {
                let neededCapacity = processedTokenCount + remainingTokens
                try await growKVCacheAndRebind(neededCapacity: neededCapacity)
            }
        }

        // Sentinel: submit an empty command buffer on the same serial queue.
        // Its addCompletedHandler fires after all real sampler callbacks (serial
        // queue FIFO ordering via MTLDispatchListApply), guaranteeing every
        // continuation.yield has returned before the caller calls finish().
        // We use a bare command buffer instead of the sampler to avoid the shared
        // MPSGraphExecutableExecutionDescriptor issue in MPSGraphCompositeSampler.
        await withCheckedContinuation { (sentinelCont: CheckedContinuation<Void, Never>) in
            do {
                let queue = pipelineQueue
                guard let cmdBuf = queue.makeCommandBuffer() else {
                    sentinelCont.resume()
                    return
                }
                cmdBuf.addCompletedHandler { _ in sentinelCont.resume() }
                cmdBuf.commit()
            }
        }
    }

    // MARK: - Chunked Prefill

    mutating func processChunkedInput(tokens: [Int32]) async throws -> ArraySlice<Int32> {
        var remainingTokens = tokens[...]

        // Static-chunk prefill function: full S=C chunks ride the "prefill"
        // graph (its own static logits buffer), the tail runs as S=1 steps on
        // the main graph, and exactly one token is left for the sampled step.
        if prefillFunction != nil {
            let chunkLength = prefillChunkLength
            while remainingTokens.count - 1 >= chunkLength {
                try await _encodeChunk(tokens: Array(remainingTokens.prefix(chunkLength)))
                remainingTokens = remainingTokens.dropFirst(chunkLength)
            }
            while remainingTokens.count > 1 {
                try await _encodeChunk(tokens: [remainingTokens.first!])
                remainingTokens = remainingTokens.dropFirst()
            }
            return remainingTokens
        }

        let chunkSize = config.prefillChunkSize

        try logits.ensureCapacity(forContextLength: chunkSize)

        while remainingTokens.count > chunkSize {
            let chunk = Array(remainingTokens.prefix(chunkSize))
            try await _encodeChunk(tokens: chunk)
            remainingTokens = remainingTokens.dropFirst(chunkSize)
        }

        return remainingTokens
    }

    private mutating func _encodeChunk(tokens: [Int32]) async throws {
        let queryLength = tokens.count
        let currentStep = processedTokenCount
        if !perTokenInputs.isEmpty {
            try requireSingleTokenStep(queryLength)
        }

        // Full-length chunks ride the static-chunk "prefill" function when present;
        // everything else (S=1 tail steps, plain dynamic graphs) uses the main one.
        let usePrefillFunction = prefillFunction != nil && queryLength == prefillChunkLength
            && queryLength > 1
        let encodeFunction = usePrefillFunction ? prefillFunction! : function
        let idsDesc = usePrefillFunction ? prefillInputIdsBaseDesc! : inputIdsBaseDesc
        let posDesc = usePrefillFunction ? prefillPositionIdsBaseDesc! : positionIdsBaseDesc
        let logitsDesc = usePrefillFunction ? prefillLogitsBaseDesc! : logitsBaseDesc
        let logitsTargetBuffer = usePrefillFunction ? prefillLogitsBuffer! : logits.metalBuffer

        let chunkID = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIPipelinedChunk",
            details: "step=\(currentStep) qLen=\(queryLength)"
        )

        // Write at the chunk's natural position so each chunk occupies a disjoint
        // region of inputTokensBuffer. Encode holds a live MTLBuffer reference — writing
        // all chunks at offset 0 would race with the GPU reading the previous chunk.
        let ptr = inputTokensBuffer.contents().bindMemory(
            to: Int32.self, capacity: processedTokenCount + queryLength)
        for (i, token) in tokens.enumerated() {
            ptr[processedTokenCount + i] = token
        }

        let cachePosBuffer = cachePositionBuffers[step % pipelineDepth]
        let posLength = processedTokenCount + queryLength

        // Build async values and encode
        let tokenShape = [1, queryLength]
        let tokenStrides = try resolvedStrides(descriptor: idsDesc, shape: tokenShape)
        let posShape = [1, posLength]
        let posStrides = try resolvedStrides(descriptor: posDesc, shape: posShape)

        let tokenValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: inputTokensBuffer,
            byteOffset: processedTokenCount * MemoryLayout<Int32>.size,
            scalarType: .int32, shape: tokenShape, strides: tokenStrides)
        let posValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: cachePosBuffer, byteOffset: 0,
            scalarType: .int32, shape: posShape, strides: posStrides)

        var asyncInputs: [String: InferenceFunction.AsyncValue] = [
            inputIdsName: tokenValue, positionIdsName: posValue,
        ]
        if !perTokenInputs.isEmpty, let chunkToken = tokens.first {
            try bindPerTokenInputs(token: chunkToken, position: currentStep, into: &asyncInputs)
        }
        // Static inputs: same caller-owned buffer every step, nothing to fill.
        for staticInput in staticInputs {
            asyncInputs[staticInput.name] = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: staticInput.buffer,
                byteOffset: 0,
                scalarType: staticInput.scalarType,
                shape: staticInput.shape,
                strides: staticInput.strides
            )
        }

        let keyBuffer = kvCache.keyBinding.metalBuffer
        let keyShape = kvCache.keyBinding.layout.shape
        let keyStrides = kvCache.keyBinding.layout.strides
        let valBuffer = kvCache.valueBinding.metalBuffer
        let valShape = kvCache.valueBinding.layout.shape
        let valStrides = kvCache.valueBinding.layout.strides
        var keyState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: keyBuffer, byteOffset: 0,
            scalarType: keyCacheScalarType, shape: keyShape, strides: keyStrides)
        var valState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: valBuffer, byteOffset: 0,
            scalarType: valueCacheScalarType, shape: valShape, strides: valStrides)
        let logitsShape = [1, queryLength, vocabSize]
        let logitsStrides = try resolvedStrides(descriptor: logitsDesc, shape: logitsShape)

        try encodeWithStates(
            function: encodeFunction, inputs: asyncInputs,
            keyState: &keyState, keyCacheName: keyCacheName,
            valState: &valState, valueCacheName: valueCacheName,
            additionalStates: additionalStates,
            logitsBuffer: logitsTargetBuffer, logitsName: logitsOutputName,
            logitsShape: logitsShape, logitsStrides: logitsStrides,
            computeStream: computeStream)

        processedTokenCount += queryLength
        step += 1
        InstrumentsProfiler.endCustomInterval(name: "CoreAIPipelinedChunk", signpostID: chunkID)
    }

    mutating func reset() {
        let span = InstrumentsProfiler.beginReset(engine: "CoreAI-Pipelined")
        processedTokenCount = 0
        step = 0
        cachedSampler = nil
        cachedSamplerTemperature = nil
        lastSampledToken = nil
        // Zero SSM states so the next conversation starts from a clean slate.
        // Per-token input slots need no clearing: each step's slot is fully
        // rewritten by the provider before it is bound.
        additionalStates?.reset()
        span.end()
    }

    // MARK: - Warmup

    mutating func performWarmup(queryLength: Int, samplingConfig: SamplingConfiguration?) async throws {
        let warmupStart = ContinuousClock.now
        let warmupSpan = InstrumentsProfiler.beginWarmup()

        // A single warmup at any shape primes the framework's internal caches
        // (reshape, kernel compilation, state pool). Benchmarks show no benefit
        // from warming every bucket shape — the jump from none→any is what matters.
        let defaultWarmupLength = 256

        var shapesToWarm: [Int]
        if queryLength > 0 {
            shapesToWarm = [queryLength]
        } else {
            shapesToWarm = [1, defaultWarmupLength]
        }
        if !perTokenInputs.isEmpty {
            // Per-token-input graphs are S=1 static — larger warmup shapes would be rejected.
            shapesToWarm = [1]
        }

        CLILogger.log("Running warmup for \(shapesToWarm.count) shape(s)")

        let maxShape = shapesToWarm.last ?? 1
        try logits.ensureCapacity(forContextLength: maxShape)

        do {
            let queue = pipelineQueue
            if try kvCache.ensureCapacity(forContextLength: maxShape, queue: queue) {
                CLILogger.log("KV cache grew to \(kvCache.currentCapacity) for warmup")
            }
        }

        let warmupSampler = try MPSGraphSamplerFactory.makeSampler(
            device: device,
            vocabSize: config.vocabSize,
            temperature: samplingConfig?.temperature ?? 0
        )

        for shape in shapesToWarm {
            // Write dummy tokens
            let ptr = inputTokensBuffer.contents().bindMemory(to: Int32.self, capacity: shape)
            for i in 0..<shape { ptr[i] = 1 }

            let cachePosBuffer = cachePositionBuffers[step % pipelineDepth]
            let posLength = processedTokenCount + shape

            let tShape = [1, shape]
            let tStrides = try resolvedStrides(descriptor: inputIdsBaseDesc, shape: tShape)
            let pShape = [1, posLength]
            let pStrides = try resolvedStrides(descriptor: positionIdsBaseDesc, shape: pShape)

            let tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: inputTokensBuffer, byteOffset: 0,
                scalarType: .int32, shape: tShape, strides: tStrides)
            let posValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: cachePosBuffer, byteOffset: 0,
                scalarType: .int32, shape: pShape, strides: pStrides)
            var asyncInputs: [String: InferenceFunction.AsyncValue] = [
                inputIdsName: tokenValue, positionIdsName: posValue,
            ]
            if !perTokenInputs.isEmpty {
                // Warm with the same dummy token the ids buffer holds, at position 0.
                try bindPerTokenInputs(token: 1, position: 0, into: &asyncInputs)
            }
            // Static inputs: same caller-owned buffer every step, nothing to fill.
            for staticInput in staticInputs {
                asyncInputs[staticInput.name] = unsafe InferenceFunction.AsyncValue(
                    unsafeBuffer: staticInput.buffer,
                    byteOffset: 0,
                    scalarType: staticInput.scalarType,
                    shape: staticInput.shape,
                    strides: staticInput.strides
                )
            }

            let keyBuffer = kvCache.keyBinding.metalBuffer
            let kShape = kvCache.keyBinding.layout.shape
            let kStrides = kvCache.keyBinding.layout.strides
            let valBuffer = kvCache.valueBinding.metalBuffer
            let vShape = kvCache.valueBinding.layout.shape
            let vStrides = kvCache.valueBinding.layout.strides
            var keyState = unsafe InferenceFunction.AsyncMutableValue(
                unsafeBuffer: keyBuffer, byteOffset: 0,
                scalarType: keyCacheScalarType, shape: kShape, strides: kStrides)
            var valState = unsafe InferenceFunction.AsyncMutableValue(
                unsafeBuffer: valBuffer, byteOffset: 0,
                scalarType: valueCacheScalarType, shape: vShape, strides: vStrides)
            let lShape = [1, shape, vocabSize]
            let lStrides = try resolvedStrides(descriptor: logitsBaseDesc, shape: lShape)

            try encodeWithStates(
                function: function, inputs: asyncInputs,
                keyState: &keyState, keyCacheName: keyCacheName,
                valState: &valState, valueCacheName: valueCacheName,
                additionalStates: additionalStates,
                logitsBuffer: logits.metalBuffer, logitsName: logitsOutputName,
                logitsShape: lShape, logitsStrides: lStrides,
                computeStream: computeStream)

            // Warm up argmax kernel using pipeline-matched decode buffers
            let warmupLogitsBuffer = decodeLogitsBuffers[step % pipelineDepth]
            let warmupOutputBuffer = decodeOutputBuffers[step % pipelineDepth]
            let logitsOffset = (shape - 1) * vocabSize * MemoryLayout<UInt16>.size

            do {
                let queue = pipelineQueue
                warmupSampler.encode(
                    to: queue,
                    logitsBuffer: warmupLogitsBuffer,
                    logitsOffset: logitsOffset,
                    outputBuffer: warmupOutputBuffer,
                    outputOffset: 0,
                    completion: { _ in }
                )
            }

            step += 1
        }

        await computeStream.currentWorkCompleted()
        reset()

        warmupSpan.end()
        let warmupElapsed = milliseconds(since: warmupStart)
        CLILogger.log(
            "CoreAI pipelined warmup complete (\(shapesToWarm.count) shapes): \(String(format: "%.2f", warmupElapsed))ms"
        )
    }
}

extension CoreAIPipelinedEngine {
    /// Async sequence of `InferenceOutput` produced by `generate()`.
    ///
    /// Unlike the CPU engines, the pipelined engine samples on-device and drives
    /// output from a producer `Task`, so this sequence forwards an underlying
    /// `AsyncThrowingStream`. The producer records the `stopReason` directly.
    public struct GenerationSequence: InferenceOutputSequence {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        let base: AsyncThrowingStream<InferenceOutput, any Error>
        let stopReasonStore: StopReasonStore

        public var stopReason: StopReason? { stopReasonStore.stopReason }

        public func setStopReason(_ reason: StopReason) {
            stopReasonStore.set(reason)
        }

        public func makeAsyncIterator() -> Iterator {
            Iterator(base: base.makeAsyncIterator(), stopReasonStore: stopReasonStore)
        }
    }
}

extension CoreAIPipelinedEngine.GenerationSequence {
    public struct Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        var base: AsyncThrowingStream<InferenceOutput, any Error>.AsyncIterator
        let stopReasonStore: StopReasonStore

        public mutating func next() async throws -> InferenceOutput? {
            do {
                return try await base.next()
            } catch is CancellationError {
                // The producer Task is independent and won't observe the
                // consumer's cancellation, so record it from the consumer side.
                stopReasonStore.set(.cancelled)
                throw CancellationError()
            } catch {
                stopReasonStore.set(.error)
                throw error
            }
        }
    }
}
