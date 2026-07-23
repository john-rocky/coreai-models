// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// DiffusionGemma block-AR generation driver — Swift port of `_diffgemma_driver.py` (the
// P2b-validated reference). Three phases per `canvasLength`-token block:
//
//   PREFILL : encoder(prompt)                         -> 60 per-layer KV (frozen for the block)
//   DENOISE : for curStep = N..1:                     -> bidirectional canvas refinement
//               logits      = decoder(canvas, soft_embeds, KV)        # softcapped in-graph
//               processed   = logits / temperature(curStep)
//               denoiser    = multinomial(softmax(processed))         # accept/renoise trajectory
//               argmaxCanvas= argmax(processed)                       # this COMMITS
//               canvas      = renoise(accept(canvas, denoiser, processed))
//               soft_embeds = softProj(processed)                     # probs@embed for next step
//               stop if stable & confident
//   COMMIT  : argmaxCanvas -> appended tokens (-> next block's prefill)
//
// ⚠️ SKELETON pending P4 export: runs once the encoder/decoder/soft_proj graphs exist; the
// graph-IO dtype/shape plumbing (TODOs below) is finalized against the exported bundles, and
// the build is verified then (deferred while the P2c driver holds the Mac).

/// A graph function the driver runs (encoder / decoder / soft_proj). Matches
/// `CoreAIDiffusionModelFunction.predict`'s signature so that runner conforms via a one-line
/// extension — kept abstract here so this driver lives in CoreAILanguageModels without
/// depending on the image-diffusion module; the concrete runner is injected at construction.
public protocol DiffusionGraphFunction: Sendable {
    func predict(inputs: [String: NDArray]) async throws -> [String: [Float]]
}

/// Small seedable RNG (SplitMix64) for canvas init / renoise / multinomial. Generation is
/// stochastic by design, so this need not match torch's RNG bit-for-bit — only be reproducible.
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public struct DiffusionLMPipeline {
    public struct Config: Sendable {
        public var canvasLength = 256
        public var maxSteps = 48
        public var tMin: Float = 0.4
        public var tMax: Float = 0.8
        public var entropyBound: Float = 0.1
        public var stabilityThreshold = 1
        public var confidenceThreshold: Float = 0.005
        public var vocabSize: Int
        public var hiddenSize: Int
        public var numLayers: Int
        public var eosIds: Set<Int32> = [1, 106, 50]
        public init(vocabSize: Int, hiddenSize: Int, numLayers: Int) {
            self.vocabSize = vocabSize
            self.hiddenSize = hiddenSize
            self.numLayers = numLayers
        }
    }

    let encoder: any DiffusionGraphFunction
    let decoder: any DiffusionGraphFunction
    /// processed logits [1, CL, vocab] -> soft_embeds [1, CL, hidden] (= softmax·embed; see P4 doc).
    let softProj: any DiffusionGraphFunction
    let config: Config

    public init(encoder: any DiffusionGraphFunction, decoder: any DiffusionGraphFunction,
                softProj: any DiffusionGraphFunction, config: Config) {
        self.encoder = encoder
        self.decoder = decoder
        self.softProj = softProj
        self.config = config
    }

    /// Generate up to `maxNewTokens` tokens from a tokenized prompt (multi-block).
    public func generate(promptIds: [Int32], maxNewTokens: Int, seed: UInt64 = 0) async throws -> [Int32] {
        let CL = config.canvasLength, V = config.vocabSize, H = config.hiddenSize
        var rng = SplitMix64(seed: seed)
        var inputIds = promptIds
        var generated: [Int32] = []

        let nBlocks = (maxNewTokens + CL - 1) / CL
        for _ in 0..<nBlocks {
            // PREFILL — encoder(prompt) -> per-layer KV (flat [Float] outputs, kept for the block).
            let promptLen = inputIds.count
            let encOut = try await encoder.predict(inputs: [
                "input_ids": ndInt32(inputIds, shape: [1, promptLen]),
                "position_ids": ndInt32((0..<Int32(promptLen)).map { $0 }, shape: [1, promptLen]),
            ])
            let kvInputs = try kvNDArrays(from: encOut, promptLen: promptLen)

            // DENOISE
            var sampler = EntropyBoundSampler(
                entropyBound: config.entropyBound, canvasLength: CL, vocabSize: V, maxDenoisingSteps: config.maxSteps)
            var stopping = StableAndConfidentStopping(
                stabilityThreshold: config.stabilityThreshold, confidenceThreshold: config.confidenceThreshold)
            var canvas = sampler.initializeCanvas(using: &rng)
            var argmaxCanvas = canvas
            var softEmbeds = [Float](repeating: 0, count: CL * H)  // step 0 = zeros (= None branch)
            let decPos = (Int32(promptLen)..<Int32(promptLen + CL)).map { $0 }

            for curStep in stride(from: config.maxSteps, through: 1, by: -1) {
                var decInputs = kvInputs
                decInputs["canvas_ids"] = ndInt32(canvas, shape: [1, CL])
                decInputs["position_ids"] = ndInt32(decPos, shape: [1, CL])
                decInputs["soft_embeds"] = ndFloat(softEmbeds, shape: [1, CL, H])
                guard let raw = try await decoder.predict(inputs: decInputs)["logits"] else {
                    throw DiffusionLMError.missingOutput("logits")
                }  // [CL*V], softcapped in-graph
                let temp = linearTemperature(
                    curStep: curStep, tMin: config.tMin, tMax: config.tMax, maxDenoisingSteps: config.maxSteps)
                let processed = raw.map { $0 / temp }

                let denoiser = multinomialRows(processed, rows: CL, vocab: V, using: &rng)
                argmaxCanvas = argmaxRows(processed, rows: CL, vocab: V)
                let accepted = sampler.acceptCanvas(
                    current: canvas, denoiser: denoiser, logits: processed, curStep: curStep)
                canvas = sampler.renoiseCanvas(accepted: accepted, randomCanvas: sampler.initializeCanvas(using: &rng))

                // soft_embeds for the NEXT step = softmax(processed) @ embed (heavy matmul -> graph).
                guard let soft = try await softProj.predict(
                    inputs: ["processed_logits": ndFloat(processed, shape: [1, CL, V])])["soft_embeds"] else {
                    throw DiffusionLMError.missingOutput("soft_embeds")
                }
                softEmbeds = soft

                if stopping(argmaxCanvas: argmaxCanvas, logits: processed, vocab: V) { break }
            }

            // COMMIT — emit argmax canvas; stop the block at the first EOS.
            inputIds += argmaxCanvas
            if let eos = argmaxCanvas.firstIndex(where: { config.eosIds.contains($0) }) {
                generated += argmaxCanvas[..<eos]
                break
            }
            generated += argmaxCanvas
        }
        return generated
    }

    // MARK: - host-side ops (the heavy matmuls are graphs; these are O(CL*vocab) glue)

    /// argmax over vocab for each of `rows` positions. `logits` is `[rows*vocab]` row-major.
    func argmaxRows(_ logits: [Float], rows: Int, vocab: Int) -> [Int32] {
        var out = [Int32](repeating: 0, count: rows)
        logits.withUnsafeBufferPointer { buf in
            for r in 0..<rows {
                let base = r * vocab
                var best = 0
                var bestV = buf[base]
                for v in 1..<vocab where buf[base + v] > bestV { bestV = buf[base + v]; best = v }
                out[r] = Int32(best)
            }
        }
        return out
    }

    /// Per-row multinomial sample of `softmax(logits)` (stable softmax + inverse-CDF).
    func multinomialRows(_ logits: [Float], rows: Int, vocab: Int, using rng: inout some RandomNumberGenerator) -> [Int32] {
        var out = [Int32](repeating: 0, count: rows)
        logits.withUnsafeBufferPointer { buf in
            for r in 0..<rows {
                let base = r * vocab
                var maxv = -Float.greatestFiniteMagnitude
                for v in 0..<vocab where buf[base + v] > maxv { maxv = buf[base + v] }
                var sum: Float = 0
                for v in 0..<vocab { sum += Foundation.exp(buf[base + v] - maxv) }
                let target = Float.random(in: 0..<1, using: &rng) * sum
                var acc: Float = 0
                var pick = vocab - 1
                for v in 0..<vocab {
                    acc += Foundation.exp(buf[base + v] - maxv)
                    if acc >= target { pick = v; break }
                }
                out[r] = Int32(pick)
            }
        }
        return out
    }

    // MARK: - NDArray plumbing  (TODO(P4): match dtype to each graph's inputDescriptors)

    private func ndInt32(_ values: [Int32], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .int32)
        var view = a.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<values.count { ptr[i] = values[i] }
        }
        return a
    }

    private func ndFloat(_ values: [Float], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .float32)  // TODO(P4): .float16 if the graph wants it
        var view = a.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<values.count { ptr[i] = values[i] }
        }
        return a
    }

    /// Rebuild the decoder's 60 KV inputs from the encoder's flat [Float] outputs. Per-layer shapes
    /// are HETEROGENEOUS: full layers head_dim 512 / 2 KV heads, sliding 256 / 8 (see P4 doc).
    /// TODO(P4): pass the per-layer (nKV, headDim) from config (encoder output_names = enc_k_i/enc_v_i).
    private func kvNDArrays(from encOut: [String: [Float]], promptLen: Int) throws -> [String: NDArray] {
        var kv: [String: NDArray] = [:]
        for i in 0..<config.numLayers {
            for name in ["enc_k_\(i)", "enc_v_\(i)"] {
                guard let flat = encOut[name] else { throw DiffusionLMError.missingOutput(name) }
                // TODO(P4): real shape [1, nKV_of(i), promptLen, headDim_of(i)] — placeholder infers nothing.
                kv[name] = ndFloat(flat, shape: [flat.count])
            }
        }
        return kv
    }
}

public enum DiffusionLMError: Error, LocalizedError {
    case missingOutput(String)
    public var errorDescription: String? {
        switch self {
        case .missingOutput(let n): return "Diffusion graph did not return expected output '\(n)'"
        }
    }
}
