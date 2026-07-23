// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// DiffusionGemma discrete **entropy-renoise** sampler + stopping — the genuinely-new
// diffusion surface. The Gemma4-MoE backbone is reused unchanged; only these post-logits
// ops drive the denoise canvas. Faithful Swift port of `_diffgemma_sampler.py` (itself
// verified BIT-EXACT 9/9 vs transformers' `generation_diffusion_gemma.py`).
//
// All ops are batch-1 (the block-AR driver denoises one canvas at a time) over flat
// `[Float]`/`[Int32]` arrays, matching this package's scheduler idiom. Logits are laid out
// `[canvasLength * vocabSize]` row-major (position-major).

/// Linear temperature schedule: `t = tMin + (tMax - tMin) * (curStep / N)`.
/// The driver iterates `curStep = N ... 1` (reverse), so temperature runs tMax -> ~tMin.
public func linearTemperature(curStep: Int, tMin: Float, tMax: Float, maxDenoisingSteps: Int) -> Float {
    tMin + (tMax - tMin) * (Float(curStep) / Float(maxDenoisingSteps))
}

/// Per-position Shannon entropy of a categorical distribution given **logits**, matching
/// `torch.distributions.Categorical(logits=).entropy()` (numerically-stable log-softmax form:
/// `H = -Σ p·logp` where `logp = logit - logsumexp`). `logits` is `[count * vocab]` row-major;
/// returns `[count]`.
public func categoricalEntropy(logits: [Float], count: Int, vocab: Int) -> [Float] {
    var out = [Float](repeating: 0, count: count)
    logits.withUnsafeBufferPointer { buf in
        for p in 0..<count {
            let base = p * vocab
            var maxv = -Float.greatestFiniteMagnitude
            for v in 0..<vocab where buf[base + v] > maxv { maxv = buf[base + v] }
            var sumExp: Float = 0
            for v in 0..<vocab { sumExp += Foundation.exp(buf[base + v] - maxv) }
            let lse = maxv + Foundation.log(sumExp)
            var ent: Float = 0
            for v in 0..<vocab {
                let logp = buf[base + v] - lse
                ent -= Foundation.exp(logp) * logp
            }
            out[p] = ent
        }
    }
    return out
}

/// Initializes a canvas with uniform-vocab tokens, accepts the lowest-entropy prefix of
/// positions (sum of all strictly-lower-entropy positions <= `entropyBound`), renoises the rest.
public struct EntropyBoundSampler {
    public let entropyBound: Float
    public let canvasLength: Int
    public let vocabSize: Int
    public let maxDenoisingSteps: Int

    /// Set by `acceptCanvas`, consumed by `renoiseCanvas`. `true` = position was accepted.
    public private(set) var acceptedTokenMask: [Bool]

    public init(entropyBound: Float, canvasLength: Int, vocabSize: Int, maxDenoisingSteps: Int) {
        self.entropyBound = entropyBound
        self.canvasLength = canvasLength
        self.vocabSize = vocabSize
        self.maxDenoisingSteps = maxDenoisingSteps
        self.acceptedTokenMask = [Bool](repeating: false, count: canvasLength)
    }

    /// Uniform-vocab random canvas, `[canvasLength]`. The RNG is injected (the python sampler
    /// draws `torch.randint` internally; the driver/tests wire an explicit source) so generation
    /// is reproducible and the renoise step is unit-testable.
    public func initializeCanvas(using rng: inout some RandomNumberGenerator) -> [Int32] {
        (0..<canvasLength).map { _ in Int32(Int.random(in: 0..<vocabSize, using: &rng)) }
    }

    /// Accept the lowest-entropy prefix: sort positions by entropy ascending; accept position `i`
    /// iff the sum of all STRICTLY-lower-entropy positions <= `entropyBound`; scatter back to a
    /// per-position mask; return `where(mask, denoiser, current)`. Mutates `acceptedTokenMask`.
    public mutating func acceptCanvas(current: [Int32], denoiser: [Int32], logits: [Float], curStep: Int) -> [Int32] {
        let entropy = categoricalEntropy(logits: logits, count: canvasLength, vocab: vocabSize)
        // Ascending sort of position indices by entropy; ties keep input order (deterministic).
        let order = (0..<canvasLength).sorted { entropy[$0] != entropy[$1] ? entropy[$0] < entropy[$1] : $0 < $1 }
        var mask = [Bool](repeating: false, count: canvasLength)
        var precedingSum: Float = 0  // = cumulative_entropy - this_entropy (strict prefix)
        for pos in order {
            mask[pos] = precedingSum <= entropyBound
            precedingSum += entropy[pos]
        }
        acceptedTokenMask = mask
        return (0..<canvasLength).map { mask[$0] ? denoiser[$0] : current[$0] }
    }

    /// Replace rejected positions (`!acceptedTokenMask`) with fresh tokens from `randomCanvas`,
    /// keep accepted positions. (`randomCanvas` injected for determinism/testability.)
    public func renoiseCanvas(accepted: [Int32], randomCanvas: [Int32]) -> [Int32] {
        (0..<canvasLength).map { acceptedTokenMask[$0] ? accepted[$0] : randomCanvas[$0] }
    }
}

/// Stops when the greedy (argmax) canvas is unchanged across `stabilityThreshold` steps AND
/// the mean per-position entropy of the processed logits is < `confidenceThreshold`.
public struct StableAndConfidentStopping {
    public let stabilityThreshold: Int
    public let confidenceThreshold: Float
    /// Ring buffer of the last `stabilityThreshold` argmax canvases (nil until the first call,
    /// then seeded with -1 sentinels — matching the python init).
    private var argmaxCanvasHistory: [[Int32]]?

    public init(stabilityThreshold: Int, confidenceThreshold: Float) {
        self.stabilityThreshold = stabilityThreshold
        self.confidenceThreshold = confidenceThreshold
    }

    public mutating func callAsFunction(argmaxCanvas: [Int32], logits: [Float], vocab: Int) -> Bool {
        let stable: Bool
        if stabilityThreshold == 0 {
            stable = true
        } else {
            if argmaxCanvasHistory == nil {
                argmaxCanvasHistory = Array(
                    repeating: [Int32](repeating: -1, count: argmaxCanvas.count),
                    count: stabilityThreshold)
            }
            stable = argmaxCanvasHistory!.allSatisfy { $0 == argmaxCanvas }
            argmaxCanvasHistory!.removeFirst()         // roll(shifts: -1)
            argmaxCanvasHistory!.append(argmaxCanvas)  // history[-1] = argmaxCanvas
        }
        let entropy = categoricalEntropy(logits: logits, count: argmaxCanvas.count, vocab: vocab)
        let meanEntropy = entropy.reduce(0, +) / Float(entropy.count)
        return stable && (meanEntropy < confidenceThreshold)
    }

    public mutating func reset() { argmaxCanvasHistory = nil }
}
