// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// P5 generation: the DiffusionGemma entropy-renoise sampler + multinomial, inlined here
// (pure [Float]/[Int32], no CoreAILanguageModels dependency so the gate tool stays light).
// Faithful copy of CoreAILanguageModels/Diffusion/DiffusionSampler.swift (bit-exact 9/9 vs
// transformers' generation_diffusion_gemma.py) + the driver's multinomial/argmax step.

import Accelerate
import Foundation

/// HOST soft_proj: soft = scale * softmax(processed[count,vocab]) @ embed[vocab,hidden].
/// Done on the CPU (Accelerate BLAS) instead of the engine: the 262144-wide softmax@embed graph,
/// run on the GPU after a decode, triggers MTL4CommandQueueError that corrupts the next decode
/// (it's the same giant op that was the torch-MPS root cause). The host computes it cleanly.
func hostSoftProj(processed: [Float], embed: [Float], count: Int, vocab: Int, hidden: Int, scale: Float) -> [Float] {
    var probs = [Float](repeating: 0, count: count * vocab)
    processed.withUnsafeBufferPointer { pb in
        probs.withUnsafeMutableBufferPointer { qb in
            for p in 0..<count {
                let base = p * vocab
                var mx = -Float.greatestFiniteMagnitude
                for v in 0..<vocab where pb[base + v] > mx { mx = pb[base + v] }
                var s: Float = 0
                for v in 0..<vocab { let e = Foundation.exp(pb[base + v] - mx); qb[base + v] = e; s += e }
                let inv = 1.0 / s
                for v in 0..<vocab { qb[base + v] *= inv }
            }
        }
    }
    var soft = [Float](repeating: 0, count: count * hidden)
    probs.withUnsafeBufferPointer { a in embed.withUnsafeBufferPointer { b in soft.withUnsafeMutableBufferPointer { c in
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(count), Int32(hidden), Int32(vocab),
                    scale, a.baseAddress, Int32(vocab), b.baseAddress, Int32(hidden),
                    0.0, c.baseAddress, Int32(hidden))
    }}}
    return soft
}

/// t = tMin + (tMax - tMin) * (curStep / N). Driver iterates curStep = N..1 => tMax -> ~tMin.
func linearTemperature(curStep: Int, tMin: Float, tMax: Float, steps: Int) -> Float {
    tMin + (tMax - tMin) * (Float(curStep) / Float(steps))
}

/// Per-position Shannon entropy from logits [count*vocab] row-major -> [count]
/// (matches torch.distributions.Categorical(logits=).entropy()).
func categoricalEntropy(_ logits: [Float], count: Int, vocab: Int) -> [Float] {
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
            for v in 0..<vocab { let lp = buf[base + v] - lse; ent -= Foundation.exp(lp) * lp }
            out[p] = ent
        }
    }
    return out
}

/// Per position: multinomial draw from softmax(logits) AND argmax. One pass over vocab each.
/// `processed` is [count*vocab] row-major. Returns (denoiser[count], argmax[count]).
func multinomialAndArgmax(_ processed: [Float], count: Int, vocab: Int,
                          _ rng: inout some RandomNumberGenerator) -> (denoiser: [Int32], argmax: [Int32]) {
    var denoiser = [Int32](repeating: 0, count: count)
    var argmax = [Int32](repeating: 0, count: count)
    // draw all uniforms up front (host RNG; generation is stochastic, exact draws need not match torch)
    let us = (0..<count).map { _ in Float(Double.random(in: 0..<1, using: &rng)) }
    processed.withUnsafeBufferPointer { buf in
        for p in 0..<count {
            let base = p * vocab
            var maxv = -Float.greatestFiniteMagnitude
            var amax = 0
            for v in 0..<vocab where buf[base + v] > maxv { maxv = buf[base + v]; amax = v }
            var sumExp: Float = 0
            for v in 0..<vocab { sumExp += Foundation.exp(buf[base + v] - maxv) }
            // inverse-CDF sample in softmax space (probs = exp(logit-max)/sumExp)
            let target = us[p] * sumExp
            var acc: Float = 0
            var pick = vocab - 1
            for v in 0..<vocab {
                acc += Foundation.exp(buf[base + v] - maxv)
                if acc >= target { pick = v; break }
            }
            denoiser[p] = Int32(pick)
            argmax[p] = Int32(amax)
        }
    }
    return (denoiser, argmax)
}

/// FUSED per-step host work: ONE exp pass over the [count*vocab] softcapped logits yields per-position
/// entropy, argmax, a multinomial draw, AND the self-conditioning soft-embeds — collapsing the THREE
/// separate softmax recomputations of categoricalEntropy + multinomialAndArgmax + hostSoftProj (the
/// per-step bottleneck once the grouped-MoE kernel made the GPU forward cheap). Numerically equivalent:
///   entropy = lse - sum(p*z)            (the categorical-entropy identity; p=softmax(z), z=logits/temp)
///   denoiser = inverse-CDF multinomial  (same per-position uniform draw / RNG order as multinomialAndArgmax)
///   soft     = embedScale * softmax(z) @ embed   (Accelerate sgemm, reusing the already-exp'd probs)
func fusedHostStep(raw: [Float], temp: Float, count: Int, vocab: Int, hidden: Int,
                   embed: [Float], embedScale: Float, soft0: Bool,
                   _ rng: inout some RandomNumberGenerator)
    -> (entropy: [Float], meanEnt: Float, argmax: [Int32], denoiser: [Int32], soft: [Float]) {
    let invTemp = 1.0 / temp
    var probs = [Float](repeating: 0, count: count * vocab)   // exp(z-max); normalized in-place for the sgemm
    var entropy = [Float](repeating: 0, count: count)
    var argmax = [Int32](repeating: 0, count: count)
    var denoiser = [Int32](repeating: 0, count: count)
    // Draw the per-position uniforms up front (same count + RNG order as multinomialAndArgmax).
    let us = (0..<count).map { _ in Float(Double.random(in: 0..<1, using: &rng)) }
    raw.withUnsafeBufferPointer { rb in probs.withUnsafeMutableBufferPointer { pb in
        for p in 0..<count {
            let base = p * vocab
            var mx = -Float.greatestFiniteMagnitude
            var amax = 0
            for v in 0..<vocab { let z = rb[base + v] * invTemp; if z > mx { mx = z; amax = v } }
            var sumExp: Float = 0
            var wz: Float = 0
            for v in 0..<vocab {
                let z = rb[base + v] * invTemp
                let e = Foundation.exp(z - mx)
                pb[base + v] = e
                sumExp += e
                wz += e * z
            }
            let invSum = 1.0 / sumExp
            entropy[p] = (mx + Foundation.log(sumExp)) - wz * invSum   // = lse - sum(p*z)
            argmax[p] = Int32(amax)
            let target = us[p] * sumExp                                // inverse-CDF in unnormalized exp space
            var acc: Float = 0
            var pick = vocab - 1
            for v in 0..<vocab { acc += pb[base + v]; if acc >= target { pick = v; break } }
            denoiser[p] = Int32(pick)
            if !soft0 { for v in 0..<vocab { pb[base + v] *= invSum } }  // normalize for the soft sgemm
        }
    }}
    let meanEnt = entropy.reduce(0, +) / Float(count)
    var soft: [Float] = []
    if !soft0 {
        soft = [Float](repeating: 0, count: count * hidden)
        probs.withUnsafeBufferPointer { a in embed.withUnsafeBufferPointer { b in soft.withUnsafeMutableBufferPointer { c in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(count), Int32(hidden), Int32(vocab),
                        embedScale, a.baseAddress, Int32(vocab), b.baseAddress, Int32(hidden),
                        0.0, c.baseAddress, Int32(hidden))
        }}}
    }
    return (entropy, meanEnt, argmax, denoiser, soft)
}

/// Entropy-bound accept + uniform renoise (mutating mask between the two calls).
struct EntropyBoundSampler {
    let entropyBound: Float
    let canvasLength: Int
    let vocabSize: Int
    private(set) var acceptedTokenMask: [Bool]

    init(entropyBound: Float, canvasLength: Int, vocabSize: Int) {
        self.entropyBound = entropyBound
        self.canvasLength = canvasLength
        self.vocabSize = vocabSize
        self.acceptedTokenMask = [Bool](repeating: false, count: canvasLength)
    }

    func initializeCanvas(_ rng: inout some RandomNumberGenerator) -> [Int32] {
        (0..<canvasLength).map { _ in Int32(Int.random(in: 0..<vocabSize, using: &rng)) }
    }

    /// Accept the lowest-entropy prefix (sum of STRICTLY-lower-entropy positions <= bound).
    mutating func acceptCanvas(current: [Int32], denoiser: [Int32], logits: [Float], entropy: [Float]) -> [Int32] {
        let order = (0..<canvasLength).sorted { entropy[$0] != entropy[$1] ? entropy[$0] < entropy[$1] : $0 < $1 }
        var mask = [Bool](repeating: false, count: canvasLength)
        var precedingSum: Float = 0
        for pos in order { mask[pos] = precedingSum <= entropyBound; precedingSum += entropy[pos] }
        acceptedTokenMask = mask
        return (0..<canvasLength).map { mask[$0] ? denoiser[$0] : current[$0] }
    }

    func renoiseCanvas(accepted: [Int32], randomCanvas: [Int32]) -> [Int32] {
        (0..<canvasLength).map { acceptedTokenMask[$0] ? accepted[$0] : randomCanvas[$0] }
    }
}

/// Seedable RNG (SplitMix64) so a generation run is reproducible.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

func writeInt32(_ values: [Int32], _ path: String) {
    values.withUnsafeBytes { try? Data($0).write(to: URL(fileURLWithPath: path)) }
}

/// Stop when argmax canvas is stable across `stability` steps AND mean entropy < confidence.
struct StableAndConfidentStopping {
    let stability: Int
    let confidence: Float
    private var history: [[Int32]]?
    init(stability: Int, confidence: Float) { self.stability = stability; self.confidence = confidence }

    mutating func step(argmaxCanvas: [Int32], meanEntropy: Float) -> Bool {
        let stable: Bool
        if stability == 0 { stable = true } else {
            if history == nil {
                history = Array(repeating: [Int32](repeating: -1, count: argmaxCanvas.count), count: stability)
            }
            stable = history!.allSatisfy { $0 == argmaxCanvas }
            history!.removeFirst(); history!.append(argmaxCanvas)
        }
        return stable && (meanEntropy < confidence)
    }
}
