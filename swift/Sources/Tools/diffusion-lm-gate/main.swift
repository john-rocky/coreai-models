// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// DiffusionGemma P4 engine FORWARD GATE (Swift / real engine path).
//
// The Python rt.AIModel.load path cannot run this MoE-at-q=256 model: the GPU delegate
// falls back to ANE (ANECompile fails) and the CPU delegate blows up gather_mm memory.
// Both need `expectFrequentReshapes`, which only the Swift SpecializationOptions exposes
// (PreparedModel.prepare derives it for dynamic/"main" models). This tool runs the exact
// seed-0 forward where torch-MPS-eager gave logits cos = -0.002 vs CPU, on the MPSGraph
// engine, and dumps the logits for Python to compare against the CPU baseline.
//
//   encoder(prompt, pos)               -> 60 KV NDArrays (passed straight to the decoder)
//   soft_proj(self_cond[1,CL,V] fp32)  -> soft_embeds[1,CL,H] fp16
//   decoder(canvas, pos, soft=0,  KV)  -> logits0 (step 0 sanity)
//   decoder(canvas, pos, soft_embeds, KV) -> logits1 (THE GATE: real self-cond path)
//
// Run: cd ~/code/coreai/coreai-models && swift run -c release diffusion-lm-gate \
//        ../_diffgemma_coreai ../_diffgemma_gate_io

import CoreAI
import CoreAIShared
import Foundation

// MARK: - small IO helpers

setvbuf(stdout, nil, _IONBF, 0)  // unbuffered: progress prints reach the redirected log live

func die(_ msg: String) -> Never { FileHandle.standardError.write(Data((msg + "\n").utf8)); exit(1) }

func readInt32(_ path: String) -> [Int32] {
    guard let d = FileManager.default.contents(atPath: path) else { die("missing \(path)") }
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
}
func readFloat32(_ path: String) -> [Float] {
    guard let d = FileManager.default.contents(atPath: path) else { die("missing \(path)") }
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
func writeFloat32(_ values: [Float], _ path: String) {
    values.withUnsafeBytes { try? Data($0).write(to: URL(fileURLWithPath: path)) }
}

func ndInt32(_ values: [Int32], _ shape: [Int]) -> NDArray {
    var a = NDArray(shape: shape, scalarType: .int32)
    var view = a.mutableView(as: Int32.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<values.count { ptr[i] = values[i] }
    }
    return a
}
func ndFloat32(_ values: [Float], _ shape: [Int]) -> NDArray {
    var a = NDArray(shape: shape, scalarType: .float32)
    var view = a.mutableView(as: Float.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<values.count { ptr[i] = values[i] }
    }
    return a
}
func ndFloat16(_ values: [Float], _ shape: [Int]) -> NDArray {
    var a = NDArray(shape: shape, scalarType: .float16)
    var view = a.mutableView(as: Float16.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        for i in 0..<values.count { ptr[i] = Float16(values[i]) }
    }
    return a
}
func ndFloat16Zeros(_ shape: [Int]) -> NDArray {
    ndFloat16([Float](repeating: 0, count: shape.reduce(1, *)), shape)
}
func ndToFloats(_ array: NDArray) -> [Float] {
    var out = [Float]()
    switch array.scalarType {
    case .float16:
        array.view(as: Float16.self).withUnsafePointer { ptr, shape, _ in
            let n = (0..<shape.count).reduce(1) { $0 * shape[$1] }
            out.reserveCapacity(n)
            for i in 0..<n { out.append(Float(ptr[i])) }
        }
    case .float32:
        array.view(as: Float.self).withUnsafePointer { ptr, shape, _ in
            let n = (0..<shape.count).reduce(1) { $0 * shape[$1] }
            out.reserveCapacity(n)
            for i in 0..<n { out.append(ptr[i]) }
        }
    default: die("unsupported output scalar type \(array.scalarType)")
    }
    return out
}
func ndToInt32(_ array: NDArray) -> [Int32] {
    var out = [Int32]()
    array.view(as: Int32.self).withUnsafePointer { ptr, shape, _ in
        let n = (0..<shape.count).reduce(1) { $0 * shape[$1] }
        out.reserveCapacity(n)
        for i in 0..<n { out.append(ptr[i]) }
    }
    return out
}

// MARK: - main

let args = CommandLine.arguments
guard args.count >= 3 else { die("usage: diffusion-lm-gate <bundle_dir> <io_dir>") }
let bundleDir = args[1]
let ioDir = args[2]

struct Meta: Decodable { let SP: Int; let CL: Int; let V: Int; let H: Int; let n_layers: Int }
let meta: Meta = {
    guard let d = FileManager.default.contents(atPath: "\(ioDir)/meta.json"),
          let m = try? JSONDecoder().decode(Meta.self, from: d) else { die("bad meta.json") }
    return m
}()
print("meta SP=\(meta.SP) CL=\(meta.CL) V=\(meta.V) H=\(meta.H) layers=\(meta.n_layers)")

// Gate-mode inputs. In DG_GEN (generation) mode these are unused (gen reads gen_prompt_ids.i32
// and builds its own canvas/positions/soft), so tolerate their absence rather than die at startup.
let genMode = ProcessInfo.processInfo.environment["DG_GEN"] != nil
func readInt32OrEmpty(_ p: String) -> [Int32] { FileManager.default.fileExists(atPath: p) ? readInt32(p) : [] }
func readFloat32OrEmpty(_ p: String) -> [Float] { FileManager.default.fileExists(atPath: p) ? readFloat32(p) : [] }
let promptIds = readInt32OrEmpty("\(ioDir)/prompt_ids.i32")
let canvasIds = readInt32OrEmpty("\(ioDir)/canvas_ids.i32")
let encPos = readInt32OrEmpty("\(ioDir)/enc_pos.i32")
let decPos = readInt32OrEmpty("\(ioDir)/dec_pos.i32")
let selfCond = readFloat32OrEmpty("\(ioDir)/self_cond.f32")
if !genMode && selfCond.isEmpty { die("gate mode needs \(ioDir)/self_cond.f32 (or run with DG_GEN=1)") }

func loadGraph(_ name: String) async -> PreparedModel {
    let url = URL(fileURLWithPath: "\(bundleDir)/\(name).aimodel")
    do {
        let pm: PreparedModel
        // DG_SPEC overrides the auto specialization options. "noreshape" = GPU WITHOUT the auto
        // expectFrequentReshapes (the suspected cause of the reuse->zeros bug); "ane" = NeuralEngine.
        switch ProcessInfo.processInfo.environment["DG_SPEC"] {
        case "noreshape":
            pm = try await PreparedModel.prepare(at: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        case "ane":
            pm = try await PreparedModel.prepare(at: url, options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
        default:
            pm = try await PreparedModel.prepare(at: url)  // dynamic -> GPU + expectFrequentReshapes
        }
        print("loaded \(name).aimodel (\(pm.structure)) graphs=\(pm.model.functionNames.count)")
        return pm
    } catch { die("load \(name) failed: \(error)") }
}

func run(_ pm: PreparedModel, _ inputs: [String: NDArray]) async -> [String: NDArray] {
    do {
        guard let fn = try pm.model.loadFunction(named: "main") else { die("no main fn") }
        var outputs = try await fn.run(inputs: inputs)
        var res: [String: NDArray] = [:]
        for name in fn.descriptor.outputNames {
            if let nd = outputs.remove(name)?.ndArray { res[name] = nd }
        }
        return res
    } catch { die("run failed: \(error)") }
}

// Run via an ALREADY-LOADED function handle (loaded once, reused every step — exactly how the
// shipped LLM engines decode token-by-token). Calling loadFunction() per step (as run() does) is
// what corrupts reused decodes to zeros; holding one handle dodges that with NO reload/leak.
func runFn(_ fn: InferenceFunction, _ inputs: [String: NDArray]) async -> [String: NDArray] {
    do {
        var outputs = try await fn.run(inputs: inputs)
        var res: [String: NDArray] = [:]
        for name in fn.descriptor.outputNames {
            if let nd = outputs.remove(name)?.ndArray { res[name] = nd }
        }
        return res
    } catch { die("runFn failed: \(error)") }
}

// The 30-layer q=256 decoder overflows the GPU command queue as one graph (MTL4Command-
// QueueError storm — MPSGraph can't bring up the whole working set at once). When the bundle
// carries decoder_chunks.json the decoder is split into <=6-layer sub-graphs chained host-side
// (hidden handoff); else fall back to the monolithic decoder.aimodel.
struct DecChunks: Decodable { let chunk_size: Int; let n_layers: Int; let ranges: [[Int]] }
let chunksManifest: DecChunks? = {
    guard let d = FileManager.default.contents(atPath: "\(bundleDir)/decoder_chunks.json")
    else { return nil }
    return try? JSONDecoder().decode(DecChunks.self, from: d)
}()

// Load ALL graphs UP FRONT (before any run). Preparing a decoder graph AFTER the encoder has
// run+compiled fails with ENOENT (the encoder run leaves engine/tmp state that breaks a later
// prepare); loading everything first dodges it. The gate runs ONE decode (each chunk runs once)
// so the reuse-stale-model crash (2nd run on one AIModel) never triggers.
let t0 = Date()
let encoder = await loadGraph("encoder")
// DG_ENC_ONLY: load + run ONLY the encoder at several DIFFERENT prompt lengths to test whether a
// DYNAMIC (non-static) encoder graph runs without the MPSGraph dynamic-shape SIGSEGV. Exits after.
if ProcessInfo.processInfo.environment["DG_ENC_ONLY"] != nil {
    for L in [20, 30, 50, 100] {
        let ids = (0..<L).map { Int32(($0 * 7 + 3) % 1000) }
        let pos = (0..<L).map { Int32($0) }
        let te = Date()
        let o = await run(encoder, ["input_ids": ndInt32(ids, [1, L]),
                                    "position_ids": ndInt32(pos, [1, L])])
        print("[enc-only] L=\(L): ran OK, \(o.count) outputs in \(String(format: "%.2fs", -te.timeIntervalSinceNow))")
    }
    print("[enc-only] DYNAMIC encoder ran at L=20,30,50,100 WITHOUT SIGSEGV ✓")
    exit(0)
}
let softProj = await loadGraph("soft_proj")
var decoderChunks: [PreparedModel] = []
var decoder: PreparedModel? = nil
if let cm = chunksManifest {
    for j in 0..<cm.ranges.count { decoderChunks.append(await loadGraph("decoder_chunk\(j)")) }
    print("decoder = \(cm.ranges.count) chunks of <= \(cm.chunk_size) layers: \(cm.ranges)")
} else {
    decoder = await loadGraph("decoder")
}
print(String(format: "loaded graphs in %.1fs", -t0.timeIntervalSinceNow))

// ===== P5 GENERATION MODE (DG_GEN=1) — the 3-phase denoise loop -> coherent "Paris" =====
// Encode the REAL prompt once -> KV; then 48 denoise steps { decode chunk-chain -> softcap logits
// -> temp-divide -> host multinomial/argmax + entropy-bound accept/renoise (Swift sampler) ->
// soft_proj(processed) -> next soft_embeds }, stop on stable&confident, commit argmax.
// NOTE: requires the bundle exported static for the prompt length (SP below). The gate bundle is
// SP=20; re-export with --trace-seq 26 for this prompt. NOTE: runs each chunk + soft_proj 48x ->
// if the reuse-stale-model crash (decision #6) fires on the 2nd run, switch decodeChain to reload
// the chunks per step (the encoder still runs once).
if ProcessInfo.processInfo.environment["DG_GEN"] != nil {
    guard let cm = chunksManifest else { die("DG_GEN needs the chunked bundle (decoder_chunks.json)") }
    // PROMPT = single source of truth: <io>/gen_prompt_ids.i32 (generated from _diffgemma_prompt.json),
    // text "What is the capital of France? Answer in one short sentence." (chat-templated, 26 tokens incl
    // <bos> + user/model turn markers). The bundle MUST be re-exported static for SP = this file's length
    // (e.g. --trace-seq 26). Printed below so the run log shows exactly what was generated.
    let realIds = readInt32("\(ioDir)/gen_prompt_ids.i32")
    guard !realIds.isEmpty else { die("DG_GEN: \(ioDir)/gen_prompt_ids.i32 missing/empty") }
    let env = ProcessInfo.processInfo.environment
    // DG_PADMASK = free variable-length input: the encoder is exported at a fixed static SP=meta.SP;
    // a shorter real prompt (L tokens) is RIGHT-PADDED to SP and the PAD positions are masked out of
    // the canvas cross-attention (the decoder bundle MUST be exported with --pad-mask). The canvas
    // sits right AFTER the real prompt (dec_pos = L..L+CL). Without it: preset path, SP = prompt len.
    let padMask = env["DG_PADMASK"] != nil
    let realL = realIds.count
    let SP = padMask ? meta.SP : realL, CL = meta.CL, V = meta.V, H = meta.H
    if padMask && realL > SP { die("DG_PADMASK: prompt len \(realL) > static SP \(SP)") }
    let promptIds = padMask ? realIds + Array(repeating: Int32(0), count: SP - realL) : realIds
    let canvasStart = padMask ? realL : SP   // canvas RoPE offset = right after the REAL prompt
    print("[gen] PROMPT realL=\(realL) SP=\(SP) padMask=\(padMask) ids=\(realIds)")
    let maxSteps = Int(env["DG_GEN_STEPS"] ?? "") ?? 48
    let soft0 = env["DG_GEN_SOFT0"] != nil   // DIAGNOSTIC: keep soft_embeds=0 every step (isolate reuse vs soft-path)
    let reload = env["DG_GEN_RELOAD"] != nil  // reload each chunk fresh per step (avoid the reuse->silent-zeros bug)
    let rebuildKV = env["DG_REBUILD_KV"] != nil  // rebuild encoder-output KV into fresh NDArrays (decision #6: graph outputs reused across runs)
    let skipZero = env["DG_GEN_SKIPZERO"] != nil  // host workaround: drop steps whose decode silently returned 0s (meanEnt~ln V)
    let seed = UInt64(env["DG_SEED"].flatMap { UInt64($0) } ?? 0)
    let repeatN = Int(env["DG_GEN_REPEAT"] ?? "") ?? 1   // run N gens in ONE process; rep>=2 measures WARM prefill
    let chunkTime = env["DG_GEN_CHUNKTIME"] != nil       // per-chunk decode timing

    // Canvas positions [canvasStart, canvasStart+CL) — right after the REAL prompt (= [SP,..) preset,
    // = [L,..) when padded). Fixed for a given prompt, same every rep.
    let decPosG = ndInt32((0..<CL).map { Int32(canvasStart + $0) }, [1, CL])
    // Additive cross-attention mask over the decoder keys [enc_kv(SP) ++ canvas(CL)] = [1,1,CL,SP+CL]:
    // 0 on the real prompt [0,realL) and the whole canvas, -1e4 on the padded enc positions [realL,SP).
    // Same for every canvas query / layer / step (the pad set is fixed per prompt). nil when !padMask.
    // Mask convention: the EXTERNALIZED engine SDPA uses an ADDITIVE mask (0=attend, -inf=mask) —
    // VERIFIED: additive makes the canvas ignore the pad (france converges in 3 steps == the SP=26
    // preset), whereas bool-as-float (1/0) leaks the pad (28 steps). (The eager _vanilla_sdpa reference
    // bool-izes via logical_not, but it only runs the quant trace with an all-attend ref, so it agrees.)
    let maskAdditive = env["DG_MASK_BOOL"] == nil
    let attnMaskG: NDArray? = {
        guard padMask, realL < SP else { return nil }
        let K = SP + CL
        let attendV: Float = maskAdditive ? 0 : 1
        let maskV: Float = maskAdditive ? -30000 : 0   // -30000 fits fp16, ~-inf after softmax
        var m = [Float](repeating: attendV, count: CL * K)
        for q in 0..<CL { for j in realL..<SP { m[q * K + j] = maskV } }
        return ndFloat16(m, [1, 1, CL, K])
    }()

    // Load each chunk's "main" function ONCE; reuse the handle every step + every rep (runFn). This is
    // how the shipped LLM engines decode without corruption — the per-step loadFunction() in run() was
    // the cause of the reused-decode zeros. (DG_GEN_RELOAD still re-prepares the whole graph per step.)
    var chunkFns: [InferenceFunction] = []
    for (j, pm) in decoderChunks.enumerated() {
        do {
            guard let fn = try pm.model.loadFunction(named: "main") else { die("chunk \(j): no main fn") }
            chunkFns.append(fn)
        } catch { die("chunk \(j) loadFunction failed: \(error)") }
    }
    // GPU-sampler bundle: the last chunk outputs argmax/entropy/soft_embeds (computed on-device) instead
    // of the 67MB logits, and takes a `temp` scalar input. Detected from its output names.
    let gpuSampler = (chunkFns.last?.descriptor.outputNames.contains("argmax")) ?? false
    if gpuSampler { print("[gen] GPU sampler ON: last chunk -> argmax/entropy/soft_embeds (greedy, on-device, no logits readback)") }

    // HOST soft_proj table (embed[V,H] fp32). Loaded ONCE (2.95GB; never changes). The softmax@embed
    // runs on the CPU (Accelerate) — running it on the engine after a decode MTL4-corrupts (q=256 era).
    let embedTable: [Float] = soft0 ? [] : readFloat32("\(ioDir)/embed.f32")
    let embedScale: Float = 53.0659966456864  // H**0.5 (H=2816), matches ScaledEmbedding.embed_scale
    if !soft0 {
        guard embedTable.count == V * H else { die("embed.f32 size \(embedTable.count) != V*H \(V*H)") }
        print("[gen] host soft_proj: loaded embed \(V)x\(H)")
    }

    // kvG is (re)produced by PREFILL each rep; decodeChainGen reads the latest (reference capture).
    var kvG: [String: NDArray] = [:]

    // decode chunk-chain (canvas ids + soft_embeds -> softcapped logits [CL*V]). Returns
    // (logits, gpuForwardSec, readbackSec) to break the GPU-compute cost out from the host round-trip.
    func decodeChainGen(_ canvas: [Int32], _ soft: NDArray, _ temp: Float)
        async -> ([String: NDArray], Double, Double) {
        var hidden: NDArray? = nil
        var fwdSec = 0.0, rbSec = 0.0
        for (j, r) in cm.ranges.enumerated() {
            let s = r[0], e = r[1], isFirst = (j == 0), isLast = (j == cm.ranges.count - 1)
            var inp: [String: NDArray] = ["position_ids": decPosG]
            for li in s..<e { inp["enc_k_\(li)"] = kvG["enc_k_\(li)"]!; inp["enc_v_\(li)"] = kvG["enc_v_\(li)"]! }
            if isFirst { inp["canvas_ids"] = ndInt32(canvas, [1, CL]); inp["soft_embeds"] = soft }
            else { inp["hidden"] = hidden! }
            if let mask = attnMaskG { inp["attn_mask"] = mask }   // free-input pad mask (--pad-mask bundle)
            if isLast && gpuSampler { inp["temp"] = ndFloat32([temp], [1]) }   // fused-sampler temperature
            // Default: reuse the once-loaded function handle (runFn). reload: re-prepare per step.
            let tc = Date()
            let out: [String: NDArray]
            if reload {
                out = await run(await loadGraph("decoder_chunk\(j)"), inp)
            } else {
                out = await runFn(chunkFns[j], inp)
            }
            let fdt = -tc.timeIntervalSinceNow; fwdSec += fdt
            if chunkTime { print(String(format: "    chunk %d [%d,%d) fwd %.2fs", j, s, e, fdt)) }
            // LAST chunk: return its output map; the caller reads either the 67MB logits (fused-host path)
            // or the small argmax/entropy/soft_embeds (GPU-sampler path) — the readback is the caller's choice.
            if isLast { return (out, fwdSec, rbSec) }
            let tr = Date()
            hidden = ndFloat16(ndToFloats(out["hidden"]!), [1, CL, H])   // inter-chunk hidden handoff
            rbSec += -tr.timeIntervalSinceNow
        }
        die("gen: no last chunk")
    }

    for rep in 0..<repeatN {
        if repeatN > 1 { print("===== GEN rep \(rep + 1)/\(repeatN) =====") }
        // PREFILL: encode the prompt once -> per-layer KV (reused every step, just NDArray data).
        let tg = Date()
        let encG = await run(encoder, [
            "input_ids": ndInt32(promptIds, [1, SP]),
            "position_ids": ndInt32((0..<SP).map { Int32($0) }, [1, SP]),
        ])
        kvG = [:]
        for i in 0..<cm.n_layers {
            guard let k = encG["enc_k_\(i)"], let v = encG["enc_v_\(i)"] else { die("gen: encoder missing KV \(i)") }
            // rebuildKV: copy encoder outputs into fresh NDArrays so the chunks don't reuse a graph-output
            // NDArray (carries the encoder's MLIR module ref) across all steps (decision #6).
            kvG["enc_k_\(i)"] = rebuildKV ? ndFloat16(ndToFloats(k), Array(k.shape)) : k
            kvG["enc_v_\(i)"] = rebuildKV ? ndFloat16(ndToFloats(v), Array(v.shape)) : v
        }
        print(String(format: "[gen] prefill SP=%d in %.2fs%@", SP, -tg.timeIntervalSinceNow,
                     rep == 0 ? " (COLD: incl first-run graph compile)" : " (WARM)"))

        var rng = SplitMix64(seed: seed)
        var sampler = EntropyBoundSampler(entropyBound: 0.1, canvasLength: CL, vocabSize: V)
        var stopping = StableAndConfidentStopping(stability: 1, confidence: 0.005)
        var canvas = sampler.initializeCanvas(&rng)
        var argmaxCanvas = canvas
        var soft = ndFloat16Zeros([1, CL, H])  // step 0: soft_embeds = 0 (== self_cond None)
        let tgen = Date()
        // The temperature/denoise schedule tracks GOOD steps (good counts down maxSteps..1); skipped
        // zero-decodes (engine reuse bug) do NOT advance it, so the GOOD steps span the full 0.8->0.4
        // schedule (a proper N-step denoise) even though some iterations may be wasted zeros.
        var good = maxSteps
        var iter = 0
        let iterCap = maxSteps * 5
        while good >= 1 && iter < iterCap {
            iter += 1
            let ts = Date()
            let temp = linearTemperature(curStep: good, tMin: 0.4, tMax: 0.8, steps: maxSteps)
            let (out, fwdSec, rbSecI) = await decodeChainGen(canvas, soft, temp)
            let tRead = Date()
            // GPU-sampler bundle: argmax (greedy committed token) + entropy + soft_embeds come straight off
            // the device (no 67MB logits readback). Else: read the [CL*V] logits + run the fused host step.
            let entropy: [Float], amax: [Int32], denoiser: [Int32], softFloats: [Float]
            let meanEnt: Float
            if gpuSampler {
                amax = ndToInt32(out["argmax"]!)
                let en = ndToFloats(out["entropy"]!)
                entropy = en; meanEnt = en.reduce(0, +) / Float(CL); denoiser = amax   // greedy denoiser
                softFloats = soft0 ? [] : ndToFloats(out["soft_embeds"]!)
            } else {
                let raw = ndToFloats(out["logits"]!)                                   // 67MB readback
                let r = fusedHostStep(raw: raw, temp: temp, count: CL, vocab: V, hidden: H,
                                      embed: embedTable, embedScale: embedScale, soft0: soft0, &rng)
                entropy = r.entropy; meanEnt = r.meanEnt; amax = r.argmax; denoiser = r.denoiser; softFloats = r.soft
            }
            let rbSec = rbSecI + (-tRead.timeIntervalSinceNow)
            let tHost = Date()
            // skipZero: the engine reuse bug silently returns all-zero logits on some reused decodes
            // (meanEnt ~= ln(vocab) = 12.49). Drop those — keep prior canvas/soft, don't advance good.
            if skipZero && meanEnt > 10.0 {
                print(String(format: "[gen] iter %d SKIP (zero meanEnt=%.3f) good=%d (%.1fs)", iter, meanEnt, good, -ts.timeIntervalSinceNow))
                continue
            }
            argmaxCanvas = amax
            // Persist the committed canvas EVERY step: a later GPU step can deadlock, so writing here
            // guarantees the best-so-far (already-converged) argmax canvas is captured even if it hangs.
            writeInt32(argmaxCanvas, "\(ioDir)/gen_ids.i32")
            let accepted = sampler.acceptCanvas(current: canvas, denoiser: denoiser, logits: [], entropy: entropy)
            canvas = sampler.renoiseCanvas(accepted: accepted, randomCanvas: sampler.initializeCanvas(&rng))
            let done = stopping.step(argmaxCanvas: argmaxCanvas, meanEntropy: meanEnt)
            let hostSec = -tHost.timeIntervalSinceNow   // accept/renoise (+ fused host on the non-sampler path)
            if !soft0 { soft = ndFloat16(softFloats, [1, CL, H]) }
            let softSec = 0.0
            let acc = sampler.acceptedTokenMask.filter { $0 }.count
            let uniq = Set(argmaxCanvas).count
            print(String(format: "[gen] good %2d/%d (iter %d) t=%.3f acc=%3d/%d meanEnt=%.4f uniq=%3d stop=%@ | step %.2fs = fwd %.2f + rb %.2f + host %.2f + soft %.2f",
                         maxSteps - good + 1, maxSteps, iter, temp, acc, CL, meanEnt, uniq, done ? "Y" : "n",
                         -ts.timeIntervalSinceNow, fwdSec, rbSec, hostSec, softSec))
            good -= 1
            if done { break }
        }
        writeInt32(argmaxCanvas, "\(ioDir)/gen_ids.i32")
        print(String(format: "[gen] rep %d DONE in %.2fs; argmax head=%@", rep + 1, -tgen.timeIntervalSinceNow,
                     String(describing: Array(argmaxCanvas.prefix(20)))))
    }
    print("[gen] wrote \(ioDir)/gen_ids.i32 (decode with the tokenizer in .venv-diffgemma)")
    exit(0)
}

// 1) encoder -> KV NDArrays (passed straight to the decoder, no flatten/reshape).
var te = Date()
let encOut = await run(encoder, [
    "input_ids": ndInt32(promptIds, [1, meta.SP]),
    "position_ids": ndInt32(encPos, [1, meta.SP]),
])
var kv: [String: NDArray] = [:]
for i in 0..<meta.n_layers {
    for n in ["enc_k_\(i)", "enc_v_\(i)"] {
        guard let a = encOut[n] else { die("encoder missing \(n)") }
        kv[n] = a
    }
}
print(String(format: "encoder: %d KV in %.1fs", kv.count, -te.timeIntervalSinceNow))

// Localization dumps: encoder enc_hidden + sample KV (sliding layer 0, full/k_eq_v layer 5,
// last layer 29) so _diffgemma_gate_compare.py can pinpoint whether a low logits cos is an
// encoder bug, a soft_proj bug, or a decoder bug (vs the baseline enc_hidden / enc_k_*/enc_v_*).
if let eh = encOut["enc_hidden"] { writeFloat32(ndToFloats(eh), "\(ioDir)/eng_enc_hidden.f32") }
for li in [0, 5, 29] {
    if let k = kv["enc_k_\(li)"] { writeFloat32(ndToFloats(k), "\(ioDir)/eng_enc_k_\(li).f32") }
    if let v = kv["enc_v_\(li)"] { writeFloat32(ndToFloats(v), "\(ioDir)/eng_enc_v_\(li).f32") }
}

// 2) soft_proj(self_cond) -> soft_embeds (fp16). Rebuild into a FRESH NDArray: a graph's
// OUTPUT NDArray carries a reference to its own MLIR module, and feeding it straight into the
// decoder crashes the decoder's shape resolution (getFuncOp SymbolTable lookup -> SIGSEGV).
te = Date()
let spOut = await run(softProj, ["logits": ndFloat32(selfCond, [1, meta.CL, meta.V])])
guard let softRaw = spOut["soft_embeds"] else { die("soft_proj missing soft_embeds") }
let softEmbeds = ndFloat16(ndToFloats(softRaw), [1, meta.CL, meta.H])
writeFloat32(ndToFloats(softRaw), "\(ioDir)/eng_soft.f32")  // vs baseline soft1
print(String(format: "soft_proj in %.1fs (soft_embeds %@)", -te.timeIntervalSinceNow,
             String(describing: softRaw.scalarType)))

// 3) decoder — ONE decode (step1). Each graph runs exactly once (single run, so no reuse).
// Monolithic: feed canvas+pos+soft + all 60 KV in one shot.
let decodeMono: (NDArray) async -> [Float] = { soft in
    var inp = kv
    inp["canvas_ids"] = ndInt32(canvasIds, [1, meta.CL])
    inp["position_ids"] = ndInt32(decPos, [1, meta.CL])
    inp["soft_embeds"] = soft
    let out = await run(decoder!, inp)
    guard let lg = out["logits"] else { die("decoder missing logits") }
    return ndToFloats(lg)
}

// Chunked: chain decoder_chunk0..K. chunk0 takes (canvas,pos,soft, KV[s:e)) -> hidden; each
// later chunk takes (hidden,pos, KV[s:e)) -> hidden; the last -> logits. Each chunk gets only
// the encoder KV for its own absolute layer range. The hidden handoff is REBUILT into a fresh
// NDArray between chunks (a graph output carries its MLIR module ref → feeding it straight into
// the next graph crashes shape resolution / getFuncOp SymbolTable SIGSEGV; same fix as soft_embeds).
let decodeChunked: (NDArray, DecChunks) async -> [Float] = { soft, cm in
    var hidden: NDArray? = nil
    for (j, r) in cm.ranges.enumerated() {
        let s = r[0], e = r[1]
        let isFirst = (j == 0), isLast = (j == cm.ranges.count - 1)
        var inp: [String: NDArray] = ["position_ids": ndInt32(decPos, [1, meta.CL])]
        for li in s..<e {
            guard let k = kv["enc_k_\(li)"], let v = kv["enc_v_\(li)"] else { die("missing KV \(li)") }
            inp["enc_k_\(li)"] = k
            inp["enc_v_\(li)"] = v
        }
        if isFirst {
            inp["canvas_ids"] = ndInt32(canvasIds, [1, meta.CL])
            inp["soft_embeds"] = soft
        } else {
            inp["hidden"] = hidden!
        }
        let tc = Date()
        let out = await run(decoderChunks[j], inp)
        if isLast {
            guard let lg = out["logits"] else { die("chunk \(j) missing logits") }
            print(String(format: "  chunk %d [%d,%d) -> logits in %.1fs", j, s, e, -tc.timeIntervalSinceNow))
            return ndToFloats(lg)
        }
        guard let hRaw = out["hidden"] else { die("chunk \(j) missing hidden") }
        hidden = ndFloat16(ndToFloats(hRaw), [1, meta.CL, meta.H])  // rebuild fresh
        print(String(format: "  chunk %d [%d,%d) -> hidden in %.1fs", j, s, e, -tc.timeIntervalSinceNow))
    }
    die("no last chunk produced logits")
}

// ONE decode per process (each graph runs once → no reuse-stale-model crash). DG_GATE_STEP0=1
// runs the chain with soft_embeds=0 (== self_cond=None) → eng_logits0.f32 (the decoder-without-
// self-cond anchor, run as a SEPARATE process to dodge reuse); else step1 (REAL self_cond) →
// eng_logits1.f32 (THE gate). soft=0 reproduces self_cond=None since self_conditioning(e,0)=post_norm(e).
let step0 = ProcessInfo.processInfo.environment["DG_GATE_STEP0"] != nil
let softInput = step0 ? ndFloat16Zeros([1, meta.CL, meta.H]) : softEmbeds
te = Date()
let logits = chunksManifest != nil
    ? await decodeChunked(softInput, chunksManifest!)
    : await decodeMono(softInput)
let outName = step0 ? "eng_logits0.f32" : "eng_logits1.f32"
print(String(format: "decoder %@ (GATE) in %.1fs", step0 ? "step0" : "step1", -te.timeIntervalSinceNow))
writeFloat32(logits, "\(ioDir)/\(outName)")
print("wrote \(outName) (\(logits.count) floats)")
print(String(format: "GATE forward done in %.1fs total", -t0.timeIntervalSinceNow))
