// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// spec-decode — lossless greedy n-gram speculative decoding on a DENSE Core AI
// decode bundle, plus the two measurements that decide whether it can pay.
//
//   --mode rows         the logits-path gate: row i of a [1, q, vocab] verify forward
//                       must equal the argmax of a plain S=1 decode at that position.
//                       If the fast path will not surrender all q rows, nothing else
//                       in this tool is meaningful — run this first.
//   --mode verify-cost  per-forward wall time at S = 1,2,4,8,16,32 (the c_v table):
//                       how many decode steps one verify pass costs.
//   --mode gen          one generation, spec on (K>0) or off (K=0).
//   --mode ab           spec off then on, same prompt, same process: proves the
//                       token streams are IDENTICAL (lossless) and reports the speedup.
//
// Greedy only, batch 1. Speculative decoding changes SPEED, never the output: every
// emitted token is the target model's own argmax, which `--mode ab` re-proves on
// every run rather than asserting.
//
// Run: cd ~/code/coreai/coreai-models && swift build -c release --product spec-decode
//      .build/release/spec-decode --model exports/<bundle> --mode ab --prompt "..."

import ArgumentParser
import CoreAI
import CoreAILanguageModels
import CoreAIShared
import Foundation
import Tokenizers

@main
struct Main {
    static func main() async throws {
        setvbuf(stdout, nil, _IOLBF, 0)  // stream progress when stdout is a log file
        await SpecDecode.main()
    }
}

enum RunMode: String, ExpressibleByArgument {
    case rows
    case verifyCost = "verify-cost"
    case gen
    case ab
}

struct SpecDecode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "spec-decode",
        abstract: "Lossless n-gram speculative decoding on a dense Core AI decode bundle"
    )

    @Option(name: .customLong("model"), help: "Model bundle directory")
    var model: String

    @Option(help: "Prompt text")
    var prompt: String?

    @Option(name: .customLong("prompt-file"), help: "Read the prompt from a UTF-8 file")
    var promptFile: String?

    @Option(name: .customLong("mode"), help: "rows | verify-cost | gen | ab")
    var mode: RunMode = .ab

    @Option(name: .customShort("k"), help: "Draft length per round (0 = plain greedy)")
    var k: Int = 8

    @Flag(name: .customLong("adaptive"), help: "Grow/shrink the draft length with recent acceptance")
    var adaptive: Bool = false

    @Option(name: .customLong("max-tokens"), help: "Generated tokens per run")
    var maxTokens: Int = 256

    @Option(name: .customLong("ngram-max"), help: "Longest suffix the drafter matches on")
    var ngramMax: Int = 3

    @Option(name: .customLong("ngram-min"), help: "Shortest suffix the drafter matches on")
    var ngramMin: Int = 1

    @Flag(name: .customLong("ngram-first-match"), help: "Draft off the FIRST earlier match, not the most recent")
    var ngramFirstMatch: Bool = false

    @Option(name: .customLong("kv-capacity"), help: "KV cache capacity in tokens")
    var kvCapacity: Int = 4096

    @Option(name: .customLong("prefill-chunk"), help: "Prompt tokens per prefill forward")
    var prefillChunk: Int = 256

    @Option(name: .customLong("repeat"), help: "A/B pairs (ab) or runs (gen)")
    var repeatCount: Int = 1

    @Option(name: .customLong("rows"), help: "Rows compared in --mode rows")
    var rowCount: Int = 8

    @Option(name: .customLong("sweep-s"), help: "Comma-separated query lengths for --mode verify-cost")
    var sweepS: String = "1,2,4,8,16,32"

    @Flag(name: .customLong("raw"), help: "Feed the prompt verbatim (skip the chat template)")
    var raw: Bool = false

    @Option(name: .customLong("tools-file"), help: "JSON array of tool specs for the chat template")
    var toolsFile: String?

    @Option(name: .customLong("messages-file"), help: "JSON array of chat messages (overrides --prompt)")
    var messagesFile: String?

    @Flag(name: .customLong("print-text"), help: "Print the generated text")
    var printText: Bool = false

    @Option(name: .customLong("json"), help: "Write the result summary to this JSON file")
    var jsonPath: String?

    @Option(name: .customLong("label"), help: "Workload label carried into the JSON summary")
    var label: String?

    private var sweepLengths: [Int] {
        sweepS.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            .filter { $0 > 0 }
    }

    // MARK: - Run

    mutating func run() async throws {
        let bundle = try LanguageBundle(from: model)
        let assetURL = PreparedModel.resolveCoreAIModelURL(
            from: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main))

        let tokenizer = try await bundle.loadTokenizer()
        let promptTokens = try promptTokenIds(tokenizer: tokenizer)
        let stops = stopTokenIds(bundle: bundle, tokenizer: tokenizer)

        guard promptTokens.count + maxTokens + k + 1 <= kvCapacity else {
            throw SpecDecodeError(
                "prompt \(promptTokens.count) + \(maxTokens) generated + draft \(k) does not fit "
                    + "--kv-capacity \(kvCapacity)")
        }

        // One forward never feeds more than a prefill chunk or a full draft round.
        let maxQuery = max(
            prefillChunk, k + 1, mode == .verifyCost ? (sweepLengths.max() ?? 32) : 1, rowCount)
        let loadStart = ContinuousClock.now
        let engine = try await SpecModel(
            assetURL: assetURL, kvCapacity: kvCapacity, maxQuery: maxQuery)
        print(
            String(
                format: "loaded %@ · vocab %d · KV %d tok (%.2f GB) · %.1f s",
                engine.name, engine.vocab, kvCapacity,
                Double(engine.kvByteCount) / 1e9, seconds(since: loadStart)))
        print("prompt: \(promptTokens.count) tokens · stop ids \(stops.sorted())")

        var summary: [String: Any] = [
            "model": engine.name,
            "mode": mode.rawValue,
            "prompt_tokens": promptTokens.count,
            "max_tokens": maxTokens,
            "k": k,
            "ngram": ["max": ngramMax, "min": ngramMin, "prefer_recent": !ngramFirstMatch],
        ]
        if let label { summary["label"] = label }

        switch mode {
        case .rows:
            summary["rows"] = try await runRowGate(engine: engine, promptTokens: promptTokens)
        case .verifyCost:
            summary["verify_cost"] = try await runVerifyCost(
                engine: engine, promptTokens: promptTokens)
        case .gen:
            var runs: [[String: Any]] = []
            for _ in 0..<repeatCount {
                let run = try await generate(
                    engine: engine, promptTokens: promptTokens, stops: stops,
                    draftK: k, tokenizer: tokenizer)
                runs.append(run.summary)
            }
            summary["runs"] = runs
        case .ab:
            summary["ab"] = try await runAB(
                engine: engine, promptTokens: promptTokens, stops: stops, tokenizer: tokenizer)
        }

        if let jsonPath {
            let data = try JSONSerialization.data(
                withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: jsonPath))
            print("wrote \(jsonPath)")
        }
    }

    // MARK: - Mode: rows (the logits-path gate)

    /// Feed the SAME tokens two ways and compare per-position argmax:
    ///   sequential — `rowCount` plain S=1 decodes, each reading row 0;
    ///   batched    — ONE S=rowCount forward, reading all rows.
    /// Row i of the batched forward is conditioned on the prefix through token i, which
    /// is exactly the sequential run's conditioning at that step, so the two must agree
    /// token for token. They are the same graph, so this is a claim about the runtime
    /// handing back every row — not about the model.
    private func runRowGate(engine: SpecModel, promptTokens: [Int32]) async throws -> [String: Any] {
        print("\n=== row gate: S=1 sequential vs S=\(rowCount) batched ===")

        engine.reset()
        try await prefill(engine: engine, tokens: promptTokens)
        var sequential: [(token: Int32, value: Float)] = []
        var fed: [Int32] = []
        var next = engine.argmaxWithValue(row: engine.lastRows - 1)
        for _ in 0..<rowCount {
            sequential.append(next)
            fed.append(next.token)
            try await engine.forward([next.token])
            engine.commit(1)
            next = engine.argmaxWithValue(row: 0)
        }
        // `fed` = the tokens the batched forward will consume; `expected[i]` = the
        // sequential run's prediction AFTER fed[i] — i.e. row i+1 of the sequential
        // stream, which is what batched row i must reproduce (token and value alike).
        let expected = Array(sequential.dropFirst()) + [next]

        engine.reset()
        try await prefill(engine: engine, tokens: promptTokens)
        try await engine.forward(fed)
        let batched = (0..<rowCount).map { engine.argmaxWithValue(row: $0) }

        var mismatches = 0
        var maxDelta = 0.0
        print("row  fed        S=1 argmax   S=\(rowCount) argmax  Δtop")
        for i in 0..<rowCount {
            let delta = abs(Double(expected[i].value) - Double(batched[i].value))
            maxDelta = max(maxDelta, delta)
            let ok = expected[i].token == batched[i].token
            if !ok { mismatches += 1 }
            print(
                String(
                    format: "%3d  %-9d  %-11d  %-11d  %.4f %@",
                    i, fed[i], expected[i].token, batched[i].token, delta,
                    ok ? "" : "  <-- MISMATCH"))
        }
        let verdict = mismatches == 0 ? "PASS" : "FAIL"
        print(
            String(
                format: "ROW GATE %@ — %d/%d rows match, max |Δ| on the top logit %.5f",
                verdict, rowCount - mismatches, rowCount, maxDelta))
        return [
            "rows": rowCount, "mismatches": mismatches, "max_top_delta": maxDelta,
            "pass": mismatches == 0,
        ]
    }

    // MARK: - Mode: verify-cost

    /// Per-forward wall time at a set of query lengths, all at the SAME context offset
    /// (nothing is committed), so the only variable is S. `c_v` = cost of a verify pass
    /// in units of one decode step; a pass pays whenever it commits more than `c_v` tokens.
    private func runVerifyCost(engine: SpecModel, promptTokens: [Int32]) async throws
        -> [[String: Any]]
    {
        print("\n=== verify cost sweep ===")
        engine.reset()
        try await prefill(engine: engine, tokens: promptTokens)

        let lengths = sweepLengths.filter { $0 <= engine.maxQuery }
        let trials = 5
        var baseline = 0.0
        var rows: [[String: Any]] = []
        print("   S   median ms   vs S=1   S sequential decodes")
        for s in lengths {
            let feed = [Int32](repeating: promptTokens.last ?? 1, count: s)
            var samples: [Double] = []
            for trial in 0...trials {
                try await engine.forward(feed)  // never committed: same offset every time
                if trial > 0 { samples.append(engine.lastForwardMs) }  // trial 0 pays the JIT
            }
            samples.sort()
            let median = samples[samples.count / 2]
            if s == 1 { baseline = median }
            let ratio = baseline > 0 ? median / baseline : 1
            print(
                String(
                    format: "%4d   %8.1f   %6.2fx   %8.1f ms", s, median, ratio,
                    baseline * Double(s)))
            rows.append([
                "s": s, "median_ms": median, "vs_decode_step": ratio,
                "sequential_ms": baseline * Double(s),
            ])
        }
        return rows
    }

    // MARK: - Mode: ab

    private func runAB(
        engine: SpecModel, promptTokens: [Int32], stops: Set<Int32>, tokenizer: any Tokenizer
    ) async throws -> [String: Any] {
        var reference: [Int32]?
        var offRuns: [GenerationRun] = []
        var onRuns: [GenerationRun] = []
        var lossless = true

        // off first in every pair, so the ON run is the one that inherits the warmer GPU —
        // if thermals bias the ratio at all, they bias it against the claim.
        for _ in 0..<repeatCount {
            for specOn in [false, true] {
                let run = try await generate(
                    engine: engine, promptTokens: promptTokens, stops: stops,
                    draftK: specOn ? k : 0, tokenizer: tokenizer)
                if let reference {
                    if reference != run.tokens {
                        lossless = false
                        let firstDiff =
                            zip(reference, run.tokens).enumerated().first { $0.element.0 != $0.element.1 }?
                            .offset ?? min(reference.count, run.tokens.count)
                        print(
                            "  ⚠️ token streams diverge at index \(firstDiff) "
                                + "(\(reference.count) vs \(run.tokens.count) tokens)")
                    }
                } else {
                    reference = run.tokens
                }
                if specOn { onRuns.append(run) } else { offRuns.append(run) }
            }
        }

        let offBest = offRuns.map(\.tokensPerSecond).max() ?? 0
        let onBest = onRuns.map(\.tokensPerSecond).max() ?? 0
        let speedup = offBest > 0 ? onBest / offBest : 0
        print("")
        print(
            String(
                format: "LOSSLESS %@ — %d runs, %d tokens each, token streams %@",
                lossless ? "PASS" : "FAIL", offRuns.count + onRuns.count, reference?.count ?? 0,
                lossless ? "identical" : "DIVERGED (see the warning above)"))
        print(
            String(
                format: "SPEC OFF %.2f tok/s · SPEC ON %.2f tok/s · %.2fx (best of %d each)",
                offBest, onBest, speedup, repeatCount))
        if printText, let reference {
            print("--- text ---")
            print(tokenizer.decode(tokens: reference.map(Int.init), skipSpecialTokens: false))
        }
        return [
            "lossless": lossless,
            "off": offRuns.map(\.summary),
            "on": onRuns.map(\.summary),
            "off_best_tps": offBest,
            "on_best_tps": onBest,
            "speedup": speedup,
        ]
    }

    // MARK: - The loop

    struct GenerationRun {
        var specOn = false
        var draftK = 0
        var tokens: [Int32] = []
        var rounds = 0
        var targetForwards = 0
        var acceptedDrafts = 0
        var draftedTokens = 0
        var roundsWithDraft = 0
        var draftBudgetSum = 0
        var decodeSeconds = 0.0
        /// Sum of the verify forwards' own wall time (encode + GPU drain).
        var forwardSeconds = 0.0
        var stopped = false

        /// Everything the decode loop spends outside the forward: drafting, row argmax,
        /// bookkeeping. The 27B spec-decode work lost half its wall clock here.
        var hostSeconds: Double { max(0, decodeSeconds - forwardSeconds) }

        var tokensPerSecond: Double { decodeSeconds > 0 ? Double(tokens.count) / decodeSeconds : 0 }
        /// Average tokens committed per verify forward — the speedup over plain decode.
        var tokensPerForward: Double {
            targetForwards > 0 ? Double(tokens.count) / Double(targetForwards) : 0
        }
        /// Drafted tokens accepted per round; the ā of the kickoff arithmetic.
        var alpha: Double { rounds > 0 ? Double(acceptedDrafts) / Double(rounds) : 0 }
        /// Share of drafted tokens that survived verification.
        var draftAcceptRate: Double {
            draftedTokens > 0 ? Double(acceptedDrafts) / Double(draftedTokens) : 0
        }
        /// Mean draft length actually attempted (== k when fixed, lower when adaptive).
        var meanDraftBudget: Double {
            rounds > 0 ? Double(draftBudgetSum) / Double(rounds) : 0
        }

        var summary: [String: Any] {
            [
                "spec": specOn, "k": draftK, "generated": tokens.count, "rounds": rounds,
                "target_forwards": targetForwards, "accepted_drafts": acceptedDrafts,
                "drafted_tokens": draftedTokens, "rounds_with_draft": roundsWithDraft,
                "mean_draft_budget": meanDraftBudget,
                "alpha": alpha, "tokens_per_forward": tokensPerForward,
                "draft_accept_rate": draftAcceptRate,
                "decode_seconds": decodeSeconds, "tokens_per_second": tokensPerSecond,
                "forward_seconds": forwardSeconds, "host_seconds": hostSeconds,
                "hit_stop": stopped,
            ]
        }
    }

    /// One greedy generation. `draftK == 0` is the baseline: the identical loop with no
    /// proposals, i.e. a plain S=1 decode per forward — so the A/B compares two paths
    /// through the same code, not two programs.
    private func generate(
        engine: SpecModel, promptTokens: [Int32], stops: Set<Int32>, draftK: Int,
        tokenizer: any Tokenizer
    ) async throws -> GenerationRun {
        var drafter = NgramDrafter()
        drafter.maxNgram = ngramMax
        drafter.minNgram = ngramMin
        drafter.preferRecent = !ngramFirstMatch

        var run = GenerationRun(specOn: draftK > 0, draftK: draftK)

        engine.reset()
        let prefillStart = ContinuousClock.now
        try await prefill(engine: engine, tokens: promptTokens)
        let prefillSeconds = seconds(since: prefillStart)

        // The prompt's last row already predicts the first generated token.
        var pending = engine.argmax(row: engine.lastRows - 1)
        var context = promptTokens

        // Warm every query length the loop will use (1...draftK+1). A dynamic-shape graph
        // specializes per shape on first sight; without this the first round of each shape
        // pays JIT inside the timed window.
        for s in 1...(draftK + 1) {
            try await engine.forward([Int32](repeating: pending, count: s))
        }

        // Adaptive draft length. The cost sweep on this bundle is a step function —
        // S=2 costs the same as a decode step, S=4…9 cost a flat ~1.45× — so a fixed
        // K can only lose where acceptance is weak, and a K that falls back to 1 has a
        // free floor. Grow on a fully-accepted round, shrink on any rejection.
        var currentK = adaptive ? min(draftK, 1) : draftK

        let decodeStart = ContinuousClock.now
        while run.tokens.count < maxTokens {
            if stops.contains(pending) {
                run.stopped = true
                break
            }
            let budget = min(currentK, maxTokens - run.tokens.count - 1, engine.room - 2)
            let draft = budget > 0
                ? drafter.propose(context: context + [pending], k: budget) : []
            run.draftBudgetSum += max(budget, 0)

            try await engine.forward([pending] + draft)
            run.forwardSeconds += engine.lastForwardMs / 1000
            run.targetForwards += 1
            run.rounds += 1
            run.draftedTokens += draft.count
            if !draft.isEmpty { run.roundsWithDraft += 1 }

            // Row i holds the model's own next-token argmax after feed[i]. Accept the
            // longest prefix of the draft the target would have produced itself.
            var accepted = 0
            while accepted < draft.count, engine.argmax(row: accepted) == draft[accepted] {
                accepted += 1
            }
            let committed = [pending] + draft.prefix(accepted)
            pending = engine.argmax(row: accepted)
            engine.commit(committed.count)  // reject == do not advance (dense KV rollback)
            run.acceptedDrafts += accepted
            context.append(contentsOf: committed)

            if adaptive && !draft.isEmpty {
                currentK = accepted == draft.count
                    ? min(draftK, currentK + 2)
                    : max(1, currentK - 1)
            }

            for token in committed {
                if stops.contains(token) {
                    run.stopped = true
                    break
                }
                run.tokens.append(token)
                if run.tokens.count >= maxTokens { break }
            }
            if run.stopped { break }
        }
        run.decodeSeconds = seconds(since: decodeStart)

        print(
            String(
                format:
                    "RUN spec=%@ k=%d%@ gen=%d fwd=%d alpha=%.2f tok/fwd=%.2f accept=%.0f%% "
                    + "decode=%.2fs (fwd %.2f + host %.2f) tps=%.2f (prefill %.2fs, %.0f tok/s)%@",
                run.specOn ? "on " : "off", draftK,
                adaptive && draftK > 0 ? String(format: "(adaptive, mean %.1f)", run.meanDraftBudget) : "",
                run.tokens.count, run.targetForwards,
                run.alpha, run.tokensPerForward, run.draftAcceptRate * 100,
                run.decodeSeconds, run.forwardSeconds, run.hostSeconds,
                run.tokensPerSecond, prefillSeconds,
                Double(promptTokens.count) / prefillSeconds, run.stopped ? " [stop]" : ""))
        if printText && mode == .gen {
            print("--- text ---")
            print(tokenizer.decode(tokens: run.tokens.map(Int.init), skipSpecialTokens: false))
        }
        return run
    }

    /// Prompt prefill in chunks; the last chunk leaves its logits in the buffer.
    private func prefill(engine: SpecModel, tokens: [Int32]) async throws {
        var index = 0
        while index < tokens.count {
            let end = min(index + prefillChunk, tokens.count)
            try await engine.forward(Array(tokens[index..<end]))
            engine.commit(end - index)
            index = end
        }
    }

    // MARK: - Prompt / stops

    private func promptTokenIds(tokenizer: any Tokenizer) throws -> [Int32] {
        var text = prompt ?? "Hello, how are you?"
        if let promptFile {
            text = try String(contentsOfFile: promptFile, encoding: .utf8)
        }
        if raw {
            return tokenizer.encode(text: text).map(Int32.init)
        }
        let messages: [Message] =
            try messagesFile.map { try Self.loadJSONObjects(path: $0) }
            ?? [["role": "user", "content": text]]
        let tools: [ToolSpec]? = try toolsFile.map { try Self.loadJSONObjects(path: $0) }
        return try tokenizer.applyChatTemplate(messages: messages, tools: tools).map(Int32.init)
    }

    /// Read a JSON array of objects into the `[String: any Sendable]` shape the chat
    /// template expects. JSONSerialization hands back bridged ObjC values (NSNumber,
    /// NSNull) that the Jinja sandbox renders wrong, so everything is rebuilt as native
    /// Swift and nulls are dropped rather than passed through.
    private static func loadJSONObjects(path: String) throws -> [[String: any Sendable]] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let array = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw SpecDecodeError("\(path): expected a JSON array of objects")
        }
        return array.map { nativeObject($0) }
    }

    private static func nativeObject(_ object: [String: Any]) -> [String: any Sendable] {
        var result: [String: any Sendable] = [:]
        for (key, value) in object {
            if let native = nativeValue(value) { result[key] = native }
        }
        return result
    }

    private static func nativeValue(_ value: Any) -> (any Sendable)? {
        switch value {
        case is NSNull: return nil
        case let object as [String: Any]: return nativeObject(object)
        case let array as [Any]: return array.compactMap { nativeValue($0) } as [any Sendable]
        case let string as String: return string
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue }
            let double = number.doubleValue
            return double == double.rounded() ? Int(double) : double
        default: return nil
        }
    }

    /// EOS ids. `additionalStopTokenIds` reads tokenizer_config.json; Muse-Glimmer's
    /// config declares neither `added_tokens_decoder` nor `additional_special_tokens`
    /// (transformers-5 `TokenizersBackend`), so the ATEM turn markers are resolved by
    /// name as well. `<|eom|>` is deliberately NOT a stop: it ends a message inside the
    /// turn (the reasoning channel), not the turn.
    private func stopTokenIds(bundle: LanguageBundle, tokenizer: any Tokenizer) -> Set<Int32> {
        var stops = Set<Int32>()
        if let eos = tokenizer.eosTokenId { stops.insert(Int32(eos)) }
        if let dir = bundle.tokenizerPath {
            stops.formUnion(LanguageConfig.additionalStopTokenIds(from: dir, tokenizer: tokenizer))
        }
        for token in ["<|eot|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"] {
            if let id = tokenizer.convertTokenToId(token) { stops.insert(Int32(id)) }
        }
        return stops
    }
}
