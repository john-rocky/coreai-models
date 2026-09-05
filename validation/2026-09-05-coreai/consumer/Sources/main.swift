import CoreAI
import CoreAILanguageModels
import CoreAIShared
import Foundation

// A real-bundle gate. Output is JSONL and flushed before each risky boundary.
func emit(_ event: String, _ fields: [String: Any] = [:]) {
    var record = fields
    record["event"] = event
    record["time"] = ISO8601DateFormatter().string(from: Date())
    let data = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data([10]))
}

struct GateError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw GateError(message) }
}

let args = CommandLine.arguments
guard args.count == 3 else {
    emit("error", ["message": "usage: bundle-gate BUNDLE s1|dynamic"])
    exit(64)
}
let url = URL(fileURLWithPath: args[1])
let isS1 = args[2] == "s1"

do {
    var failures: [String] = []
    func parity(_ name: String, _ actual: [Int32], _ expected: [Int32]) {
        if actual == expected {
            emit(name + "_pass")
        } else {
            failures.append(name)
            emit(name + "_fail", ["actual": actual, "expected": expected])
        }
    }
    emit("start", ["bundle": url.path, "case": args[2],
                   "chunk_threshold_env": ProcessInfo.processInfo.environment["COREAI_CHUNK_THRESHOLD"] ?? "unset"])
    let bundle = try LanguageBundle(at: url)
    let assetURL = try bundle.requireModelURL(for: "main")
    let prepared = try await PreparedModel.prepare(at: assetURL)
    let fnName = bundle.language.functionMap?.name(for: "main") ?? "main"
    guard let descriptor = prepared.model.functionDescriptor(for: fnName) else {
        throw GateError("missing function descriptor")
    }
    var inputShapes: [String: [Int]] = [:]
    var outputShapes: [String: [Int]] = [:]
    var stateShapes: [String: [Int]] = [:]
    for name in descriptor.inputNames {
        if case .ndArray(let d) = descriptor.inputDescriptor(of: name) { inputShapes[name] = d.shape }
    }
    for name in descriptor.outputNames {
        if case .ndArray(let d) = descriptor.outputDescriptor(of: name) { outputShapes[name] = d.shape }
    }
    for name in descriptor.stateNames {
        if case .ndArray(let d) = descriptor.stateDescriptor(of: name) { stateShapes[name] = d.shape }
    }
    emit("descriptor", ["structure": prepared.structure.description, "functions": prepared.model.functionNames,
                        "inputs": inputShapes, "outputs": outputShapes, "states": stateShapes])
    let query = inputShapes["input_ids"] ?? inputShapes["in_new_token_ids"] ?? []
    let logits = outputShapes[descriptor.outputNames[0]] ?? []
    try require(query.count >= 2 && logits.count >= 2, "missing sequence dimension")
    try require(isS1 ? (query[1] == 1 && logits[1] == 1) : (query[1] < 0 && logits[1] < 0),
                "fixture does not match requested shape case")

    emit("constructor_begin")
    let runner = CoreAIRunner(bundle: bundle, variant: "coreai-pipelined")
    let engine = try await runner.makeInferenceEngine()
    emit("constructor_pass", ["engine": String(describing: type(of: engine)), "processed": engine.processedTokenCount])
    let tokenizer = try await bundle.loadTokenizer()
    let promptText = "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    let prompt = tokenizer.encode(text: promptText, addSpecialTokens: false).map(Int32.init)
    let budget = 24
    try require(prompt.count > 1, "prompt must test multi-token prefill")

    func generate(_ phase: String, _ tokens: [Int32]) async throws -> [Int32] {
        emit("generation_begin", ["phase": phase, "input_ids": tokens, "prompt_tokens": tokens.count,
                                   "processed_before": engine.processedTokenCount, "max_tokens": budget])
        let sequence = try await engine.generate(with: tokens, samplingConfiguration: .greedy,
                                                inferenceOptions: InferenceOptions(maxTokens: budget))
        var ids: [Int32] = []
        for try await output in sequence { ids.append(output.tokenId) }
        let decoded = tokenizer.decode(tokens: ids.map(Int.init))
        emit("generation_end", ["phase": phase, "ids": ids, "text": decoded, "tokens": ids.count,
                                 "prefix_hit": engine.lastPrefixHitCount, "processed": engine.processedTokenCount])
        let expectedProcessed = tokens.count + ids.count - 1
        if engine.processedTokenCount != expectedProcessed {
            failures.append(phase + "_processed_count")
            emit("processed_count_fail", ["phase": phase, "expected": expectedProcessed,
                                          "actual": engine.processedTokenCount])
        }
        try require(ids.count == budget, "incomplete generation in \(phase)")
        try require(ids.allSatisfy { $0 >= 0 && $0 < bundle.vocabSize }, "out-of-vocabulary output in \(phase)")
        try require(!decoded.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "empty text in \(phase)")
        return ids
    }

    let first = try await generate("turn1", prompt)
    let suffix = tokenizer.encode(text: "<|im_end|>\n<|im_start|>user\nName one famous landmark there.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", addSpecialTokens: false).map(Int32.init)
    let history = prompt + first + suffix
    let second = try await generate("turn2_continuation", history)
    try require(engine.lastPrefixHitCount > 0, "no shared-prefix match on the continuation")
    // A prefix match alone is not proof of retained recurrent state. Compare with a full replay.
    try await engine.reset(to: 0)
    emit("full_reset", ["processed": engine.processedTokenCount])
    try require(engine.processedTokenCount == 0, "full reset did not clear processed count")
    let secondReplay = try await generate("turn2_full_replay", history)
    parity("continuation_parity", second, secondReplay)
    try await engine.reset(to: 0)
    let secondReplayAgain = try await generate("turn2_full_replay_repeat", history)
    parity("full_replay_repeatability", secondReplayAgain, secondReplay)

    let target = max(1, engine.processedTokenCount - 2)
    if isS1 {
        var rejected = false
        do { try await engine.reset(to: target) }
        catch {
            rejected = String(describing: error).contains("Partial reset is not supported")
            emit("partial_reset_rejected", ["message": String(describing: error), "target": target])
        }
        try require(rejected, "hybrid partial rewind did not reject as documented")
    } else {
        try await engine.reset(to: target)
        emit("partial_reset_pass", ["target": target, "processed": engine.processedTokenCount])
        try require(engine.processedTokenCount == target, "dynamic partial reset count mismatch")
        let extended = history + secondReplay + suffix
        let rewound = try await generate("dynamic_after_partial_reset", extended)
        try await engine.reset(to: 0)
        let replay = try await generate("dynamic_partial_reset_reference", extended)
        parity("partial_reset_parity", rewound, replay)
    }

    try await engine.reset(to: 0)
    let repeated = try await generate("turn1_after_reset", prompt)
    parity("full_reset_repeatability", repeated, first)
    try require(failures.isEmpty, "failed checks: " + failures.joined(separator: ", "))
    emit("gate_pass", ["case": args[2], "budget": budget, "prompt_tokens": prompt.count])
} catch {
    emit("gate_fail", ["message": String(describing: error)])
    exit(2)
}
