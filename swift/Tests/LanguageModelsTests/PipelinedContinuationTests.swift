// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import Foundation
import Testing

/// Opt-in integration coverage for a local, platform-compatible language bundle.
/// Run under the host's shared GPU lock with COREAI_CONTINUATION_BUNDLE set.
@Suite(
    "Pipelined continuation state", .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_CONTINUATION_BUNDLE"] != nil)
)
struct PipelinedContinuationTests {
    @Test("Continuation consumes the final sampled token", arguments: [1, 4])
    func continuationConsumesPendingToken(budget: Int) async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_CONTINUATION_BUNDLE"])
        let bundle = try LanguageBundle(at: URL(fileURLWithPath: path))
        let engine = try await CoreAIRunner(bundle: bundle, variant: "coreai-pipelined")
            .makeInferenceEngine()

        func generate(_ input: [Int32]) async throws -> [Int32] {
            let sequence = try await engine.generate(
                with: input, samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: budget))
            var output: [Int32] = []
            for try await token in sequence { output.append(token.tokenId) }
            #expect(output.count == budget)
            // The last output token is sampled, but has not yet been consumed.
            #expect(engine.processedTokenCount == input.count + output.count - 1)
            return output
        }

        let prompt: [Int32] = [42, 43, 44]
        let first = try await generate(prompt)
        let cachedCount = engine.processedTokenCount
        let secondInput = prompt + first + [45, 46]
        let second = try await generate(secondInput)
        #expect(engine.lastPrefixHitCount == cachedCount)

        // A third turn catches duplicate pending tokens left in history.
        let thirdInput = secondInput + second + [47, 48]
        let third = try await generate(thirdInput)
        try await engine.reset(to: 0)
        let thirdReplay = try await generate(thirdInput)
        #expect(third == thirdReplay)

        try await engine.reset(to: 0)
        let secondReplay = try await generate(secondInput)
        #expect(second == secondReplay)

        // Exact continuation still needs to consume the last sampled token,
        // even when the caller adds no suffix.
        let exactInput = secondInput + secondReplay
        let exact = try await generate(exactInput)
        try await engine.reset(to: 0)
        let exactReplay = try await generate(exactInput)
        #expect(exact == exactReplay)
    }
}
