// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

@Suite("VanillaDecodingStrategy stop sequences")
struct VanillaDecodingStopTests {
    private func decode(
        engine: MockEngine, maxTokens: Int, stopSequences: [[Int32]]
    ) async throws -> [Int32] {
        let tokenizerURL = try #require(TestResources.url(for: "MinimalTokenizer"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let sequence = try await VanillaDecodingStrategy().decode(
            from: .tokens([1, 2, 3]),
            tokenizer: tokenizer,
            inferenceEngine: engine,
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: maxTokens),
            stopSequences: StopSequences(sequences: stopSequences)
        )
        var tokens: [Int32] = []
        for try await result in sequence {
            tokens.append(result.tokenId)
        }
        return tokens
    }

    @Test("A stop sequence leaves the rest of the token budget unused")
    func stopSequenceEndsGeneration() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50])
        _ = try await decode(engine: engine, maxTokens: 100, stopSequences: [[30]])

        #expect(engine.inferenceCallCount == 3)
        #expect(engine.processedTokenCount == 3 + 3)
    }

    @Test("A multi-token stop sequence ends generation at its last token")
    func multiTokenStopSequenceEndsGeneration() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50])
        _ = try await decode(engine: engine, maxTokens: 100, stopSequences: [[20, 30]])

        #expect(engine.inferenceCallCount == 3)
    }

    @Test("Without a match the budget still bounds generation")
    func budgetEndsGeneration() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50])
        _ = try await decode(engine: engine, maxTokens: 7, stopSequences: [[99]])

        #expect(engine.inferenceCallCount == 7)
    }
}
