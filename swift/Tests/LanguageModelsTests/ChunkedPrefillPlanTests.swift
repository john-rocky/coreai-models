// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("ChunkedPrefillPlan")
struct ChunkedPrefillPlanTests {
    private func plan(
        _ tokenCount: Int, start: Int = 0, chunk: Int = 64, padTailMinimum: Int? = nil,
        capacity: Int = 4096
    ) -> ChunkedPrefillPlan {
        ChunkedPrefillPlan(
            tokenCount: tokenCount, start: start, chunkLength: chunk,
            padTailMinimum: padTailMinimum, capacity: capacity)
    }

    @Test("without padding: full chunks while one more chunk and the sampled token fit, then S=1")
    func unpadded() {
        #expect(plan(0) == ChunkedPrefillPlan(fullChunks: 0, paddedTokens: 0, singleSteps: 0))
        #expect(plan(1) == ChunkedPrefillPlan(fullChunks: 0, paddedTokens: 0, singleSteps: 0))
        #expect(plan(64) == ChunkedPrefillPlan(fullChunks: 0, paddedTokens: 0, singleSteps: 63))
        #expect(plan(65) == ChunkedPrefillPlan(fullChunks: 1, paddedTokens: 0, singleSteps: 0))
        #expect(plan(200) == ChunkedPrefillPlan(fullChunks: 3, paddedTokens: 0, singleSteps: 7))
        #expect(plan(1000) == ChunkedPrefillPlan(fullChunks: 15, paddedTokens: 0, singleSteps: 39))
    }

    @Test("every token but the last is fed, whatever the plan")
    func feedsAllButLast() {
        for count in 1...300 {
            for minimum in [nil, 1, 4, 63] as [Int?] {
                let p = plan(count, padTailMinimum: minimum)
                #expect(p.fullChunks * 64 + p.paddedTokens + p.singleSteps == count - 1)
                #expect(p.paddedTokens == 0 || p.singleSteps == 0)
            }
        }
    }

    @Test("with padding: the tail rides one padded chunk from the minimum on")
    func padded() {
        #expect(plan(1000, padTailMinimum: 4) == ChunkedPrefillPlan(fullChunks: 15, paddedTokens: 39, singleSteps: 0))
        #expect(plan(200, padTailMinimum: 4) == ChunkedPrefillPlan(fullChunks: 3, paddedTokens: 7, singleSteps: 0))
        #expect(plan(15, padTailMinimum: 4) == ChunkedPrefillPlan(fullChunks: 0, paddedTokens: 14, singleSteps: 0))
        #expect(plan(64, padTailMinimum: 4) == ChunkedPrefillPlan(fullChunks: 0, paddedTokens: 63, singleSteps: 0))
        // below the minimum, S=1
        #expect(plan(68, padTailMinimum: 4) == ChunkedPrefillPlan(fullChunks: 1, paddedTokens: 0, singleSteps: 3))
        #expect(plan(69, padTailMinimum: 4) == ChunkedPrefillPlan(fullChunks: 1, paddedTokens: 4, singleSteps: 0))
        // a full last chunk needs no pad
        #expect(plan(129, padTailMinimum: 1) == ChunkedPrefillPlan(fullChunks: 2, paddedTokens: 0, singleSteps: 0))
    }

    @Test("a padded chunk needs KV room for a whole chunk past its start, else S=1")
    func capacity() {
        // 15 chunks from 0 end at 960; the padded one writes 960..<1024
        #expect(plan(1000, padTailMinimum: 4, capacity: 1024).paddedTokens == 39)
        #expect(plan(1000, padTailMinimum: 4, capacity: 1023) == ChunkedPrefillPlan(fullChunks: 15, paddedTokens: 0, singleSteps: 39))
        // the start counts: 30 tokens from position 2000 pad up to 2064
        #expect(plan(31, start: 2000, padTailMinimum: 4, capacity: 2064).paddedTokens == 30)
        #expect(plan(31, start: 2000, padTailMinimum: 4, capacity: 2063).singleSteps == 30)
    }
}
