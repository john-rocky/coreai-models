// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Prompt-lookup (n-gram) drafting — the training-free drafter.
//
// Take the last `n` tokens of the context, find an EARLIER occurrence of that same
// n-gram, and propose the tokens that followed it. No model, no weights, ~microseconds
// per round. It pays exactly where the continuation is already present in the context —
// tool calls that echo argument names, code that repeats identifiers, RAG answers that
// quote the source, a reasoning channel that restates the question — and it degenerates
// to a no-op (empty proposal → a plain S=1 decode step) on free-form prose. That is a
// property of the method, not a defect.

import Foundation

struct NgramDrafter {
    /// Longest suffix length tried first; longer match = higher precision.
    var maxNgram = 3
    /// Shortest suffix accepted. 1 proposes off a single-token match (noisy but free).
    var minNgram = 1
    /// Prefer the most recent earlier occurrence over the first one.
    var preferRecent = true

    /// Propose up to `k` continuation tokens for `context`. Empty = no match.
    func propose(context: [Int32], k: Int) -> [Int32] {
        guard k > 0, context.count >= 2 else { return [] }
        let upper = min(maxNgram, context.count - 1)
        guard upper >= minNgram else { return [] }

        for n in stride(from: upper, through: minNgram, by: -1) {
            let patternStart = context.count - n
            // Candidate starts: every position whose n-gram fits strictly before the suffix,
            // and which has at least one token after it to propose.
            let lastCandidate = patternStart - 1
            guard lastCandidate >= 0 else { continue }

            let candidates = preferRecent
                ? stride(from: lastCandidate, through: 0, by: -1)
                : stride(from: 0, through: lastCandidate, by: 1)
            for start in candidates {
                var matched = true
                for j in 0..<n where context[start + j] != context[patternStart + j] {
                    matched = false
                    break
                }
                guard matched else { continue }
                let from = start + n
                let to = min(from + k, context.count)
                guard from < to else { continue }
                return Array(context[from..<to])
            }
        }
        return []
    }
}
