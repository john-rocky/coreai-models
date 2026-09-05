# 2026-09-05 follow-up: official S=1 consumer migration and continuation fix

Apple #227 was merged at 2026-09-04 22:54:05 UTC and #212 closed. The current
official commit is `27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60`; its source tree
is identical to the previously validated PR head `d237ee3e…`. The original
constructor comparison remains in the parent validation lane. This follow-up
uses the merged official commit and addresses the remaining consumer checks.

## Results

| Check | Official merged commit | Pending-token patch |
| --- | --- | --- |
| 0.8B S=1 constructor and 24-token generation | Pass | Pass |
| 27B S=1 constructor and 24-token generation | Pass | Pass |
| Continued vs replayed processed count, all three fixtures | 90 vs 91 | 91 vs 91 |
| 27B continued vs replayed greedy output | Different | Token-identical |
| 0.8B/dynamic continuation output | Token-identical despite count error | Token-identical, count correct |
| Reset/replay and repeated first turn | Pass | Pass |
| Dynamic partial reset/replay | Pass | Pass |
| Hybrid partial reset | Explicitly rejected | Explicitly rejected |
| New integration test, budgets 1 and 4 | Fails on 0.8B and dynamic | Passes on both |

All seven dynamic generation phases are token-identical between current main
and the patch. The count discrepancy also existed on the pre-#227 dynamic
control; this is a separate continuation bug, not attributed to #227.

The four-line patch truncates history to `processedTokenCount` before prefix
resolution. The last sampled output token has not been consumed by the model,
so it must be included in the next prefill. Commit:
`48281854f9834cdcc61b9cb320a960835eb1e8f2` (source plus regression test).
The patch was tested in an isolated worktree; shared source checkouts were
not switched or edited.

## Consumer migration

Based on pipette PR #13 at `03ae30302ea2f648259c1e6f10370c47fc0a0b7f`, commit
`f72adf6c18be6fe129dc4b6a02d2df191ea68e29` pins official Apple `27a66f90…`,
removes the sidecar's forced chunk threshold, and aligns the four Rust
runtime identity fields with the committed Swift lockfile. The selected
`swift-transformers` 1.2.0, `xgrammar` 0.2.2 and `swift-jinja` 2.3.2 match the
Apple lockfile used for the real-bundle comparison. All 12 shared dependency
pins in the 13-pin consumer lockfile match the 31-pin Apple lockfile.

Both S=1 fixtures pass health, prefill (32 prompt tokens), decode (32/16),
end-to-end (32/16), memory-contract (8/1), and repeated decode HTTP requests.
The HTTP memory check validates token counts, not a memory measurement.

The Rust CLI itself also builds/caches the official sidecar without a
prebuilt-binary override. Cached and built binaries have identical SHA-256
(`pipette-cli-memory-result.json`). The standard `max_memory_usage_smoke`
benchmark runs on both fixtures, generates one token from an eight-token
prompt, shuts down normally, and saves two local result records. The
selected records are in `cli-results.json`; workspace identity files remain
local. No management-server sync was performed.

CLI decode timing was attempted: the per-command readiness flag is not
routed for this Core AI cell, so the documented environment override was
used. Its 90-second readiness gate then timed out due to shared host CPU
load. No threshold was relaxed. These attempts are preserved; there is no
claim of a successful readiness-gated CLI throughput measurement.

Validation: conventions, `cargo fmt --all --check`, workspace clippy,
workspace nextest (2,033 passed / 7 skipped), and workspace doctests pass.
The first Rust attempt caught an existing lockfile test's version-only
assumption after changing to a commit pin; it now verifies both repository
URLs and version-or-revision values. Import grouping in the touched sidecar
module was corrected. Other PR #13 review items are outside this change.

## Provenance and reproduction

Mac Studio `Mac16,9`, M4 Max, 128 GiB; macOS 27.0 `26A5416b`; Xcode 27 beta 5
`27A5237l`; macOS SDK 27.0 `26A5406c`; Swift 6.4. The same shared GPU lock
wrapper was used for every inference process. See `../environment.json`,
`../hardware.json`, `artifacts.json`, the pinned bundle manifests, and
`../logs/followup-*.{log,result.json}` for commands, outputs and hashes.
The first successful patch result was saved as `FIRST_FIX_RESULT.md` before
publication.

Bundle revisions are unchanged from the original lane:
- Qwen3.5 0.8B: `df983f659f8501f3da0892baca4c39523594716b`.
- Qwen3.8 27B: `2f66df29215a6fa7ce59d320225dcaa8cc4f15c2`.
- Dynamic Qwen3 0.6B: `7ecc49f36eb49415a0d1888d6de5e1f5bc79a113`.

The public review branch includes a portable consumer test package and full
technical evidence. The 24-token gate intentionally does not stop at EOS.
These checks do not establish all cancellation/early-EOS behavior, long
conversations, other bundles, iPhone execution, a release-OS result, or that
all existing Apple/pipette review requests have been resolved.

## Remaining timing command

After the shared host meets pipette's normal load/thermal criteria, acquire
the shared GPU lock and run the command saved in
`../logs/followup-cli-v2-decode-s1.result.json` (or use
`PIPETTE_READINESS_MAX_WAIT_SECS=420` for a longer wait). The consumer helper
`pipette_cli_smoke.py` contains the exact invocation. This environmental
limitation does not block submitting the verified fix and migration.

