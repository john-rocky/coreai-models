# Pipelined continuation reproduction (2026-09-05)

Apple PR #227 is merged at `27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60`.
It fixes real S=1 constructor/prefill failures. A separate continuation bug
remains: history includes the last sampled token even though it has not yet
been processed. Reusing that prefix skips the token and shifts subsequent
positions by one. This fork's four-line patch limits reusable history to
`processedTokenCount`.

On a Mac Studio M4 Max (128 GiB), macOS 27.0 `26A5416b`, Xcode 27 beta 5
`27A5237l`, macOS SDK 27.0 `26A5406c`, current main fails the continuation
count check for all three fixtures below; the patch passes all three. The
27B greedy continuation also changes tokens on main and becomes identical
to full replay with the patch. Dynamic first-turn/reset/partial-reset
outputs remain token-identical. This is a bounded real-bundle check, not a
claim covering every model, cancellation path, iPhone, or release OS.

| Fixture | Immutable Hugging Face revision |
| --- | --- |
| `mlboydaisuke/qwen3.5-0.8B-CoreAI`, `gpu-pipelined-b2/qwen3_5_0_8b_decode_int8hu_block32_sym` | `df983f659f8501f3da0892baca4c39523594716b` |
| `mlboydaisuke/Qwen3.8-27B-CoreAI`, `gpu-pipelined/qwen3_8_27b_decode_int4lin` | `2f66df29215a6fa7ce59d320225dcaa8cc4f15c2` |
| `mlboydaisuke/qwen3-0.6b-CoreAI-official`, `macos` | `7ecc49f36eb49415a0d1888d6de5e1f5bc79a113` |

`evidence.tar.gz` contains commands, exact token outputs, result JSON, environment,
and file hashes. The fixed-budget gate intentionally continues through EOS.
Hybrid partial reset remains rejected; full reset and replay are tested.

## Reproduce the integration test

Use a platform-compatible macOS bundle; acquire your shared GPU lock before
running. Do not run alongside another GPU inference process. The test is
opt-in and requires no download when the bundle already exists.

```sh
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.5.app
export COREAI_CONTINUATION_BUNDLE=/absolute/path/to/bundle
unset COREAI_CHUNK_THRESHOLD
swift test -c release --filter PipelinedContinuationTests
```

For a smaller consumer-only build, use `consumer/` (no unrelated product or
test targets). Both commands run the same committed integration-test source:

```sh
swift test --package-path validation/2026-09-05-coreai/consumer -c release --filter PipelinedContinuationTests
```

The test checks budgets 1 and 4, two successive extensions, continuation
without a suffix, processed counts, and equality with full replay. It fails
on unpatched main and passes with the patch on the 0.8B S=1 and dynamic 0.6B
fixtures. The separate `bundle-gate` additionally exercises 24-token actual
text generation, repeatability and partial reset on the three fixtures.
