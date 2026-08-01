# Your Neoverse says SVE2. Your matmul runs `neon_i8mm`.

An ISA dispatch audit for Arm — reproducible on a free CI runner in about two minutes.

**Arm Create: AI Optimization Challenge — Cloud AI track.**

---

## The finding

Neoverse N2 and Azure Cobalt 100 advertise `sve sve2 svei8mm svebf16`. Read `/proc/cpuinfo` on
one and you will believe the SVE GEMM kernels are doing your matrix multiplies.

They are not. KleidiAI gates every SVE kernel behind a runtime check for a **256-bit** vector:

```cpp
// llama.cpp — ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
if (svcntb() == 32) { features |= CPU_FEATURE_SVE; }
```

`svcntb()` returns the SVE vector length **in bytes**, so `32` means 256-bit. N2 and Cobalt are
**128-bit** parts. `svcntb()` returns `16`, `CPU_FEATURE_SVE` is never set, and every `*_sve_*`
kernel in the dispatch table is **unreachable dead code** on the most widely-available Neoverse
silicon — including the free GitHub runner every developer can reach.

Measured on `ubuntu-24.04-arm`, not asserted:

```
SVE present : yes
svcntb()    : 16 bytes (128-bit vector)

FEATURE              ADVERTISED   KLEIDIAI     NOTE
dotprod (i8 dot)     yes          USES         NEON path, always reachable
i8mm    (int8 mm)    yes          USES         neon_i8mm kernels — the real path
bf16                 yes          USES         bf16 kernels where present
SVE / SVE2           yes          refuses      ADVERTISED BUT UNREACHABLE — kleidiai.cpp:209 needs svcntb()==32

=== VERDICT ===
This core advertises SVE, but KleidiAI will NOT dispatch a single SVE kernel.
Required: svcntb() == 32 (256-bit).  Actual: 16 (128-bit).
Every *_sve_* GEMM kernel in the dispatch table is dead code here.
Your matmul is running the neon_i8mm path.
```

Exit code `2` means *advertised but refused*, so this drops straight into CI as a gate.

## Reproduce it yourself

**On a free Neoverse runner (no hardware, no cloud spend):**

1. Fork this repo.
2. Actions → **arm-dispatch-audit** → **Run workflow**.
3. Read the *"The finding — advertised vs dispatched"* step.

The workflow also clones upstream llama.cpp and greps `svcntb` out of the real source, so the
threshold is read from the code rather than taken from this README.

**Locally, on any Arm box:**

```bash
cc -O2 -march=armv8.2-a+sve -DHAVE_SVE -o isa-probe-sve src/isa-probe.c
./isa-probe-sve
```

## Honesty guardrails

An Arm engineer is going to check these, so they are stated up front:

- **The gate is KleidiAI's, not ggml's.** ggml proper *does* use SVE elsewhere (`vec.cpp`,
  `simd-gemm.h`). The claim is scoped to KleidiAI's kernel dispatch. Saying "ggml ignores SVE"
  would be wrong.
- **The SVE kernels exist.** There is a `..._16x8_sve_i8mm` GEMM kernel in the table. The claim is
  that it is *unreachable on 128-bit cores*, not that it is absent.
- **The portable build refuses to guess.** `svcntb()` is an SVE intrinsic, so a build without
  `+sve` cannot read the vector length. Rather than print a fabricated `0`, that build reports
  `unknown`, prints the rebuild command, and exits `3`. Only the `+sve` build renders a verdict.
  (This was a real bug caught by CI on the first run — the portable build had been printing
  `Actual: 0 (0-bit)`.)
- **Apple Silicon has no SVE at all.** On an M2 the probe exits `0` with "the NEON/i8mm path is
  the only path" — which is why tutorial numbers from SVE-capable hardware transfer nowhere.

## Why this matters

Every "deploy an LLM on Arm with llama.cpp and KleidiAI" guide tells you to look for SVE2 in
`/proc/cpuinfo`. On the server parts most people can actually rent — and on the CI runner they get
for free — that flag is telling you about a code path your process will never enter. The
optimization you think you enabled is not running, and nothing warns you.

This repo turns that from folklore into a two-minute check anyone can run against their own
hardware.

## What's here

| Path | What it is |
|---|---|
| `src/isa-probe.c` | The probe. Advertised-vs-dispatched, no dependencies, distinct exit codes. |
| `.github/workflows/audit.yml` | Fork → Run workflow → same numbers on your own free N2 runner. |

## Licence

Apache-2.0.
