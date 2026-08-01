/*
 * isa-probe.c — what the CPU advertises vs what KleidiAI will actually dispatch.
 *
 * The whole point of this contest entry: /proc/cpuinfo on a Neoverse N2 says
 * "sve sve2 svei8mm svebf16", so every tutorial assumes the SVE GEMM kernels
 * are in play. They are not. KleidiAI's runtime gate requires a 256-bit vector:
 *
 *     // kleidiai.cpp:209
 *     if (svcntb() == 32) { features |= CPU_FEATURE_SVE; }
 *
 * svcntb() returns the SVE vector length in BYTES. N2 and Cobalt are 128-bit,
 * so svcntb() == 16, the flag is never set, and every SVE kernel in the table
 * is unreachable dead code on the most widely-available Neoverse silicon.
 *
 * This probe prints both halves of that contradiction side by side so the claim
 * is checkable in one command instead of taken on trust.
 *
 * Build:  cc -O2 -o isa-probe isa-probe.c            (portable, no SVE needed)
 *         cc -O2 -march=armv8.2-a+sve -DHAVE_SVE ... (to actually call svcntb)
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(HAVE_SVE) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/* KleidiAI's threshold, in bytes. 32 bytes = 256-bit vector. */
#define KLEIDIAI_REQUIRED_SVCNTB 32

typedef struct {
    const char *name;
    int advertised;   /* CPU says it has it            */
    int usable;       /* KleidiAI will actually use it */
    const char *note;
} feature_t;

static int detect_hwcap(unsigned long bit, int which)
{
#if defined(__linux__)
    unsigned long caps = getauxval(which == 2 ? AT_HWCAP2 : AT_HWCAP);
    return (caps & bit) != 0;
#else
    (void)bit; (void)which;
    return 0;
#endif
}

#if defined(__APPLE__)
static int sysctl_flag(const char *key)
{
    int v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(key, &v, &len, NULL, 0) != 0) return 0;
    return v;
}
#endif

/* Vector length in bytes, or 0 when the core has no SVE at all. */
static unsigned long sve_vector_bytes(void)
{
#if defined(HAVE_SVE) && defined(__ARM_FEATURE_SVE)
    return (unsigned long)svcntb();
#else
    return 0;
#endif
}

int main(void)
{
    printf("=== Arm ISA dispatch audit ===\n\n");

#if defined(__APPLE__)
    char brand[128] = {0};
    size_t bl = sizeof(brand);
    sysctlbyname("machdep.cpu.brand_string", brand, &bl, NULL, 0);
    printf("host        : %s (Apple Silicon)\n", brand);
    int has_i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
    int has_bf16 = sysctl_flag("hw.optional.arm.FEAT_BF16");
    int has_dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    int has_sve = sysctl_flag("hw.optional.arm.FEAT_SVE");
#elif defined(__linux__)
    printf("host        : aarch64 Linux\n");
    int has_i8mm = detect_hwcap(HWCAP2_I8MM, 2);
    int has_bf16 = detect_hwcap(HWCAP2_BF16, 2);
    int has_dotprod = detect_hwcap(HWCAP_ASIMDDP, 1);
    int has_sve = detect_hwcap(HWCAP_SVE, 1);
#else
    printf("host        : non-Arm — this probe is a no-op here\n");
    int has_i8mm = 0, has_bf16 = 0, has_dotprod = 0, has_sve = 0;
#endif

    unsigned long vb = sve_vector_bytes();

    printf("SVE present : %s\n", has_sve ? "yes" : "no");
    if (has_sve) {
        if (vb) {
            printf("svcntb()    : %lu bytes (%lu-bit vector)\n", vb, vb * 8);
        } else {
            printf("svcntb()    : not measured — rebuild with -march=armv8.2-a+sve -DHAVE_SVE\n");
        }
    }
    printf("\n");

    /* The gate. This single comparison is the entire finding.
     *
     * Only the SVE build can answer it: svcntb() is an SVE intrinsic, so a
     * portable build has no way to read the vector length. Reporting "0 bits"
     * from that build would be a fabricated measurement, so the portable build
     * declines to render a verdict instead. */
    int can_measure = !has_sve || vb > 0;
    int kleidiai_sve = has_sve && vb == KLEIDIAI_REQUIRED_SVCNTB;

    feature_t feats[] = {
        { "dotprod (i8 dot)", has_dotprod, has_dotprod, "NEON path, always reachable" },
        { "i8mm    (int8 mm)", has_i8mm, has_i8mm, "neon_i8mm kernels — the real path" },
        { "bf16",              has_bf16, has_bf16, "bf16 kernels where present" },
        { "SVE / SVE2",        has_sve, kleidiai_sve,
          !can_measure
            ? "UNKNOWN — portable build cannot call svcntb(); rebuild with +sve"
            : (has_sve && !kleidiai_sve
                 ? "ADVERTISED BUT UNREACHABLE — kleidiai.cpp:209 needs svcntb()==32"
                 : (has_sve ? "reachable (256-bit vector)" : "absent on this core")) },
    };

    printf("%-20s %-12s %-12s %s\n", "FEATURE", "ADVERTISED", "KLEIDIAI", "NOTE");
    printf("%-20s %-12s %-12s %s\n", "-------", "----------", "--------", "----");
    for (size_t i = 0; i < sizeof(feats) / sizeof(feats[0]); i++) {
        int is_sve_row = (i == 3);
        printf("%-20s %-12s %-12s %s\n",
               feats[i].name,
               feats[i].advertised ? "yes" : "no",
               (is_sve_row && !can_measure) ? "unknown"
                                            : (feats[i].usable ? "USES" : "refuses"),
               feats[i].note);
    }

    printf("\n=== VERDICT ===\n");
    if (!can_measure) {
        /* Refuse to state a vector length we did not measure. */
        printf("This core advertises SVE, but this build cannot read the vector length.\n");
        printf("svcntb() is an SVE intrinsic and is absent from the portable build.\n");
        printf("Rebuild to settle it:\n");
        printf("  cc -O2 -march=armv8.2-a+sve -DHAVE_SVE -o isa-probe-sve src/isa-probe.c\n");
        return 3;   /* distinct exit code: "advertised, not measurable here" */
    }
    if (has_sve && !kleidiai_sve) {
        printf("This core advertises SVE, but KleidiAI will NOT dispatch a single SVE kernel.\n");
        printf("Required: svcntb() == %d (256-bit).  Actual: %lu (%lu-bit).\n",
               KLEIDIAI_REQUIRED_SVCNTB, vb, vb * 8);
        printf("Every *_sve_* GEMM kernel in the dispatch table is dead code here.\n");
        printf("Your matmul is running the neon_i8mm path.\n");
        return 2;   /* distinct exit code: "advertised but refused" */
    }
    if (!has_sve) {
        printf("No SVE on this core at all — the NEON/i8mm path is the only path.\n");
        printf("Benchmark numbers from SVE-capable tutorials do not transfer here.\n");
        return 0;
    }
    printf("SVE is present AND 256-bit: KleidiAI would dispatch SVE kernels on this core.\n");
    return 0;
}
