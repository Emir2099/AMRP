"""
script6_additive_beta_sweep.py
==============================
Pre-registered beta fairness sweep for the additive-gating ablation.

Protocol (written before any results seen):
  Phase 1 -- Coarse sweep: beta in {0.1, 0.25, 0.5, 1.0, 2.0, 4.0}
             10 Hz noise, n=5 seeds, both baselines (STDP/AMRP) run once.
  Phase 2 -- Adaptive refinement:
             * If coarse winner is INTERIOR: add 2 geometric-mean points
               between winner and each immediate coarse neighbour.
             * If coarse winner is at the LOW EDGE (0.1): extend down
               to beta=0.03, 0.05 and rerun.
             * If coarse winner is at the HIGH EDGE (4.0): extend up
               to beta=6.0, 8.0 and rerun.
             If still on boundary after extension, report that result
             honestly (same language as Sec 6.3/Sec 8 for tau_astro).
  Outcome  -- Lock in optimal beta, save to amrp_output/script6_beta_sweep_output.txt.

Goldilocks criterion (unchanged from Experiment C):
  G = sep_norm - 0.6 * nvm_norm
where both quantities are normalised to [0,1] across the full comparison
set (STDP + AMRP + all tested beta values).
"""

import sys
import os
import numpy as np
from brian2 import Hz

# Ensure parent directory is on path when run from workspace root
sys.path.insert(0, os.path.dirname(__file__))
import amrp_simulation as sim

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), 'amrp_output', 'script6_beta_sweep_output.txt'
)

# ── Pre-registered constants ──────────────────────────────────────────────────
SEEDS          = [42, 123, 456, 789, 101]   # n=5 seeds
NOISE_HZ       = 10                          # challenging noise level
BETAS_COARSE   = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
ALPHA_GOLDILOCKS = 0.6                       # eq. (8), Sec 5.5 -- not a new constant

# stdout encoding fix for Windows cp1252
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_n(noise_hz, n_seeds, use_amrp=False, use_additive=False, beta=1.0):
    """Run n_seeds trials and return (nvm_list, sep_list)."""
    nvm_list, sep_list = [], []
    for s in n_seeds:
        r = sim.run_experiment(
            noise_hz * Hz,
            use_amrp     = use_amrp,
            use_additive = use_additive,
            beta         = beta,
            seed_val     = s,
            record_traces= False,
        )
        nvm_list.append(r['nvm_wear'])
        sep_list.append(r['signal_separation'])
    return nvm_list, sep_list


def goldilocks_scores(all_means):
    """
    Normalise and compute G = sep_norm - 0.6*nvm_norm across the full
    comparison set.  all_means: list of (label, nvm_mean, sep_mean).
    Returns list of (label, G).
    """
    nvms = [x[1] for x in all_means]
    seps = [x[2] for x in all_means]
    nvm_min, nvm_max = min(nvms), max(nvms)
    sep_min, sep_max = min(seps), max(seps)
    denom_nvm = nvm_max - nvm_min if nvm_max > nvm_min else 1e-9
    denom_sep = sep_max - sep_min if sep_max > sep_min else 1e-9

    scores = []
    for label, nvm, sep in all_means:
        nvm_norm = (nvm - nvm_min) / denom_nvm
        sep_norm = (sep - sep_min) / denom_sep
        g = sep_norm - ALPHA_GOLDILOCKS * nvm_norm
        scores.append((label, g, nvm, sep))
    return scores


def geom_interp(a, b, n=2):
    """Return n geometric-mean-spaced values strictly between a and b."""
    log_a, log_b = np.log(a), np.log(b)
    return [float(np.exp(log_a + (log_b - log_a) * k / (n + 1)))
            for k in range(1, n + 1)]


# ── Phase 1 — Coarse sweep ────────────────────────────────────────────────────

def phase1(log):
    log("=" * 70)
    log("PHASE 1: Coarse beta sweep")
    log(f"  Noise: {NOISE_HZ} Hz | Seeds: {SEEDS} | n={len(SEEDS)}")
    log(f"  beta grid (coarse): {BETAS_COARSE}")
    log("=" * 70)

    # Baselines
    log("\n--- Baselines ---")
    log("  STDP  ...", end="")
    stdp_nvm, stdp_sep = run_n(NOISE_HZ, SEEDS, use_amrp=False, use_additive=False)
    log(f"  NVM={np.mean(stdp_nvm):.4f}+/-{np.std(stdp_nvm):.4f}"
        f"  sep={np.mean(stdp_sep):.4f}+/-{np.std(stdp_sep):.4f}")

    log("  AMRP  ...", end="")
    amrp_nvm, amrp_sep = run_n(NOISE_HZ, SEEDS, use_amrp=True, use_additive=False)
    log(f"  NVM={np.mean(amrp_nvm):.4f}+/-{np.std(amrp_nvm):.4f}"
        f"  sep={np.mean(amrp_sep):.4f}+/-{np.std(amrp_sep):.4f}")

    # Additive sweep
    log("\n--- Additive ablation coarse sweep ---")
    add_results = {}  # beta -> {'nvm': list, 'sep': list}
    for beta in BETAS_COARSE:
        log(f"  b={beta:.2f} ...", end="")
        nvm_l, sep_l = run_n(NOISE_HZ, SEEDS, use_amrp=False,
                              use_additive=True, beta=beta)
        add_results[beta] = {'nvm': nvm_l, 'sep': sep_l}
        log(f"  NVM={np.mean(nvm_l):.4f}+/-{np.std(nvm_l):.4f}"
            f"  sep={np.mean(sep_l):.4f}+/-{np.std(sep_l):.4f}")

    # Goldilocks across full set (STDP + AMRP + all betas)
    all_means = [
        ('STDP', np.mean(stdp_nvm), np.mean(stdp_sep)),
        ('AMRP', np.mean(amrp_nvm), np.mean(amrp_sep)),
    ] + [(f'add(b={b:.2f})', np.mean(add_results[b]['nvm']),
                              np.mean(add_results[b]['sep']))
         for b in BETAS_COARSE]

    scores = goldilocks_scores(all_means)
    log("\n--- Goldilocks scores (G = sep_norm - 0.6*nvm_norm, eq. 8) ---")
    for label, g, nvm, sep in sorted(scores, key=lambda x: -x[1]):
        log(f"  {label:<20s}  G={g:+.4f}  NVM={nvm:.4f}  sep={sep:.4f}")

    # Identify coarse winner among additive conditions only
    add_score_list = []
    for beta in BETAS_COARSE:
        lbl = f'add(b={beta:.2f})'
        for label, g, nvm, sep in scores:
            if label == lbl:
                add_score_list.append((beta, g))
    best_beta, best_g = max(add_score_list, key=lambda x: x[1])
    log(f"\n  Coarse winner: b={best_beta} (G={best_g:+.4f})")

    return stdp_nvm, stdp_sep, amrp_nvm, amrp_sep, add_results, best_beta


# ── Phase 2 — Adaptive refinement ────────────────────────────────────────────

def phase2(log, stdp_nvm, stdp_sep, amrp_nvm, amrp_sep, add_results, best_beta):
    log("\n" + "=" * 70)
    log("PHASE 2: Adaptive refinement")

    beta_min_tested = BETAS_COARSE[0]
    beta_max_tested = BETAS_COARSE[-1]
    at_low  = abs(best_beta - beta_min_tested) < 1e-9
    at_high = abs(best_beta - beta_max_tested) < 1e-9

    if at_low:
        refine_betas = [0.03, 0.05]
        log(f"  Coarse winner b={best_beta} is at LOW EDGE -- extending range down.")
        log(f"  New candidates: {refine_betas}")
    elif at_high:
        refine_betas = [6.0, 8.0]
        log(f"  Coarse winner b={best_beta} is at HIGH EDGE -- extending range up.")
        log(f"  New candidates: {refine_betas}")
    else:
        # Interior: find coarse neighbours
        idx = BETAS_COARSE.index(best_beta)
        left  = BETAS_COARSE[idx - 1]
        right = BETAS_COARSE[idx + 1]
        refine_betas = geom_interp(left, best_beta) + geom_interp(best_beta, right)
        refine_betas = sorted(set(round(b, 4) for b in refine_betas))
        log(f"  Coarse winner b={best_beta} is interior (neighbours: {left}, {right}).")
        log(f"  Geometric-mean refinement candidates: {refine_betas}")

    log("\n--- Refinement sweep ---")
    for beta in refine_betas:
        log(f"  b={beta:.4f} ...", end="")
        nvm_l, sep_l = run_n(NOISE_HZ, SEEDS, use_amrp=False,
                              use_additive=True, beta=beta)
        add_results[beta] = {'nvm': nvm_l, 'sep': sep_l}
        log(f"  NVM={np.mean(nvm_l):.4f}+/-{np.std(nvm_l):.4f}"
            f"  sep={np.mean(sep_l):.4f}+/-{np.std(sep_l):.4f}")

    # Recompute Goldilocks over expanded set
    all_betas_tested = sorted(add_results.keys())
    all_means = [
        ('STDP', np.mean(stdp_nvm), np.mean(stdp_sep)),
        ('AMRP', np.mean(amrp_nvm), np.mean(amrp_sep)),
    ] + [(f'add(b={b:.4f})', np.mean(add_results[b]['nvm']),
                               np.mean(add_results[b]['sep']))
         for b in all_betas_tested]

    scores = goldilocks_scores(all_means)
    log("\n--- Goldilocks scores (refined, full set) ---")
    for label, g, nvm, sep in sorted(scores, key=lambda x: -x[1]):
        log(f"  {label:<24s}  G={g:+.4f}  NVM={nvm:.4f}  sep={sep:.4f}")

    # Identify refined winner
    add_score_list2 = []
    for beta in all_betas_tested:
        lbl = f'add(b={beta:.4f})'
        for label, g, nvm, sep in scores:
            if label == lbl:
                add_score_list2.append((beta, g))
    final_beta, final_g = max(add_score_list2, key=lambda x: x[1])

    # Check if still at boundary
    all_betas_sorted = sorted(all_betas_tested)
    still_at_boundary = (
        abs(final_beta - all_betas_sorted[0]) < 1e-9 or
        abs(final_beta - all_betas_sorted[-1]) < 1e-9
    )

    log(f"\n  Refined winner: b={final_beta} (G={final_g:+.4f})")
    if still_at_boundary:
        log(
            f"\n  NOTE (to include in manuscript Sec 6.4): b={final_beta} is the best value"
            f" in the tested range but is NOT a bracketed interior optimum -- the"
            f" sensitivity surface may extend further in this direction."
            f" This limitation is reported consistently with Sec 6.3/Sec 8 (tau_astro boundary)."
        )
    else:
        log(f"\n  b={final_beta} is bracketed -- interior optimum confirmed.")

    return final_beta, final_g, still_at_boundary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    lines = []

    def log(msg, end="\n"):
        print(msg, end=end, flush=True)
        lines.append(msg + end)

    log("script6_additive_beta_sweep.py")
    log(f"Pre-registered protocol: G = sep_norm - {ALPHA_GOLDILOCKS}*nvm_norm (eq. 8)")
    log(f"Noise: {NOISE_HZ} Hz | Seeds: {SEEDS}")

    (stdp_nvm, stdp_sep,
     amrp_nvm, amrp_sep,
     add_results, best_beta_coarse) = phase1(log)

    final_beta, final_g, boundary_flag = phase2(
        log, stdp_nvm, stdp_sep, amrp_nvm, amrp_sep, add_results, best_beta_coarse
    )

    log("\n" + "=" * 70)
    log("LOCKED BETA FOR SCRIPT7")
    log(f"  OPTIMAL_BETA = {final_beta}")
    log(f"  G_score      = {final_g:+.4f}")
    log(f"  Boundary flag: {boundary_flag}")
    log("=" * 70)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nOutput saved → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
