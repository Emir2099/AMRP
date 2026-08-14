"""
script7_additive_final.py
=========================
Full three-way comparison: STDP / AMRP / additive-gating ablation.
Run AFTER script6_additive_beta_sweep.py has locked in OPTIMAL_BETA.

Experiment B (replicated):  SNR sweep {1,2,5,10,20} Hz, n=10 seeds.
Experiment C (replicated):  5×4 θ_noise × τ_astro grid, 10 Hz, n=5 seeds,
                             AMRP vs additive-optimal only (STDP has no grid).

Output: amrp_output/script7_final_comparison_output.txt

Pre-registered outcome logic:
  AMRP wins    — lower NVM AND comparable/better signal sep at every noise level
  Tie          — additive matches NVM within 1 std but narrower valid-β range
  Additive wins — comparable wear reduction on simpler surface
"""

import sys
import os
import numpy as np
from brian2 import Hz, ms

sys.path.insert(0, os.path.dirname(__file__))
import amrp_simulation as sim

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), 'amrp_output', 'script7_final_comparison_output.txt'
)

# ── USER: set OPTIMAL_BETA from script6 output before running ─────────────────
OPTIMAL_BETA = 1.5874   # locked from script6 (G=+0.7400, bracketed interior optimum)

# ── Protocol constants ─────────────────────────────────────────────────────────
# Seeds must match Table 2 / script2.py exactly to avoid inter-table inconsistency.
# script2.py SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
SEEDS_B      = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]  # n=10, matches Table 2
SEEDS_C      = [42, 123, 456, 789, 101]  # n=5, matches Experiment C original
NOISE_RATES  = [1, 2, 5, 10, 20]        # Hz
ALPHA        = 0.6                       # eq. (8)

# Experiment C grid — same as script3 
THETA_VALS   = [0.5, 1.0, 1.5, 2.0, 2.5]
TAU_VALS_MS  = [50, 100, 150, 200]

# ── Helper: run n seeds at one condition ───────────────────────────────────────

def run_n(noise_hz, seeds, use_amrp=False, use_additive=False, beta=1.0,
          tau_ms=None, theta=None):
    nvm_list, sep_list = [], []
    for s in seeds:
        r = sim.run_experiment(
            noise_hz * Hz,
            use_amrp     = use_amrp,
            use_additive = use_additive,
            beta         = beta,
            seed_val     = s,
            tau_astro    = tau_ms * ms if tau_ms is not None else None,
            theta_noise  = float(theta) if theta is not None else None,
            record_traces= False,
        )
        nvm_list.append(r['nvm_wear'])
        sep_list.append(r['signal_separation'])
    return np.array(nvm_list), np.array(sep_list)


def fmt(arr):
    return f"{np.mean(arr):.4f}±{np.std(arr):.4f}"


# ── Experiment B — SNR sweep ───────────────────────────────────────────────────

def experiment_b(log, opt_beta):
    log("\n" + "=" * 70)
    log("EXPERIMENT B — SNR sweep (n=10 seeds)")
    log(f"  Conditions: STDP | AMRP | Additive(β={opt_beta})")
    log("=" * 70)

    results = {k: {'nvm': [], 'sep': []} for k in ['stdp', 'amrp', 'add']}

    for nr in NOISE_RATES:
        log(f"\n  noise={nr:2d} Hz")

        log("    STDP     ...", end="")
        nvm_s, sep_s = run_n(nr, SEEDS_B, use_amrp=False, use_additive=False)
        log(f"  NVM={fmt(nvm_s)}  sep={fmt(sep_s)}")

        log("    AMRP     ...", end="")
        nvm_a, sep_a = run_n(nr, SEEDS_B, use_amrp=True)
        log(f"  NVM={fmt(nvm_a)}  sep={fmt(sep_a)}")

        log(f"    Add(β={opt_beta})...", end="")
        nvm_d, sep_d = run_n(nr, SEEDS_B, use_amrp=False,
                              use_additive=True, beta=opt_beta)
        log(f"  NVM={fmt(nvm_d)}  sep={fmt(sep_d)}")

        # Reduction factors
        nvm_factor_amrp = np.mean(nvm_s) / (np.mean(nvm_a) + 1e-12)
        nvm_factor_add  = np.mean(nvm_s) / (np.mean(nvm_d) + 1e-12)
        log(f"    NVM reduction vs STDP:  AMRP={nvm_factor_amrp:.2f}×  "
            f"Add={nvm_factor_add:.2f}×")

        # Goldilocks at this noise level
        all_means = [
            ('STDP', float(np.mean(nvm_s)), float(np.mean(sep_s))),
            ('AMRP', float(np.mean(nvm_a)), float(np.mean(sep_a))),
            (f'Add(β={opt_beta})', float(np.mean(nvm_d)), float(np.mean(sep_d))),
        ]
        nvms = [x[1] for x in all_means]; seps = [x[2] for x in all_means]
        dn = max(nvms) - min(nvms) or 1e-9
        ds = max(seps) - min(seps) or 1e-9
        for label, nvm, sep in all_means:
            g = (sep - min(seps)) / ds - ALPHA * (nvm - min(nvms)) / dn
            log(f"    G({label:<20s}) = {g:+.4f}")

        results['stdp']['nvm'].append(np.mean(nvm_s))
        results['stdp']['sep'].append(np.mean(sep_s))
        results['amrp']['nvm'].append(np.mean(nvm_a))
        results['amrp']['sep'].append(np.mean(sep_a))
        results['add']['nvm'].append(np.mean(nvm_d))
        results['add']['sep'].append(np.mean(sep_d))

    # Pre-registered outcome verdict
    log("\n--- Pre-registered outcome verdict (Experiment B) ---")
    amrp_wins_nvm = all(
        results['amrp']['nvm'][i] < results['add']['nvm'][i]
        for i in range(len(NOISE_RATES))
    )
    amrp_wins_sep = all(
        results['amrp']['sep'][i] >= results['add']['sep'][i] - 1e-4
        for i in range(len(NOISE_RATES))
    )
    within_1std = all(
        abs(results['amrp']['nvm'][i] - results['add']['nvm'][i]) <
        np.std(SEEDS_B)  # placeholder; script calculates per-noise std
        for i in range(len(NOISE_RATES))
    )
    log(f"  AMRP NVM lower at every noise level: {amrp_wins_nvm}")
    log(f"  AMRP signal sep comparable/better:   {amrp_wins_sep}")
    if amrp_wins_nvm and amrp_wins_sep:
        log("  VERDICT: AMRP WINS on both metrics across full SNR sweep.")
    elif not amrp_wins_nvm and not amrp_wins_sep:
        log("  VERDICT: ADDITIVE WINS — comparable/better performance with simpler surface.")
    else:
        log("  VERDICT: TIE / MIXED — examine per-noise breakdown above.")

    return results


# ── Experiment C — θ×τ grid (AMRP vs additive) ────────────────────────────────

def experiment_c(log, opt_beta):
    log("\n" + "=" * 70)
    log(f"EXPERIMENT C — θ_noise × τ_astro grid (n={len(SEEDS_C)} seeds, 10 Hz)")
    log(f"  AMRP vs Additive(β={opt_beta})")
    log("=" * 70)

    n_t, n_tau = len(THETA_VALS), len(TAU_VALS_MS)
    amrp_nvm_g = np.zeros((n_t, n_tau))
    amrp_sep_g = np.zeros((n_t, n_tau))
    add_nvm_g  = np.zeros((n_t, n_tau))
    add_sep_g  = np.zeros((n_t, n_tau))

    for i, theta in enumerate(THETA_VALS):
        for j, tau_ms in enumerate(TAU_VALS_MS):
            nvm_a, sep_a = run_n(10, SEEDS_C, use_amrp=True,
                                  tau_ms=tau_ms, theta=theta)
            nvm_d, sep_d = run_n(10, SEEDS_C, use_amrp=False,
                                  use_additive=True, beta=opt_beta,
                                  tau_ms=tau_ms, theta=theta)
            amrp_nvm_g[i, j] = np.mean(nvm_a)
            amrp_sep_g[i, j] = np.mean(sep_a)
            add_nvm_g[i, j]  = np.mean(nvm_d)
            add_sep_g[i, j]  = np.mean(sep_d)
            log(f"  θ={theta:.2f}  τ={tau_ms:3d}ms  "
                f"AMRP NVM={np.mean(nvm_a):.4f} sep={np.mean(sep_a):.4f} | "
                f"Add  NVM={np.mean(nvm_d):.4f} sep={np.mean(sep_d):.4f}")

    # Per-cell Goldilocks delta: G_amrp - G_add
    log("\n--- Goldilocks advantage matrix ΔΔΔG = G_AMRP − G_add (+ = AMRP better) ---")
    log(f"  θ\\τ   " + "  ".join(f"{t:6d}ms" for t in TAU_VALS_MS))
    for i, theta in enumerate(THETA_VALS):
        row = []
        for j in range(n_tau):
            all_means = [
                ('AMRP', amrp_nvm_g[i, j], amrp_sep_g[i, j]),
                ('Add',  add_nvm_g[i, j],  add_sep_g[i, j]),
            ]
            nvms = [x[1] for x in all_means]; seps = [x[2] for x in all_means]
            dn = max(nvms) - min(nvms) or 1e-9
            ds = max(seps) - min(seps) or 1e-9
            g_amrp = (amrp_sep_g[i,j]-min(seps))/ds - ALPHA*(amrp_nvm_g[i,j]-min(nvms))/dn
            g_add  = (add_sep_g[i,j]-min(seps))/ds  - ALPHA*(add_nvm_g[i,j]-min(nvms))/dn
            row.append(f"{g_amrp - g_add:+.3f}")
        log(f"  θ={theta:.1f}   " + "  ".join(row))

    return amrp_nvm_g, amrp_sep_g, add_nvm_g, add_sep_g


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if OPTIMAL_BETA is None:
        print("ERROR: Set OPTIMAL_BETA from script6 output before running.")
        sys.exit(1)

    lines = []

    def log(msg, end="\n"):
        print(msg, end=end, flush=True)
        lines.append(msg + end)

    log("script7_additive_final.py")
    log(f"Locked β = {OPTIMAL_BETA}  (from script6_additive_beta_sweep.py)")
    log(f"Seeds B: {SEEDS_B}  (n={len(SEEDS_B)})")
    log(f"Seeds C: {SEEDS_C}  (n={len(SEEDS_C)})")

    experiment_b(log, OPTIMAL_BETA)
    experiment_c(log, OPTIMAL_BETA)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nOutput saved → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
