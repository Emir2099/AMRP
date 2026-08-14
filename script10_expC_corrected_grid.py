"""
script10_expC_corrected_grid.py
================================
Reruns Experiment C at the ORIGINAL Table 3 theta_noise grid:
    theta_noise in {0.3, 0.7, 1.0, 1.5, 2.5}

Fixes two issues in section_6_4_draft.md:
  Issue 1: ablation used {0.5,1.0,1.5,2.0,2.5} -- does not match Table 3's
           {0.3, 0.7, 1.0, 1.5, 2.5}. Using the original grid allows direct
           row-for-row comparison between Table 3 and Table Y.
  Issue 2: previous Exp C output had no std devs. This script reports
           mean +/- std for both AMRP and Add NVM/sep at every cell.

Also flags close NVM calls (gap < 1 combined pooled SD) so the "15/20" tally
can be qualified correctly in the manuscript.

Run manually:
    python -X utf8 script10_expC_corrected_grid.py

Output: amrp_output/script10_expC_corrected_output.txt
"""

import sys, os
import numpy as np
from brian2 import ms, Hz

sys.path.insert(0, os.path.dirname(__file__))
import amrp_simulation as sim

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), 'amrp_output', 'script10_expC_corrected_output.txt'
)

# Original Table 3 grid -- matches exactly
THETA_VALS  = [0.3, 0.7, 1.0, 1.5, 2.5]          # original Table 3 rows
TAU_VALS_MS = [50,  100, 150, 200]                  # original Table 3 cols
SEEDS       = [42, 123, 456, 789, 101]              # n=5, same as Table 3
NOISE_HZ    = 10
BETA        = 1.5874
ALPHA       = 0.6                                   # eq. 8

def run_cell(theta, tau_ms, seeds, use_amrp, beta=None):
    nvm_list, sep_list = [], []
    for s in seeds:
        kwargs = dict(
            noise_rate    = NOISE_HZ * Hz,
            use_amrp      = use_amrp,
            use_additive  = (not use_amrp and beta is not None),
            beta          = beta if beta is not None else 1.0,
            tau_astro     = tau_ms * ms,
            theta_noise   = theta,
            seed_val      = s,
            record_traces = False,
        )
        r = sim.run_experiment(**kwargs)
        nvm_list.append(r['nvm_wear'])
        sep_list.append(r['signal_separation'])
    return np.array(nvm_list), np.array(sep_list)

def main():
    lines = []
    def log(msg, end='\n'):
        print(msg, end=end, flush=True)
        lines.append(msg + end)

    log("script10_expC_corrected_grid.py")
    log(f"Grid: theta={THETA_VALS}  tau_ms={TAU_VALS_MS}")
    log(f"Seeds: {SEEDS}  n={len(SEEDS)} | noise={NOISE_HZ}Hz | beta={BETA}")
    log("=" * 80)

    # Store full results for ΔG matrix
    results = {}

    for theta in THETA_VALS:
        for tau_ms in TAU_VALS_MS:
            key = (theta, tau_ms)
            log(f"\n  theta={theta:.1f}  tau={tau_ms:3d}ms", end="")

            nvm_a, sep_a = run_cell(theta, tau_ms, SEEDS, use_amrp=True)
            nvm_d, sep_d = run_cell(theta, tau_ms, SEEDS, use_amrp=False, beta=BETA)

            amrp_nvm_m, amrp_nvm_s = np.mean(nvm_a), np.std(nvm_a)
            amrp_sep_m, amrp_sep_s = np.mean(sep_a), np.std(sep_a)
            add_nvm_m,  add_nvm_s  = np.mean(nvm_d), np.std(nvm_d)
            add_sep_m,  add_sep_s  = np.mean(sep_d), np.std(sep_d)

            nvm_winner = 'AMRP' if amrp_nvm_m < add_nvm_m else 'Add'
            nvm_gap    = abs(amrp_nvm_m - add_nvm_m)
            pooled_sd  = np.sqrt((amrp_nvm_s**2 + add_nvm_s**2) / 2)
            close_call = nvm_gap < pooled_sd

            log(f"  AMRP NVM={amrp_nvm_m:.2f}+/-{amrp_nvm_s:.2f}  sep={amrp_sep_m:.4f}+/-{amrp_sep_s:.4f}")
            log(f"  Add  NVM={add_nvm_m:.2f}+/-{add_nvm_s:.2f}  sep={add_sep_m:.4f}+/-{add_sep_s:.4f}")
            flag = "  ** CLOSE CALL: gap={:.1f} < pooledSD={:.1f}".format(nvm_gap, pooled_sd) if close_call else ""
            log(f"  NVM winner: {nvm_winner}  gap={nvm_gap:.1f}  pooledSD={pooled_sd:.1f}{flag}")

            results[key] = {
                'amrp_nvm': (amrp_nvm_m, amrp_nvm_s),
                'amrp_sep': (amrp_sep_m, amrp_sep_s),
                'add_nvm':  (add_nvm_m,  add_nvm_s),
                'add_sep':  (add_sep_m,  add_sep_s),
                'nvm_winner': nvm_winner,
                'close_call': close_call,
                'gap': nvm_gap,
                'pooled_sd': pooled_sd,
            }

    # Summary tables
    log("\n" + "=" * 80)
    log("SUMMARY — NVM winner per cell (original Table 3 theta grid)")
    log(f"{'theta':>6}  {'tau':>6}  {'AMRP NVM':>18}  {'Add NVM':>18}  {'winner':>6}  {'close?':>8}")
    log("-" * 80)
    amrp_wins, add_wins, close_calls = 0, 0, []
    for theta in THETA_VALS:
        for tau_ms in TAU_VALS_MS:
            r = results[(theta, tau_ms)]
            an, as_ = r['amrp_nvm']
            dn, ds  = r['add_nvm']
            w = r['nvm_winner']
            c = "** CLOSE" if r['close_call'] else ""
            log(f"  {theta:>5.1f}  {tau_ms:>5d}ms  {an:>7.2f}+/-{as_:>6.2f}  {dn:>7.2f}+/-{ds:>6.2f}  {w:>6}  {c}")
            if w == 'AMRP': amrp_wins += 1
            else: add_wins += 1
            if r['close_call']: close_calls.append((theta, tau_ms))

    log(f"\nNVM tally: AMRP wins {amrp_wins}/20  Add wins {add_wins}/20")
    if close_calls:
        log(f"Close calls (gap < pooled SD): {close_calls}")
        log(f"Clear NVM wins: AMRP={amrp_wins - sum(1 for c in close_calls if results[c]['nvm_winner']=='AMRP')}  Add={add_wins - sum(1 for c in close_calls if results[c]['nvm_winner']=='Add')}")

    # ΔG matrix
    log("\n--- Delta-G matrix (G_AMRP - G_Add, per-cell 2-condition normalisation) ---")
    log(f"  {'theta':>5}", end="")
    for tau_ms in TAU_VALS_MS:
        log(f"  {'tau='+str(tau_ms)+'ms':>10}", end="")
    log("")

    for theta in THETA_VALS:
        log(f"  {theta:>5.1f}", end="")
        for tau_ms in TAU_VALS_MS:
            r = results[(theta, tau_ms)]
            an, dn = r['amrp_nvm'][0], r['add_nvm'][0]
            as_, ds = r['amrp_sep'][0], r['add_sep'][0]
            nvm_range = abs(an - dn) or 1e-9
            sep_range = abs(as_ - ds) or 1e-9
            nvm_norm_a = (an - min(an, dn)) / nvm_range
            sep_norm_a = (as_ - min(as_, ds)) / sep_range
            g_a = sep_norm_a - ALPHA * nvm_norm_a
            g_d = (1 - sep_norm_a) - ALPHA * (1 - nvm_norm_a)
            delta_g = g_a - g_d
            log(f"  {delta_g:>+10.3f}", end="")
        log("")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nOutput saved -> {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
