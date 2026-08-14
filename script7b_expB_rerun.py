"""
script7b_expB_rerun.py
======================
Reruns ONLY Experiment B with the corrected seed list matching Table 2 / script2.py.
Experiment C does NOT need rerunning (SEEDS_C = [42,123,456,789,101] was already correct).

Seeds: [42, 123, 456, 789, 101, 202, 303, 404, 505, 606] — identical to script2.py.
Output: amrp_output/script7b_expB_rerun_output.txt
"""

import sys, os
import numpy as np
from brian2 import Hz

sys.path.insert(0, os.path.dirname(__file__))
import amrp_simulation as sim

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), 'amrp_output', 'script7b_expB_rerun_output.txt'
)

# Matches Table 2 / script2.py SEEDS exactly
SEEDS_B    = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
NOISE_RATES = [1, 2, 5, 10, 20]
OPTIMAL_BETA = 1.5874
ALPHA = 0.6

def run_n(noise_hz, seeds, use_amrp=False, use_additive=False, beta=1.0):
    nvm_list, sep_list = [], []
    for s in seeds:
        r = sim.run_experiment(
            noise_hz * Hz,
            use_amrp=use_amrp, use_additive=use_additive, beta=beta,
            seed_val=s, record_traces=False,
        )
        nvm_list.append(r['nvm_wear'])
        sep_list.append(r['signal_separation'])
    return np.array(nvm_list), np.array(sep_list)

def fmt(arr):
    return f"{np.mean(arr):.4f}+/-{np.std(arr):.4f}"

def main():
    lines = []
    def log(msg, end="\n"):
        print(msg, end=end, flush=True)
        lines.append(msg + end)

    log("script7b_expB_rerun.py")
    log(f"Seeds B (Table 2 match): {SEEDS_B}  n={len(SEEDS_B)}")
    log(f"Beta (locked): {OPTIMAL_BETA}")
    log("=" * 70)
    log("EXPERIMENT B RERUN -- SNR sweep (n=10 seeds, corrected seeds)")
    log(f"  Conditions: STDP | AMRP | Additive(b={OPTIMAL_BETA})")
    log("=" * 70)

    nvm_results = {'stdp': [], 'amrp': [], 'add': []}
    sep_results = {'stdp': [], 'amrp': [], 'add': []}

    for nr in NOISE_RATES:
        log(f"\n  noise={nr:2d} Hz")

        log(f"    STDP     ...", end="")
        nvm_s, sep_s = run_n(nr, SEEDS_B, use_amrp=False, use_additive=False)
        log(f"  NVM={fmt(nvm_s)}  sep={fmt(sep_s)}")

        log(f"    AMRP     ...", end="")
        nvm_a, sep_a = run_n(nr, SEEDS_B, use_amrp=True)
        log(f"  NVM={fmt(nvm_a)}  sep={fmt(sep_a)}")

        log(f"    Add(b={OPTIMAL_BETA})...", end="")
        nvm_d, sep_d = run_n(nr, SEEDS_B, use_amrp=False,
                              use_additive=True, beta=OPTIMAL_BETA)
        log(f"  NVM={fmt(nvm_d)}  sep={fmt(sep_d)}")

        nvm_factor_amrp = np.mean(nvm_s) / (np.mean(nvm_a) + 1e-12)
        nvm_factor_add  = np.mean(nvm_s) / (np.mean(nvm_d) + 1e-12)
        log(f"    NVM reduction vs STDP:  AMRP={nvm_factor_amrp:.2f}x  Add={nvm_factor_add:.2f}x")

        # Per-noise Goldilocks
        all_means = [
            ('STDP', float(np.mean(nvm_s)), float(np.mean(sep_s))),
            ('AMRP', float(np.mean(nvm_a)), float(np.mean(sep_a))),
            (f'Add(b={OPTIMAL_BETA})', float(np.mean(nvm_d)), float(np.mean(sep_d))),
        ]
        nvms = [x[1] for x in all_means]; seps = [x[2] for x in all_means]
        dn = max(nvms) - min(nvms) or 1e-9
        ds = max(seps) - min(seps) or 1e-9
        for label, nvm, sep in all_means:
            g = (sep - min(seps)) / ds - ALPHA * (nvm - min(nvms)) / dn
            log(f"    G({label:<22s}) = {g:+.4f}")

        for k, arr in [('stdp', nvm_s), ('amrp', nvm_a), ('add', nvm_d)]:
            nvm_results[k].append(float(np.mean(arr)))
        for k, arr in [('stdp', sep_s), ('amrp', sep_a), ('add', sep_d)]:
            sep_results[k].append(float(np.mean(arr)))

    log("\n--- Pre-registered outcome verdict (Experiment B, corrected seeds) ---")
    amrp_wins_nvm = all(
        nvm_results['amrp'][i] < nvm_results['add'][i]
        for i in range(len(NOISE_RATES))
    )
    amrp_wins_sep = all(
        sep_results['amrp'][i] >= sep_results['add'][i] - 1e-4
        for i in range(len(NOISE_RATES))
    )
    log(f"  AMRP NVM lower at every noise level: {amrp_wins_nvm}")
    log(f"  AMRP signal sep comparable/better:   {amrp_wins_sep}")
    if amrp_wins_nvm and amrp_wins_sep:
        log("  CODED VERDICT: AMRP WINS")
    elif not amrp_wins_nvm and not amrp_wins_sep:
        log("  CODED VERDICT: ADDITIVE WINS")
    else:
        log("  CODED VERDICT: TIE / MIXED -- examine per-noise breakdown above.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nOutput saved -> {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
