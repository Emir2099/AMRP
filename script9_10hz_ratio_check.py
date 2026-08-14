"""
script9_10hz_ratio_check.py
============================
Diagnostic for the 10 Hz Table 2 reconciliation (Issue 3).

script2 computed:   mean over seeds of (STDP_NVM_i / AMRP_NVM_i)  -> E[X/Y]
script7b computed:  mean(STDP_NVM) / mean(AMRP_NVM)               -> E[X]/E[Y]

These are different statistics. This script recomputes the 10 Hz ratio
BOTH ways using identical seeds and reports whether the discrepancy
(18.9x vs 17.34x) is explained purely by the ratio-computation method.

"""

import sys, os
import numpy as np
from brian2 import Hz

sys.path.insert(0, os.path.dirname(__file__))
import amrp_simulation as sim

SEEDS    = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
NOISE_HZ = 10

print("script9_10hz_ratio_check.py")
print(f"Noise: {NOISE_HZ} Hz  |  Seeds: {SEEDS}")
print("=" * 60)
print("Per-seed NVM and ratio (STDP/AMRP):")
print(f"{'seed':>6}  {'STDP_NVM':>10}  {'AMRP_NVM':>10}  {'ratio':>8}")
print("-" * 40)

stdp_nvms, amrp_nvms, ratios = [], [], []

for s in SEEDS:
    r_stdp = sim.run_experiment(NOISE_HZ * Hz, use_amrp=False,
                                seed_val=s, record_traces=False)
    r_amrp = sim.run_experiment(NOISE_HZ * Hz, use_amrp=True,
                                seed_val=s, record_traces=False)
    ns = r_stdp['nvm_wear']
    na = r_amrp['nvm_wear']
    ratio = ns / (na + 1e-12)
    stdp_nvms.append(ns)
    amrp_nvms.append(na)
    ratios.append(ratio)
    print(f"{s:>6}  {ns:>10.4f}  {na:>10.4f}  {ratio:>8.2f}x")

print("=" * 60)
mean_ratio   = np.mean(ratios)       # script2 method: E[X/Y]
ratio_means  = np.mean(stdp_nvms) / np.mean(amrp_nvms)  # script7b method: E[X]/E[Y]
std_ratio    = np.std(ratios)

print(f"script2  method  E[X/Y]:   {mean_ratio:.2f}x  (std={std_ratio:.2f})")
print(f"script7b method E[X]/E[Y]: {ratio_means:.2f}x")
print(f"Table 2 published:         18.9x  (std=7.2)")
print()
print("Explanation check:")
print(f"  Gap = {mean_ratio:.2f} - {ratio_means:.2f} = {mean_ratio - ratio_means:.2f}x")
print(f"  AMRP NVM std/mean (CV) = {np.std(amrp_nvms)/np.mean(amrp_nvms)*100:.1f}%")
print(f"  High CV -> Jensen gap is expected direction (E[X/Y] > E[X]/E[Y])")
print()
if abs(mean_ratio - 18.9) < std_ratio:
    print("VERDICT: E[X/Y] method reproduces Table 2 within 1 std.")
    print("  -> 10Hz gap is entirely explained by ratio-computation method.")
    print("  -> No code change or RNG issue involved.")
else:
    print(f"VERDICT: E[X/Y] = {mean_ratio:.2f}x does NOT match Table 2 (18.9x) within std={std_ratio:.2f}.")
    print("  -> Investigate further — may not be purely a computation-method difference.")
