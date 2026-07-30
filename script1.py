import warnings, numpy as np, os
warnings.filterwarnings('ignore')
import amrp_simulation as sim
from brian2 import *
import importlib; importlib.reload(sim)

sim.SIM_DURATION = 2000*ms
OUT = {}

print('=== Experiment B: SNR sweep at T=2000ms ===')
for nr_hz in [1, 2, 5, 10, 20]:
    nr = nr_hz * Hz
    rs = sim.run_experiment(nr, use_amrp=False, record_traces=False, seed_val=42)
    rh = sim.run_experiment(nr, use_amrp=True,  record_traces=False, seed_val=42)
    nvm_r  = rs['nvm_wear'] / (rh['nvm_wear'] + 1e-9)
    ent_r  = rs['weight_entropy'] / (rh['weight_entropy'] + 1e-9)
    sep_s  = rs['signal_separation']
    sep_h  = rh['signal_separation']
    print(f'  {nr_hz:2d}Hz  NVM_STDP={rs["nvm_wear"]:.2f}  NVM_AMRP={rh["nvm_wear"]:.2f}  '
          f'ratio={nvm_r:.1f}x  ent_ratio={ent_r:.1f}x  '
          f'sep_STDP={sep_s:.4f}  sep_AMRP={sep_h:.4f}')

print()
print('=== Experiment C: param sweep at T=2000ms ===')
THETA_G = [0.3, 0.7, 1.0, 1.5, 2.5]
TAU_G   = [50, 100, 150, 200]
for theta in THETA_G:
    for tau_ms in TAU_G:
        res = sim.run_experiment(10*Hz, use_amrp=True, tau_astro=tau_ms*ms,
                                 theta_noise=float(theta), record_traces=False, seed_val=42)
        print(f'  theta={theta:.1f}  tau={tau_ms}ms  '
              f'NVM={res["nvm_wear"]:.3f}  sep={res["signal_separation"]:.4f}')