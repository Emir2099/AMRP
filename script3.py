import warnings, numpy as np
warnings.filterwarnings('ignore')
import amrp_simulation as sim
from brian2 import *
import importlib; importlib.reload(sim)

sim.SIM_DURATION = 2000*ms
SEEDS     = [42, 123, 456, 789, 101]   # 5 seeds sufficient for grid
THETA_G   = [0.3, 0.7, 1.0, 1.5, 2.5]
TAU_G     = [50, 100, 150, 200]

nvm_all = {(t,tau): [] for t in THETA_G for tau in TAU_G}
sep_all = {(t,tau): [] for t in THETA_G for tau in TAU_G}

for seed in SEEDS:
    print(f'seed={seed}', end=' ', flush=True)
    for theta in THETA_G:
        for tau_ms in TAU_G:
            res = sim.run_experiment(10*Hz, use_amrp=True, tau_astro=tau_ms*ms,
                                     theta_noise=float(theta), record_traces=False,
                                     seed_val=seed)
            nvm_all[(theta, tau_ms)].append(res['nvm_wear'])
            sep_all[(theta, tau_ms)].append(res['signal_separation'])
    print('done')

print()
print('=== PARAM SWEEP SUMMARY ===')
for theta in THETA_G:
    for tau_ms in TAU_G:
        nv = nvm_all[(theta,tau_ms)]
        sv = sep_all[(theta,tau_ms)]
        print(f'  theta={theta:.1f}  tau={tau_ms}ms  '
              f'NVM={np.mean(nv):.3f}±{np.std(nv):.3f}  '
              f'sep={np.mean(sv):.4f}±{np.std(sv):.4f}')