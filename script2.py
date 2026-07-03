import warnings, numpy as np
warnings.filterwarnings('ignore')
import husains_rule_simulation as sim
from brian2 import *
import importlib; importlib.reload(sim)

sim.SIM_DURATION = 2000*ms
SEEDS  = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
NOISES = [1, 2, 5, 10, 20]

# Store: nvm_reduction, entropy_reduction, sep_stdp, sep_husain
results = {n: {'nvm_r':[], 'ent_r':[], 'sep_s':[], 'sep_h':[]} for n in NOISES}

for seed in SEEDS:
    print(f'seed={seed}', end=' ', flush=True)
    for nr_hz in NOISES:
        nr  = nr_hz * Hz
        rs  = sim.run_experiment(nr, use_husain=False, record_traces=False, seed_val=seed)
        rh  = sim.run_experiment(nr, use_husain=True,  record_traces=False, seed_val=seed)
        results[nr_hz]['nvm_r'].append(rs['nvm_wear'] / (rh['nvm_wear'] + 1e-9))
        results[nr_hz]['ent_r'].append(rs['weight_entropy'] / (rh['weight_entropy'] + 1e-9))
        results[nr_hz]['sep_s'].append(rs['signal_separation'])
        results[nr_hz]['sep_h'].append(rh['signal_separation'])
    print('done')

print()
print('=== MULTI-SEED SUMMARY (mean ± std, n=10 seeds) ===')
for nr_hz in NOISES:
    d = results[nr_hz]
    print(f'{nr_hz:2d}Hz | '
          f'NVM_red={np.mean(d["nvm_r"]):.1f}±{np.std(d["nvm_r"]):.1f}x | '
          f'Ent_red={np.mean(d["ent_r"]):.1f}±{np.std(d["ent_r"]):.1f}x | '
          f'Sep_STDP={np.mean(d["sep_s"]):.4f}±{np.std(d["sep_s"]):.4f} | '
          f'Sep_H={np.mean(d["sep_h"]):.4f}±{np.std(d["sep_h"]):.4f}')