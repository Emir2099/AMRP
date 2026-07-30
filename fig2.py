import importlib, amrp_simulation as sim
importlib.reload(sim)
from brian2 import *
import os, warnings, matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

sim.SIM_DURATION = 2000*ms
OUT = 'amrp_output'
os.makedirs(OUT, exist_ok=True)

s_list, h_list = [], []
for nr_hz in [1, 2, 5, 10, 20]:
    rs = sim.run_experiment(nr_hz*Hz, use_amrp=False, record_traces=False, seed_val=42)
    rh = sim.run_experiment(nr_hz*Hz, use_amrp=True,  record_traces=False, seed_val=42)
    s_list.append(rs); h_list.append(rh)

nvals = [1.0, 2.0, 5.0, 10.0, 20.0]
f2 = sim.fig_snr_curves(s_list, h_list, nvals)
f2.savefig(f'{OUT}/fig2_snr_sweep_2000ms.pdf', dpi=180,
           bbox_inches='tight', facecolor=sim.C['bg'])
print('Done — fig2_snr_sweep_2000ms.pdf')