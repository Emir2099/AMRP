"""
script5_fan_in_check.py
=======================
Checks whether the 186 differential LTM spikes (AMRP > STDP at 20 Hz, seed=42)
come disproportionately from high-fan-in LTM neurons, or from ordinary sampling.

Method:
  1. Run STDP and AMRP at 20 Hz, seed=42 (same as script4)
  2. Get per-neuron LTM spike counts from SpikeMonitor(ltm)
  3. Compute diff per neuron: AMRP_count - STDP_count
  4. Identify neurons where AMRP > STDP ("differential neurons")
  5. Look up each LTM neuron's actual in-degree (STM->LTM connections)
  6. Compare in-degree of differential neurons vs. all neurons
  7. Report whether the "heavily-connected" explanation holds

Expected in-degree distribution: Binomial(60, 0.3), mean=18, std~3.55
If ratio 3755/186=20.2 is explained by high fan-in, differential neurons
should have mean in-degree significantly above 18.
"""
import warnings, numpy as np
warnings.filterwarnings('ignore')
from brian2 import *

DT = 0.1*ms; SIM_DURATION = 2000*ms
N_INPUT=100; N_STM=60; N_LTM=30; N_SIGNAL=20
tau_m=10*ms; tau_ref=2*ms; V_THRESH=0.15; V_RESET=0.0
tau_pre=20*ms; tau_post=20*ms
A_PLUS=0.005; A_MINUS=0.0055; W_MAX=1.0; W_MIN=0.0
TAU_ASTRO=100*ms; THETA_NOISE=1.0; K_SHARP=2.0
SPIKE_INCR=1.0/N_STM; SIGNAL_RATE=20*Hz
LIF_EQS='dv/dt = -v / tau_m : 1 (unless refractory)'

STDP_MODEL='''
    w : 1
    dApre/dt = -Apre/tau_pre : 1 (event-driven)
    dApost/dt = -Apost/tau_post : 1 (event-driven)
    total_abs_dw : 1
    n_pre_events : 1
    n_post_events : 1
'''
STDP_ON_PRE='''
    v_post += w
    Apre += A_PLUS
    w = clip(w + Apost, W_MIN, W_MAX)
    total_abs_dw += abs(Apost)
    n_pre_events += 1
'''
STDP_ON_POST='''
    Apost -= A_MINUS
    w = clip(w + Apre, W_MIN, W_MAX)
    total_abs_dw += abs(Apre)
    n_post_events += 1
'''
AMRP_MODEL='''
    w : 1
    dApre/dt = -Apre/tau_pre : 1 (event-driven)
    dApost/dt = -Apost/tau_post : 1 (event-driven)
    total_abs_dw : 1
    n_pre_events : 1
    n_post_events : 1
    HR_syn : 1
'''
AMRP_ON_PRE='''
    v_post += w
    Apre += A_PLUS
    w = clip(w + HR_syn*Apost, W_MIN, W_MAX)
    total_abs_dw += abs(HR_syn*Apost)
    n_pre_events += 1
'''
AMRP_ON_POST='''
    Apost -= A_MINUS
    w = clip(w + HR_syn*Apre, W_MIN, W_MAX)
    total_abs_dw += abs(HR_syn*Apre)
    n_post_events += 1
'''

def run_and_get_ltm(noise_rate, use_amrp, seed_val=42):
    start_scope()
    np.random.seed(seed_val); seed(seed_val)
    defaultclock.dt = DT

    input_rates = np.ones(N_INPUT)*float(noise_rate/Hz)
    input_rates[:N_SIGNAL] += float(SIGNAL_RATE/Hz)
    inp = PoissonGroup(N_INPUT, rates=input_rates*Hz)

    stm = NeuronGroup(N_STM, LIF_EQS, threshold=f'v>{V_THRESH}',
                      reset=f'v={V_RESET}', refractory=tau_ref, method='euler')
    stm.v = f'rand()*{V_THRESH}'
    ltm = NeuronGroup(N_LTM, LIF_EQS, threshold=f'v>{V_THRESH}',
                      reset=f'v={V_RESET}', refractory=tau_ref, method='euler')
    ltm.v = f'rand()*{V_THRESH}'

    tau_ms_str = float(TAU_ASTRO/ms)
    astro_eqs = f'''
        drho/dt = -rho/({tau_ms_str}*ms):1
        HR = 1.0-1.0/(1.0+exp(-{K_SHARP}*(rho-{THETA_NOISE}))):1
    '''
    astrocyte = NeuronGroup(1, astro_eqs, method='euler')
    astrocyte.rho = 0.0

    syn_in = Synapses(inp, stm, 'w_in:1', on_pre='v_post+=w_in')
    syn_in.connect(p=0.15); syn_in.w_in = 0.05

    syn_stm_astro = Synapses(stm, astrocyte, on_pre=f'rho_post+={float(SPIKE_INCR)}')
    syn_stm_astro.connect(j='0')

    if use_amrp:
        syn = Synapses(stm, ltm, model=AMRP_MODEL,
                       on_pre=AMRP_ON_PRE, on_post=AMRP_ON_POST)
    else:
        syn = Synapses(stm, ltm, model=STDP_MODEL,
                       on_pre=STDP_ON_PRE, on_post=STDP_ON_POST)

    syn.connect(p=0.3)
    syn.w = 0.5; syn.total_abs_dw=0.0
    syn.n_pre_events=0.0; syn.n_post_events=0.0
    if use_amrp: syn.HR_syn = 1.0

    if use_amrp:
        @network_operation(dt=DT)
        def push_hr():
            syn.HR_syn = float(astrocyte.HR[0])

    sp_ltm = SpikeMonitor(ltm)
    objs = [inp,stm,ltm,astrocyte,syn_in,syn_stm_astro,syn,sp_ltm]
    if use_amrp: objs.append(push_hr)

    net = Network(*objs)
    net.run(SIM_DURATION)

    # per-neuron LTM spike counts
    ltm_counts = np.bincount(np.array(sp_ltm.i[:], dtype=int), minlength=N_LTM)

    # per-LTM neuron in-degree: number of STM->LTM synapses
    post_indices = np.array(syn.j[:], dtype=int)
    in_degree = np.bincount(post_indices, minlength=N_LTM)

    return ltm_counts, in_degree, len(syn.w)

print('Running STDP at 20 Hz, seed=42...')
counts_stdp, in_deg_stdp, n_syn = run_and_get_ltm(20*Hz, use_amrp=False)
print('Running AMRP at 20 Hz, seed=42...')
counts_amrp, in_deg_amrp, _ = run_and_get_ltm(20*Hz, use_amrp=True)

# Sanity check: in-degrees should be identical (same seed, same connectivity)
assert np.all(in_deg_stdp == in_deg_amrp), "In-degrees differ between runs!"
in_deg = in_deg_stdp

diff = counts_amrp.astype(int) - counts_stdp.astype(int)
total_diff = diff.sum()

print(f'\n=== FAN-IN CHECK (20 Hz, seed=42) ===')
print(f'Total synapses        : {n_syn}')
print(f'Mean in-degree        : {in_deg.mean():.2f}  (expected: {60*0.3:.1f})')
print(f'Std  in-degree        : {in_deg.std():.2f}   (expected: {(60*0.3*0.7)**0.5:.2f})')
print(f'LTM spike diff total  : {total_diff} (AMRP - STDP)')
print()

# Neurons where AMRP fired MORE than STDP
more_in_amrp = diff > 0
fewer_in_amrp = diff < 0
same = diff == 0

print(f'LTM neurons AMRP > STDP : {more_in_amrp.sum()} neurons, diff={diff[more_in_amrp].sum()} spikes')
print(f'LTM neurons AMRP < STDP : {fewer_in_amrp.sum()} neurons, diff={diff[fewer_in_amrp].sum()} spikes')
print(f'LTM neurons AMRP = STDP : {same.sum()} neurons')
print()

print('Per-neuron breakdown (neuron_id | STDP_count | AMRP_count | diff | in_degree):')
for i in range(N_LTM):
    marker = ' <-- AMRP>STDP' if diff[i] > 0 else (' <-- STDP>AMRP' if diff[i] < 0 else '')
    print(f'  LTM[{i:2d}] | {counts_stdp[i]:5d} | {counts_amrp[i]:5d} | {diff[i]:+5d} | in_deg={in_deg[i]}{marker}')

if more_in_amrp.any():
    print(f'\nMean in-degree of AMRP>STDP neurons : {in_deg[more_in_amrp].mean():.2f}')
    print(f'Mean in-degree of all LTM neurons   : {in_deg.mean():.2f}')
    print(f'Mean in-degree of AMRP=STDP neurons : {in_deg[same].mean():.2f}')
    delta = in_deg[more_in_amrp].mean() - in_deg.mean()
    print(f'Difference from mean                : {delta:+.2f}')
    print()
    if abs(delta) < in_deg.std():
        print('CONCLUSION: In-degree of differential neurons is within 1 std of mean.')
        print('            "Heavily-connected" explanation is NOT supported by data.')
        print('            Report the ratio 20.2 without causal explanation.')
    else:
        print(f'CONCLUSION: In-degree of differential neurons is {abs(delta)/in_deg.std():.1f} std from mean.')
        print('            "Heavily-connected" explanation IS supported.')
