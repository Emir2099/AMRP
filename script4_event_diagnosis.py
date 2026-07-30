"""
script4_event_diagnosis.py
==========================
Diagnoses WHY write-event counts are identical at 1-10 Hz and diverge at 20 Hz.

Strategy:
  - Replace single n_events counter with SEPARATE n_pre_events / n_post_events
  - Add SpikeMonitor(ltm) to directly count LTM spikes
  - Run STDP vs AMRP at 1 Hz and 20 Hz, seed=42 only (fast)
  - Report: on_pre events, on_post events, LTM spikes separately

This resolves whether:
  (A) on_pre is strictly identical (STM firing identical) — expected: YES
  (B) on_post diverges at 20 Hz but not at 1 Hz (LTM firing difference)
  (C) Or something else entirely is responsible
"""
import warnings, numpy as np
warnings.filterwarnings('ignore')
from brian2 import *

# ── Reproduce exact global params from amrp_simulation.py ────────────
DT           = 0.1  * ms
SIM_DURATION = 2000 * ms
N_INPUT = 100; N_STM = 60; N_LTM = 30; N_SIGNAL = 20
tau_m = 10*ms; tau_ref = 2*ms
V_THRESH = 0.15; V_RESET = 0.0
tau_pre = 20*ms; tau_post = 20*ms
A_PLUS = 0.005; A_MINUS = 0.0055
W_MAX = 1.0; W_MIN = 0.0
TAU_ASTRO = 100*ms; THETA_NOISE = 1.0; K_SHARP = 2.0
SPIKE_INCR = 1.0 / N_STM
SIGNAL_RATE = 20 * Hz

LIF_EQS = 'dv/dt = -v / tau_m : 1 (unless refractory)'

# ── SEPARATE pre/post counters ────────────────────────────────────────────────
STDP_MODEL = '''
    w              : 1
    dApre/dt       = -Apre  / tau_pre  : 1 (event-driven)
    dApost/dt      = -Apost / tau_post : 1 (event-driven)
    total_abs_dw   : 1
    n_pre_events   : 1
    n_post_events  : 1
'''
STDP_ON_PRE = '''
    v_post         += w
    Apre           += A_PLUS
    w               = clip(w + Apost, W_MIN, W_MAX)
    total_abs_dw   += abs(Apost)
    n_pre_events   += 1
'''
STDP_ON_POST = '''
    Apost          -= A_MINUS
    w               = clip(w + Apre, W_MIN, W_MAX)
    total_abs_dw   += abs(Apre)
    n_post_events  += 1
'''

AMRP_MODEL = '''
    w              : 1
    dApre/dt       = -Apre  / tau_pre  : 1 (event-driven)
    dApost/dt      = -Apost / tau_post : 1 (event-driven)
    total_abs_dw   : 1
    n_pre_events   : 1
    n_post_events  : 1
    HR_syn         : 1
'''
AMRP_ON_PRE = '''
    v_post         += w
    Apre           += A_PLUS
    w               = clip(w + HR_syn * Apost, W_MIN, W_MAX)
    total_abs_dw   += abs(HR_syn * Apost)
    n_pre_events   += 1
'''
AMRP_ON_POST = '''
    Apost          -= A_MINUS
    w               = clip(w + HR_syn * Apre, W_MIN, W_MAX)
    total_abs_dw   += abs(HR_syn * Apre)
    n_post_events  += 1
'''


def run_diag(noise_rate, use_amrp, seed_val=42):
    start_scope()
    np.random.seed(seed_val)
    seed(seed_val)
    defaultclock.dt = DT

    input_rates = np.ones(N_INPUT) * float(noise_rate / Hz)
    input_rates[:N_SIGNAL] += float(SIGNAL_RATE / Hz)
    inp = PoissonGroup(N_INPUT, rates=input_rates * Hz)

    stm = NeuronGroup(N_STM, LIF_EQS,
                      threshold=f'v > {V_THRESH}', reset=f'v = {V_RESET}',
                      refractory=tau_ref, method='euler')
    stm.v = f'rand() * {V_THRESH}'

    ltm = NeuronGroup(N_LTM, LIF_EQS,
                      threshold=f'v > {V_THRESH}', reset=f'v = {V_RESET}',
                      refractory=tau_ref, method='euler')
    ltm.v = f'rand() * {V_THRESH}'

    # Astrocyte
    tau_ms_str = float(TAU_ASTRO / ms)
    astro_eqs = f'''
        drho/dt = -rho / ({tau_ms_str}*ms) : 1
        HR = 1.0 - 1.0 / (1.0 + exp(-{K_SHARP} * (rho - {THETA_NOISE}))) : 1
    '''
    astrocyte = NeuronGroup(1, astro_eqs, method='euler')
    astrocyte.rho = 0.0

    syn_in = Synapses(inp, stm, 'w_in : 1', on_pre='v_post += w_in')
    syn_in.connect(p=0.15)
    syn_in.w_in = 0.05

    syn_stm_astro = Synapses(stm, astrocyte, on_pre=f'rho_post += {float(SPIKE_INCR)}')
    syn_stm_astro.connect(j='0')

    if use_amrp:
        syn = Synapses(stm, ltm, model=AMRP_MODEL,
                       on_pre=AMRP_ON_PRE, on_post=AMRP_ON_POST)
    else:
        syn = Synapses(stm, ltm, model=STDP_MODEL,
                       on_pre=STDP_ON_PRE, on_post=STDP_ON_POST)

    syn.connect(p=0.3)
    syn.w = 0.5
    syn.total_abs_dw = 0.0
    syn.n_pre_events = 0.0
    syn.n_post_events = 0.0
    if use_amrp:
        syn.HR_syn = 1.0

    if use_amrp:
        @network_operation(dt=DT)
        def push_hr():
            syn.HR_syn = float(astrocyte.HR[0])

    sp_stm = SpikeMonitor(stm)
    sp_ltm = SpikeMonitor(ltm)   # <-- THIS IS THE NEW MONITOR

    objs = [inp, stm, ltm, astrocyte, syn_in, syn_stm_astro, syn, sp_stm, sp_ltm]
    if use_amrp:
        objs.append(push_hr)

    net = Network(*objs)
    net.run(SIM_DURATION)

    n_pre  = float(np.sum(syn.n_pre_events[:]))
    n_post = float(np.sum(syn.n_post_events[:]))
    n_stm_spikes = len(sp_stm.i)
    n_ltm_spikes = len(sp_ltm.i)
    nvm_wear = float(np.sum(syn.total_abs_dw[:]))
    n_synapses = len(syn.w)

    # H_R at end of run
    hr_final = float(astrocyte.HR[0]) if use_amrp else float('nan')

    return dict(
        n_pre=n_pre, n_post=n_post,
        n_total=n_pre + n_post,
        n_stm_spikes=n_stm_spikes,
        n_ltm_spikes=n_ltm_spikes,
        n_synapses=n_synapses,
        nvm_wear=nvm_wear,
        hr_final=hr_final,
    )


print('=' * 72)
print('EVENT DIAGNOSIS  -- separate on_pre / on_post counters + LTM monitor')
print('=' * 72)

for nr_hz in [1, 2, 5, 10, 20]:
    nr = nr_hz * Hz
    rs = run_diag(nr, use_amrp=False, seed_val=42)
    rh = run_diag(nr, use_amrp=True,  seed_val=42)

    pre_match  = rs['n_pre']  == rh['n_pre']
    post_diff  = rh['n_post'] - rs['n_post']
    ltm_diff   = rh['n_ltm_spikes'] - rs['n_ltm_spikes']
    stm_match  = rs['n_stm_spikes'] == rh['n_stm_spikes']

    print(f'\n-- {nr_hz} Hz noise ----------------------------------------------')
    print(f'  Synapses       : {rs["n_synapses"]}')
    print(f'  STM spikes     : STDP={rs["n_stm_spikes"]}  AMRP={rh["n_stm_spikes"]}  match={stm_match}')
    print(f'  LTM spikes     : STDP={rs["n_ltm_spikes"]}  AMRP={rh["n_ltm_spikes"]}  diff={ltm_diff:+d}')
    print(f'  on_pre events  : STDP={rs["n_pre"]:.0f}  AMRP={rh["n_pre"]:.0f}  match={pre_match}')
    print(f'  on_post events : STDP={rs["n_post"]:.0f}  AMRP={rh["n_post"]:.0f}  diff={post_diff:+.0f}')
    print(f'  total events   : STDP={rs["n_total"]:.0f}  AMRP={rh["n_total"]:.0f}  '
          f'diff={100*(rh["n_total"]-rs["n_total"])/(rs["n_total"]+1e-9):+.2f}%')
    print(f'  NVM wear       : STDP={rs["nvm_wear"]:.3f}  AMRP={rh["nvm_wear"]:.3f}  '
          f'ratio={rs["nvm_wear"]/(rh["nvm_wear"]+1e-9):.1f}x')
    if not np.isnan(rh['hr_final']):
        print(f'  H_R(t_final)   : {rh["hr_final"]:.4f}')

print('\n')
print('INTERPRETATION GUIDE:')
print('  on_pre match == True  → STM firing identical (expected: always)')
print('  on_post diff == 0     → LTM fires identically (H_R had no effect on LTM rate)')
print('  on_post diff != 0     → LTM firing differs (weight divergence changed post-spike rate)')
print('  LTM diff per synapse  → on_post diff / n_synapses per LTM neuron (=n_synapses/N_LTM)')
