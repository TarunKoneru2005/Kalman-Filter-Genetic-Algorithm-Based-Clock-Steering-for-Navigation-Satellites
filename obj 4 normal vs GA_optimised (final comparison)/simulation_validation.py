import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ─────────────────────────────────────────────────────────────
# SECTION 1: SHARED MODEL (Objectives 1–3, identical seed)
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

dt         = 1.0
N          = 500
t          = np.arange(N) * dt

h0         = 2e-19
h_2        = 2e-20
sigma_meas = 10e-9          # nominal measurement noise (10 ns)

# ── Base Q and R  (manual / Objective-1 derived)
q11    = h0*dt + (h_2*dt**3)/3
q12    = (h_2*dt**2)/2
Q_base = np.array([[q11,     q12,   0.0  ],
                   [q12,  h_2*dt,   0.0  ],
                   [0.0,     0.0,   1e-24]])
R_base = np.array([[sigma_meas**2]])

# ── GA-optimal scales discovered in Objective 3
Q_GA_SCALE = 0.004459
R_GA_SCALE = 1e-4
Q_opt      = Q_base * Q_GA_SCALE  +  1e-35 * np.eye(3)
R_opt      = R_base * R_GA_SCALE

# ── System matrices
F = np.array([[1.0,  dt,  0.5*dt**2],
              [0.0, 1.0,        dt  ],
              [0.0, 0.0,       1.0  ]])
H = np.array([[1.0, 0.0, 0.0]])

# ── Simulate nominal true trajectory + measurements
L_chol       = np.linalg.cholesky(Q_base + 1e-30*np.eye(3))
x_true       = np.zeros((3, N))
z_meas       = np.zeros((1, N))
x_true[:, 0] = np.array([50e-9, 1e-9, 0.0])

for k in range(1, N):
    x_true[:, k] = F @ x_true[:, k-1] + L_chol @ np.random.randn(3)
for k in range(N):
    z_meas[0, k] = (H @ x_true[:, k])[0] + sigma_meas*np.random.randn()

x0_init = np.zeros(3)
P0_init = np.diag([1e-12, 1e-16, 1e-24])

# ─────────────────────────────────────────────────────────────
# SECTION 2: KALMAN FILTER (from Objective 2 — unchanged)
# ─────────────────────────────────────────────────────────────
def run_kf(F, H, Q, R, z, x0, P0):
    n, m, Nk = F.shape[0], H.shape[0], z.shape[1]
    x_est    = np.zeros((n, Nk))
    P_hist   = np.zeros((Nk, n, n))
    innov    = np.zeros((m, Nk))
    x_hat, P = x0.copy(), P0.copy()
    I        = np.eye(n)
    for k in range(Nk):
        xp       = F @ x_hat
        Pp       = F @ P @ F.T + Q
        inn      = z[:, k] - H @ xp
        S        = H @ Pp @ H.T + R
        K        = Pp @ H.T @ np.linalg.inv(S)
        x_hat    = xp + K @ inn
        P        = 0.5*((I - K@H) @ Pp)
        P        = 0.5*(P + P.T)
        x_est[:, k] = x_hat
        P_hist[k]   = P
        innov[:, k] = inn
    return x_est, P_hist, innov

# ─────────────────────────────────────────────────────────────
# SECTION 3: NOMINAL RUN — MANUAL vs GA
# ─────────────────────────────────────────────────────────────
x_man, P_man, inn_man = run_kf(F, H, Q_base, R_base, z_meas, x0_init, P0_init)
x_ga,  P_ga,  inn_ga  = run_kf(F, H, Q_opt,  R_opt,  z_meas, x0_init, P0_init)

err_man = x_man[0] - x_true[0]
err_ga  = x_ga[0]  - x_true[0]

# ── Convergence: first k where running MSE stabilises
#    (defined as first k where |error| < 2σ_meas AND stays < 2σ_meas for 20 steps)
def convergence_step(error, threshold, hold=20):
    for k in range(len(error) - hold):
        if np.all(np.abs(error[k:k+hold]) < threshold):
            return k
    return len(error) - 1

thresh     = 2 * sigma_meas
conv_man   = convergence_step(err_man, thresh)
conv_ga    = convergence_step(err_ga,  thresh)

# ── Running MSE (causal window of 50 samples)
WIN = 50
def running_mse(error, win):
    rmse_run = np.zeros(len(error))
    for k in range(len(error)):
        seg = error[max(0, k-win):k+1]
        rmse_run[k] = np.mean(seg**2)
    return rmse_run

rmse_run_man = running_mse(err_man, WIN)
rmse_run_ga  = running_mse(err_ga,  WIN)

# ── Full-run and steady-state metrics
SS = int(0.8 * N)
def metrics(err, P_hist, innov):
    rmse_full  = np.sqrt(np.mean(err**2)) * 1e9
    rmse_ss    = np.sqrt(np.mean(err[SS:]**2)) * 1e9
    mse_full   = np.mean(err**2)
    # 3-σ consistency: fraction of errors within ±3σ_kf
    sigma_kf   = np.sqrt(P_hist[:, 0, 0])
    within_3s  = np.mean(np.abs(err) <= 3*sigma_kf) * 100
    # Innovation autocorrelation at lag 1
    inn        = innov[0]
    inn_norm   = inn - inn.mean()
    ac_lag1    = np.corrcoef(inn_norm[:-1], inn_norm[1:])[0, 1]
    return rmse_full, rmse_ss, mse_full, within_3s, ac_lag1

rm_full, rm_ss, mse_m, w3s_m, ac_m = metrics(err_man, P_man, inn_man)
rg_full, rg_ss, mse_g, w3s_g, ac_g = metrics(err_ga,  P_ga,  inn_ga)

# ─────────────────────────────────────────────────────────────
# SECTION 4: MONTE CARLO — 100 independent noise runs
# ─────────────────────────────────────────────────────────────
N_MC = 100
mc_rmse_man = np.zeros(N_MC)
mc_rmse_ga  = np.zeros(N_MC)
mc_conv_man = np.zeros(N_MC, dtype=int)
mc_conv_ga  = np.zeros(N_MC, dtype=int)

print("Running Monte Carlo (100 trials)...", end='', flush=True)
for trial in range(N_MC):
    rng = np.random.RandomState(trial + 1000)
    # Simulate new noise realisation
    xt      = np.zeros((3, N))
    zm      = np.zeros((1, N))
    xt[:, 0] = np.array([50e-9, 1e-9, 0.0])
    for k in range(1, N):
        xt[:, k] = F @ xt[:, k-1] + L_chol @ rng.randn(3)
    for k in range(N):
        zm[0, k] = (H @ xt[:, k])[0] + sigma_meas*rng.randn()

    xm_, _, _ = run_kf(F, H, Q_base, R_base, zm, x0_init, P0_init)
    xg_, _, _ = run_kf(F, H, Q_opt,  R_opt,  zm, x0_init, P0_init)

    em = xm_[0] - xt[0]
    eg = xg_[0]  - xt[0]
    mc_rmse_man[trial] = np.sqrt(np.mean(em**2)) * 1e9
    mc_rmse_ga[trial]  = np.sqrt(np.mean(eg**2)) * 1e9
    mc_conv_man[trial] = convergence_step(em, thresh)
    mc_conv_ga[trial]  = convergence_step(eg,  thresh)

print(" done.")

# ─────────────────────────────────────────────────────────────
# SECTION 5: NOISE SENSITIVITY STUDY
# ─────────────────────────────────────────────────────────────
sigma_vals    = np.logspace(np.log10(1e-9), np.log10(100e-9), 20)  # 1–100 ns
sens_rmse_man = np.zeros(len(sigma_vals))
sens_rmse_ga  = np.zeros(len(sigma_vals))

print("Running noise sensitivity sweep (20 points)...", end='', flush=True)
for i, sig in enumerate(sigma_vals):
    rng2 = np.random.RandomState(999)
    xt2      = np.zeros((3, N))
    zm2      = np.zeros((1, N))
    xt2[:,0] = np.array([50e-9, 1e-9, 0.0])
    for k in range(1, N):
        xt2[:, k] = F @ xt2[:, k-1] + L_chol @ rng2.randn(3)
    for k in range(N):
        zm2[0, k] = (H @ xt2[:, k])[0] + sig*rng2.randn()

    R_sig = np.array([[sig**2]])
    # Manual: keep Q_base, update R to match new sigma
    xm2, _, _ = run_kf(F, H, Q_base,            R_sig,           zm2, x0_init, P0_init)
    # GA: re-scale R_opt proportionally
    xg2, _, _ = run_kf(F, H, Q_opt, R_sig*R_GA_SCALE, zm2, x0_init, P0_init)

    sens_rmse_man[i] = np.sqrt(np.mean((xm2[0]-xt2[0])**2)) * 1e9
    sens_rmse_ga[i]  = np.sqrt(np.mean((xg2[0]-xt2[0])**2)) * 1e9

print(" done.")

# ─────────────────────────────────────────────────────────────
# SECTION 6: PRINT RESULTS TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  OBJECTIVE 4 — SIMULATION VALIDATION RESULTS")
print("="*65)

print(f"\n{'─'*65}")
print(f"  {'Metric':<38} {'Manual':>10}  {'GA-Opt':>10}")
print(f"{'─'*65}")
print(f"  {'RMSE — Full Run (ns)':<38} {rm_full:>10.4f}  {rg_full:>10.4f}")
print(f"  {'RMSE — Steady State (ns)':<38} {rm_ss:>10.4f}  {rg_ss:>10.4f}")
print(f"  {'MSE — Full Run (s²)':<38} {mse_m:>10.3e}  {mse_g:>10.3e}")
print(f"  {'Convergence Step (k)':<38} {conv_man:>10d}  {conv_ga:>10d}")
print(f"  {'3-σ Consistency (%)':<38} {w3s_m:>10.1f}  {w3s_g:>10.1f}")
print(f"  {'Innovation Autocorr (lag-1)':<38} {ac_m:>10.4f}  {ac_g:>10.4f}")
print(f"{'─'*65}")
print(f"  {'RMSE Improvement Factor':<38} {'—':>10}  {rm_full/rg_full:>9.2f}×")
print(f"  {'SS RMSE Improvement Factor':<38} {'—':>10}  {rm_ss/rg_ss:>9.2f}×")
print(f"{'─'*65}")

print(f"\n{'─'*65}")
print(f"  MONTE CARLO SUMMARY  (N={N_MC} trials)")
print(f"{'─'*65}")
print(f"  {'Metric':<38} {'Manual':>10}  {'GA-Opt':>10}")
print(f"{'─'*65}")
print(f"  {'Mean RMSE (ns)':<38} {mc_rmse_man.mean():>10.4f}  {mc_rmse_ga.mean():>10.4f}")
print(f"  {'Std RMSE (ns)':<38} {mc_rmse_man.std():>10.4f}  {mc_rmse_ga.std():>10.4f}")
print(f"  {'Median RMSE (ns)':<38} {np.median(mc_rmse_man):>10.4f}  {np.median(mc_rmse_ga):>10.4f}")
print(f"  {'95th Pctile RMSE (ns)':<38} {np.percentile(mc_rmse_man,95):>10.4f}  {np.percentile(mc_rmse_ga,95):>10.4f}")
print(f"  {'Mean Convergence Step':<38} {mc_conv_man.mean():>10.1f}  {mc_conv_ga.mean():>10.1f}")
print(f"{'─'*65}")

# ─────────────────────────────────────────────────────────────
# SECTION 7: FIGURE 1 — Core validation plots
# ─────────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(16, 13))
fig1.suptitle(
    "Objective 4 — Simulation Validation:  Manual vs GA-Optimised Kalman Filter\n",
    fontsize=13, fontweight='bold'
)
gs1 = gridspec.GridSpec(3, 2, figure=fig1, hspace=0.50, wspace=0.35)

# ── Plot 1: Bias estimation — True vs Manual vs GA
ax1 = fig1.add_subplot(gs1[0, :])
ax1.plot(t, x_true[0]*1e9,   color='steelblue',   lw=2.2, label='True Bias',               zorder=5)
ax1.plot(t, z_meas[0]*1e9,   color='silver',       lw=0.6, alpha=0.6, label='Noisy Measurement')
ax1.plot(t, x_man[0]*1e9,    color='tomato',       lw=1.5, linestyle='--',
         label=f'Manual KF  (RMSE={rm_full:.3f} ns)')
ax1.plot(t, x_ga[0]*1e9,     color='limegreen',    lw=1.5, linestyle='-.',
         label=f'GA-Opt KF  (RMSE={rg_full:.4f} ns)')
ax1.fill_between(t,
    (x_ga[0] - 3*np.sqrt(P_ga[:,0,0]))*1e9,
    (x_ga[0] + 3*np.sqrt(P_ga[:,0,0]))*1e9,
    color='limegreen', alpha=0.12, label='GA ±3σ bounds')
ax1.set_ylabel('Clock Bias (ns)')
ax1.set_title('Clock Bias Estimation — Full Comparison')
ax1.legend(fontsize=8, loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Estimation error (bias)
ax2 = fig1.add_subplot(gs1[1, :])
ax2.plot(t, err_man*1e9, color='tomato',    lw=1.3, alpha=0.85,
         label=f'Manual KF Error  (RMSE={rm_full:.3f} ns)')
ax2.plot(t, err_ga*1e9,  color='limegreen', lw=1.3, alpha=0.85,
         label=f'GA-Opt KF Error  (RMSE={rg_full:.4f} ns)')
ax2.axhline(0, color='black', lw=0.6)
ax2.axvline(conv_man*dt, color='tomato',    lw=1.2, linestyle=':',
            label=f'Manual conv @ k={conv_man}')
ax2.axvline(conv_ga*dt,  color='limegreen', lw=1.2, linestyle=':',
            label=f'GA-Opt conv @ k={conv_ga}')
ax2.axvspan(SS*dt, N*dt, alpha=0.07, color='gray', label='Steady-state window')
ax2.set_ylabel('Error (ns)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Clock Bias Estimation Error with Convergence Markers')
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Running MSE
ax3 = fig1.add_subplot(gs1[2, 0])
ax3.semilogy(t, rmse_run_man*1e18, color='tomato',    lw=1.8, label='Manual')
ax3.semilogy(t, rmse_run_ga*1e18,  color='limegreen', lw=1.8, label='GA-Opt')
ax3.axvline(conv_man*dt, color='tomato',    lw=1.0, linestyle=':')
ax3.axvline(conv_ga*dt,  color='limegreen', lw=1.0, linestyle=':')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Running MSE (ns²)')
ax3.set_title(f'Running MSE  (window={WIN} s)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Plot 4: Innovation autocorrelation
ax4 = fig1.add_subplot(gs1[2, 1])
max_lag = 50
lags    = np.arange(-max_lag, max_lag+1)
inn_m_n = inn_man[0] - inn_man[0].mean()
inn_g_n = inn_ga[0]  - inn_ga[0].mean()
ac_m_full = np.correlate(inn_m_n, inn_m_n, mode='full') / (np.var(inn_m_n)*N)
ac_g_full = np.correlate(inn_g_n, inn_g_n, mode='full') / (np.var(inn_g_n)*N)
mid       = len(ac_m_full)//2
ax4.stem(lags, ac_m_full[mid-max_lag:mid+max_lag+1],
         linefmt='tomato', markerfmt='ro', basefmt='k-',
         label='Manual')
ax4.stem(lags, ac_g_full[mid-max_lag:mid+max_lag+1],
         linefmt='limegreen', markerfmt='g^', basefmt='k-',
         label='GA-Opt')
conf = 1.96 / np.sqrt(N)
ax4.axhline( conf, color='gray', lw=1.0, linestyle='--', label='95% conf')
ax4.axhline(-conf, color='gray', lw=1.0, linestyle='--')
ax4.set_xlabel('Lag')
ax4.set_ylabel('Autocorrelation')
ax4.set_title('Innovation Autocorrelation (Whiteness Test)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-max_lag, max_lag)

plt.savefig('/home/claude/clock_model/validation_fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Figure 1 saved → validation_fig1.png")

# ─────────────────────────────────────────────────────────────
# SECTION 8: FIGURE 2 — Monte Carlo & Sensitivity
# ─────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(16, 13))
fig2.suptitle(
    "Objective 4 — Monte Carlo & Noise Sensitivity Validation\n",
    fontsize=13, fontweight='bold'
)
gs2 = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.50, wspace=0.35)

# ── Plot 5: MC RMSE distributions (violin)
ax5 = fig2.add_subplot(gs2[0, :])
parts = ax5.violinplot([mc_rmse_man, mc_rmse_ga], positions=[1, 2],
                        showmeans=True, showmedians=True)
parts['bodies'][0].set_facecolor('tomato');    parts['bodies'][0].set_alpha(0.6)
parts['bodies'][1].set_facecolor('limegreen'); parts['bodies'][1].set_alpha(0.6)
for pc in ['cmeans','cmedians','cbars','cmins','cmaxes']:
    parts[pc].set_color('black'); parts[pc].set_linewidth(1.2)
ax5.set_xticks([1, 2])
ax5.set_xticklabels([f'Manual KF\nμ={mc_rmse_man.mean():.3f} ns\nσ={mc_rmse_man.std():.3f} ns',
                     f'GA-Opt KF\nμ={mc_rmse_ga.mean():.4f} ns\nσ={mc_rmse_ga.std():.4f} ns'],
                    fontsize=10)
ax5.set_ylabel('RMSE (ns)')
ax5.set_title(f'Monte Carlo RMSE Distribution  (N={N_MC} trials)  —  Improvement: '
              f'{mc_rmse_man.mean()/mc_rmse_ga.mean():.1f}×')
ax5.grid(True, alpha=0.3, axis='y')

# ── Plot 6: MC RMSE histogram
ax6 = fig2.add_subplot(gs2[1, 0])
ax6.hist(mc_rmse_man, bins=20, color='tomato',    alpha=0.65, edgecolor='white',
         label=f'Manual  μ={mc_rmse_man.mean():.2f}')
ax6.hist(mc_rmse_ga,  bins=20, color='limegreen', alpha=0.65, edgecolor='white',
         label=f'GA-Opt  μ={mc_rmse_ga.mean():.3f}')
ax6.set_xlabel('RMSE (ns)')
ax6.set_ylabel('Count')
ax6.set_title('MC RMSE Histogram')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# ── Plot 7: MC Convergence step distribution
ax7 = fig2.add_subplot(gs2[1, 1])
ax7.hist(mc_conv_man, bins=20, color='tomato',    alpha=0.65, edgecolor='white',
         label=f'Manual  μ={mc_conv_man.mean():.0f}')
ax7.hist(mc_conv_ga,  bins=20, color='limegreen', alpha=0.65, edgecolor='white',
         label=f'GA-Opt  μ={mc_conv_ga.mean():.0f}')
ax7.set_xlabel('Convergence Step (k)')
ax7.set_ylabel('Count')
ax7.set_title('MC Convergence Step Histogram')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# ── Plot 8: Noise sensitivity
ax8 = fig2.add_subplot(gs2[2, 0])
sigma_ns = sigma_vals * 1e9
ax8.loglog(sigma_ns, sens_rmse_man, 'o-', color='tomato',    lw=1.8, ms=5, label='Manual KF')
ax8.loglog(sigma_ns, sens_rmse_ga,  's-', color='limegreen', lw=1.8, ms=5, label='GA-Opt KF')
ax8.axvline(sigma_meas*1e9, color='gray', lw=1.2, linestyle='--', label='Nominal σ=10 ns')
ax8.set_xlabel('Measurement Noise σ (ns)')
ax8.set_ylabel('RMSE (ns)')
ax8.set_title('Noise Sensitivity: RMSE vs σ_meas')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3, which='both')

# ── Plot 9: Bar summary of all key metrics
ax9 = fig2.add_subplot(gs2[2, 1])
metric_labels = ['Full RMSE\n(ns)', 'SS RMSE\n(ns)', '3-σ Consist.\n(%)÷10',
                 'Conv. Step\n÷100', 'Inn. AC\n×10']
man_vals = [rm_full, rm_ss, w3s_m/10, conv_man/100, abs(ac_m)*10]
ga_vals  = [rg_full, rg_ss, w3s_g/10, conv_ga/100,  abs(ac_g)*10]
x_pos    = np.arange(len(metric_labels))
w        = 0.35
bars_m   = ax9.bar(x_pos - w/2, man_vals, w, color='tomato',    alpha=0.8, label='Manual')
bars_g   = ax9.bar(x_pos + w/2, ga_vals,  w, color='limegreen', alpha=0.8, label='GA-Opt')
ax9.set_xticks(x_pos)
ax9.set_xticklabels(metric_labels, fontsize=8)
ax9.set_title('Key Metrics Summary (scaled for display)')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')
for bar in bars_m:
    h = bar.get_height()
    ax9.text(bar.get_x()+bar.get_width()/2, h*1.02, f'{h:.2f}',
             ha='center', va='bottom', fontsize=7, color='darkred')
for bar in bars_g:
    h = bar.get_height()
    ax9.text(bar.get_x()+bar.get_width()/2, h*1.02, f'{h:.2f}',
             ha='center', va='bottom', fontsize=7, color='darkgreen')

plt.savefig('/home/claude/clock_model/validation_fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Figure 2 saved → validation_fig2.png")
print("\n[✓] Objective 4 complete.")