import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ga_optimised_kf import run_kalman_filter

# ─────────────────────────────────────────────────────────────
# SECTION 1:  RE-USE MODEL FROM OBJECTIVE 1 (same seed & params)
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

dt = 1.0          # sampling interval (s)
N  = 500          # time steps
t  = np.arange(N) * dt

# ── Allan deviation noise parameters (TCXO-class oscillator)
h0 = 2e-19
h_2 = 2e-20

# ── Process noise covariance Q (3×3) — from Objective 1
q11 = h0 * dt + (h_2 * dt**3) / 3
q12 = (h_2 * dt**2) / 2
Q = np.array([
    [q11,  q12,  0.0 ],
    [q12,  h_2 * dt, 0.0],
    [0.0,  0.0,  1e-24]
])

# ── Measurement noise variance (10 ns rms)
sigma_meas  = 10e-9
R = np.array([[sigma_meas**2]])

# ── State transition and measurement matrices
F = np.array([
    [1.0,  dt,  0.5 * dt**2],
    [0.0,  1.0,  dt         ],
    [0.0,  0.0,  1.0        ]
])
H = np.array([[1.0, 0.0, 0.0]])

# ── Simulate true clock trajectory + noisy measurements (identical to Obj 1)
L_chol = np.linalg.cholesky(Q + 1e-30 * np.eye(3))
x_true = np.zeros((3, N))
z_meas = np.zeros((1, N))
x_true[:, 0] = np.array([50e-9, 1e-9, 0.0])

for k in range(1, N):
    w = L_chol @ np.random.randn(3)
    x_true[:, k] = F @ x_true[:, k-1] + w

for k in range(N):
    v = sigma_meas * np.random.randn()
    z_meas[0, k] = (H @ x_true[:, k])[0] + v

# ─────────────────────────────────────────────────────────────
# SECTION 2:  KALMAN FILTER IMPLEMENTATION
# ─────────────────────────────────────────────────────────────

# ── Initial filter conditions (deliberately off-truth to show convergence)
x0_init = np.zeros(3)                     # assume no prior knowledge
P0_init = np.diag([1e-12, 1e-16, 1e-24]) # large initial uncertainty

x_est, P_hist, K_hist, innov = run_kalman_filter(
    F, H, Q, R, z_meas, x0_init, P0_init
)

# ─────────────────────────────────────────────────────────────
# SECTION 3:  PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────
error_meas = z_meas[0] - x_true[0]          # raw measurement error
error_kf   = x_est[0]  - x_true[0]          # KF bias estimation error
error_freq = x_est[1]  - x_true[1]          # KF frequency offset error
error_drft = x_est[2]  - x_true[2]          # KF drift error

mse_meas  = np.mean(error_meas**2)
mse_kf    = np.mean(error_kf**2)
rmse_meas = np.sqrt(mse_meas) * 1e9          # ns
rmse_kf   = np.sqrt(mse_kf)  * 1e9          # ns

# Convergence step: first k where |error_kf| < 3×sigma
thresh    = 3 * sigma_meas
conv_step = next((k for k in range(N) if abs(error_kf[k]) < thresh), None)

# Steady-state MSE (last 20% of run)
ss_start  = int(0.8 * N)
mse_ss    = np.mean(error_kf[ss_start:]**2)
rmse_ss   = np.sqrt(mse_ss) * 1e9

print("=" * 60)
print("  KALMAN FILTER RESULTS")
print("=" * 60)
print(f"\nRMSE (raw measurement)  : {rmse_meas:.3f} ns")
print(f"RMSE (Kalman estimate)  : {rmse_kf:.4f} ns")
print(f"RMSE improvement        : {rmse_meas/rmse_kf:.1f}×")
print(f"Convergence step        : k = {conv_step}  ({conv_step*dt:.0f} s)")
print(f"Steady-state RMSE       : {rmse_ss:.4f} ns  (last 20% of run)")
print(f"\nFinal Kalman Gain K     :\n{K_hist[-1].flatten()}")
print(f"\nFinal Covariance diag P : {np.diag(P_hist[-1])}")

# ─────────────────────────────────────────────────────────────
# SECTION 4:  PLOTS
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 12))
fig.suptitle(
    "Objective 2 — Discrete-Time Kalman Filter for Clock Error Estimation\n",
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── Plot 1: Clock Bias — True vs KF vs Measurement
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, x_true[0]*1e9,   color='steelblue',   lw=1.8, label='True Bias', zorder=3)
ax1.plot(t, z_meas[0]*1e9,   color='silver',       lw=0.6, alpha=0.7, label='Noisy Measurement')
ax1.plot(t, x_est[0]*1e9,    color='orangered',    lw=1.5, linestyle='--', label='KF Estimate', zorder=4)
ax1.fill_between(t,
    (x_est[0] - 3*np.sqrt(P_hist[:,0,0]))*1e9,
    (x_est[0] + 3*np.sqrt(P_hist[:,0,0]))*1e9,
    alpha=0.15, color='orangered', label='±3σ KF bound')
ax1.set_ylabel('Bias (ns)')
ax1.set_title('Clock Bias Estimation')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

# ── Plot 2: Frequency Offset estimate
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, x_true[1]*1e9, color='darkorange', lw=1.5, label='True')
ax2.plot(t, x_est[1]*1e9,  color='purple',     lw=1.4, linestyle='--', label='KF Estimate')
ax2.set_ylabel('Offset (ns/s)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Frequency Offset  x₂(t)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Frequency Drift estimate
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, x_true[2]*1e12, color='firebrick', lw=1.5, label='True')
ax3.plot(t, x_est[2]*1e12,  color='teal',      lw=1.4, linestyle='--', label='KF Estimate')
ax3.set_ylabel('Drift (ps/s²)')
ax3.set_xlabel('Time (s)')
ax3.set_title('Frequency Drift  x₃(t)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── Plot 4: Bias Estimation Error comparison
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(t, error_meas*1e9, color='gray',      lw=0.7, alpha=0.6, label=f'Raw Measurement Error  (RMSE={rmse_meas:.2f} ns)')
ax4.plot(t, error_kf*1e9,   color='orangered', lw=1.5, label=f'KF Estimation Error    (RMSE={rmse_kf:.4f} ns)')
if conv_step:
    ax4.axvline(conv_step*dt, color='green', lw=1.2, linestyle=':', label=f'Convergence @ t={conv_step}s')
ax4.axhline(0, color='black', lw=0.5)
ax4.set_ylabel('Error (ns)')
ax4.set_xlabel('Time (s)')
ax4.set_title('Clock Bias Estimation Error')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── Plot 5: Innovation sequence
ax5 = fig.add_subplot(gs[3, 0])
ax5.plot(t, innov[0]*1e9, color='steelblue', lw=0.8, alpha=0.8)
ax5.axhline( 3*sigma_meas*1e9, color='red', lw=1, linestyle='--', label='±3σ bound')
ax5.axhline(-3*sigma_meas*1e9, color='red', lw=1, linestyle='--')
ax5.set_ylabel('Innovation (ns)')
ax5.set_xlabel('Time (s)')
ax5.set_title('Innovation Sequence  (Whiteness check)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ── Plot 6: Kalman Gain convergence
ax6 = fig.add_subplot(gs[3, 1])
ax6.plot(t, K_hist[:, 0, 0], label='K₁ (bias)',       lw=1.5, color='steelblue')
ax6.plot(t, K_hist[:, 1, 0], label='K₂ (freq offset)', lw=1.5, color='darkorange')
ax6.plot(t, K_hist[:, 2, 0], label='K₃ (drift)',       lw=1.5, color='firebrick')
ax6.set_ylabel('Gain value')
ax6.set_xlabel('Time (s)')
ax6.set_title('Kalman Gain Convergence')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.savefig('kalman_filter_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[✓] Plot saved successfully")