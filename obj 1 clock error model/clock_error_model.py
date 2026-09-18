import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# 1.  MODEL PARAMETERS
# ─────────────────────────────────────────────
np.random.seed(42)

dt = 1.0          # sampling interval (seconds)
N  = 500          # number of time steps

# Allan-deviation-derived noise spectral densities
# (typical values for a TCXO-class oscillator)
h0   = 2e-19   # white frequency noise (S_y parameter)
h_2  = 2e-20   # random-walk frequency noise

# Process noise covariance matrix Q  (3x3)
# Derived from standard clock noise model
q11 = h0 * dt + (h_2 * dt**3) / 3
q12 = (h_2 * dt**2) / 2
q13 = 0.0
q22 = h_2 * dt
q23 = 0.0
q33 = 1e-24        # very small drift acceleration noise

Q = np.array([
    [q11,  q12,  q13],
    [q12,  q22,  q23],
    [q13,  q23,  q33]
])

# Measurement noise variance  (timing measurement ~ 10 ns rms)
sigma_meas = 10e-9   # 10 nanoseconds
R = np.array([[sigma_meas**2]])

# ─────────────────────────────────────────────
# 2.  STATE TRANSITION MATRIX  F
# ─────────────────────────────────────────────
#  x[k+1] = F x[k] + noise
#
#  | x1 |   | 1  dt  dt²/2 |   | x1 |
#  | x2 | = | 0   1   dt   | * | x2 |
#  | x3 |   | 0   0    1   |   | x3 |

F = np.array([
    [1.0,  dt,  0.5 * dt**2],
    [0.0,  1.0,  dt         ],
    [0.0,  0.0,  1.0        ]
])

# Measurement matrix H  (we observe only clock bias x1)
H = np.array([[1.0, 0.0, 0.0]])

# ─────────────────────────────────────────────
# 3.  SIMULATE TRUE CLOCK ERROR TRAJECTORY
# ─────────────────────────────────────────────
# Cholesky decomposition of Q for correlated noise generation
L = np.linalg.cholesky(Q + 1e-30 * np.eye(3))

x_true = np.zeros((3, N))
z_meas = np.zeros((1, N))

# Initial conditions: small bias & frequency offset, zero drift
x_true[:, 0] = np.array([50e-9, 1e-9, 0.0])   # 50 ns bias, 1 ns/s offset

for k in range(1, N):
    w = L @ np.random.randn(3)
    x_true[:, k] = F @ x_true[:, k-1] + w

for k in range(N):
    v = sigma_meas * np.random.randn()
    z_meas[0, k] = (H @ x_true[:, k])[0] + v

# ─────────────────────────────────────────────
# 4.  PRINT MODEL SUMMARY
# ─────────────────────────────────────────────
print("=" * 60)
print("  CLOCK ERROR STATE-SPACE MODEL  (Objective 1)")
print("=" * 60)
print(f"\nSampling interval      : {dt} s")
print(f"Simulation length      : {N} steps  ({N*dt} s)")
print(f"\nState vector  x = [bias (s),  freq_offset,  drift (1/s)]")
print(f"\nState Transition Matrix F:\n{F}")
print(f"\nProcess Noise Covariance Q:\n{Q}")
print(f"\nMeasurement Noise Variance R: {R[0,0]:.3e}  (sigma={sigma_meas*1e9:.1f} ns)")
print(f"\nInitial True State x0 = {x_true[:,0]}")
print(f"\nFinal True State   xN = {x_true[:,-1]}")
print(f"\nMax Clock Bias  : {np.max(np.abs(x_true[0]))*1e9:.2f} ns")
print(f"Max Freq Offset : {np.max(np.abs(x_true[1]))*1e9:.4f} ns/s")

# ─────────────────────────────────────────────
# 5.  PLOT
# ─────────────────────────────────────────────
t = np.arange(N) * dt

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Stochastic Clock Error State-Space Model\n",
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── subplot 1: Clock Bias
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, x_true[0]*1e9, 'royalblue', lw=1.5, label='True Clock Bias')
ax1.plot(t, z_meas[0]*1e9, 'gray', lw=0.6, alpha=0.5, label='Noisy Measurement')
ax1.set_ylabel('Bias (ns)')
ax1.set_title('Clock Bias  x₁(t)')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# ── subplot 2: Frequency Offset
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, x_true[1]*1e9, 'darkorange', lw=1.5)
ax2.set_ylabel('Offset (ns/s)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Frequency Offset  x₂(t)')
ax2.grid(True, alpha=0.3)

# ── subplot 3: Frequency Drift
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, x_true[2]*1e12, 'firebrick', lw=1.5)
ax3.set_ylabel('Drift (ps/s²)')
ax3.set_xlabel('Time (s)')
ax3.set_title('Frequency Drift  x₃(t)')
ax3.grid(True, alpha=0.3)

# ── subplot 4: Process Noise Q heatmap
ax4 = fig.add_subplot(gs[2, 0])
im = ax4.imshow(Q, cmap='Blues', aspect='auto')
plt.colorbar(im, ax=ax4)
ax4.set_xticks([0,1,2]); ax4.set_yticks([0,1,2])
ax4.set_xticklabels(['x₁','x₂','x₃']); ax4.set_yticklabels(['x₁','x₂','x₃'])
ax4.set_title('Process Noise Cov  Q')
for i in range(3):
    for j in range(3):
        ax4.text(j, i, f'{Q[i,j]:.1e}', ha='center', va='center',
                 fontsize=7, color='navy')

# ── subplot 5: Measurement noise histogram
ax5 = fig.add_subplot(gs[2, 1])
meas_noise = z_meas[0] - (H @ x_true)[0]
ax5.hist(meas_noise*1e9, bins=35, color='mediumseagreen', edgecolor='white', alpha=0.85)
ax5.set_xlabel('Measurement Noise (ns)')
ax5.set_ylabel('Count')
ax5.set_title(f'Measurement Noise  v(t)\nσ={np.std(meas_noise)*1e9:.2f} ns')
ax5.grid(True, alpha=0.3)

plt.savefig('clock_error_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("Results plotted and saved to 'clock_error_model.png'")