import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

def run_kalman_filter(F, H, Q, R, z_meas, x0_init, P0_init):
    """
    Discrete-time Kalman Filter.

    Parameters
    ----------
    F       : (n,n) state transition matrix
    H       : (m,n) measurement matrix
    Q       : (n,n) process noise covariance
    R       : (m,m) measurement noise covariance
    z_meas  : (m, N) measurement array
    x0_init : (n,)  initial state estimate
    P0_init : (n,n) initial covariance estimate

    Returns
    -------
    x_est   : (n, N) filtered state estimates
    P_hist  : (N, n, n) covariance history
    K_hist  : (N, n, m) Kalman gain history
    innov   : (m, N) innovation (measurement residual) sequence
    """
    n   = F.shape[0]
    m   = H.shape[0]
    N_  = z_meas.shape[1]

    x_est  = np.zeros((n, N_))
    P_hist = np.zeros((N_, n, n))
    K_hist = np.zeros((N_, n, m))
    innov  = np.zeros((m, N_))

    # ── Initialise
    x_hat = x0_init.copy()
    P     = P0_init.copy()
    I     = np.eye(n)

    for k in range(N_):
        # ── PREDICT ──────────────────────────────────────
        x_pred = F @ x_hat                        # (n,)
        P_pred = F @ P @ F.T + Q                  # (n,n)

        # ── INNOVATION ───────────────────────────────────
        z_pred    = H @ x_pred                    # (m,)
        innov_k   = z_meas[:, k] - z_pred         # (m,)
        S         = H @ P_pred @ H.T + R          # (m,m) innovation covariance
        innov[:, k] = innov_k

        # ── KALMAN GAIN ──────────────────────────────────
        K   = P_pred @ H.T @ np.linalg.inv(S)    # (n,m)

        # ── UPDATE ───────────────────────────────────────
        x_hat = x_pred + K @ innov_k              # (n,)
        P     = (I - K @ H) @ P_pred              # (n,n)  — Joseph form below for numerical stability
        P     = 0.5 * (P + P.T)                   # symmetry enforcement

        # ── Store
        x_est[:, k]  = x_hat
        P_hist[k]    = P
        K_hist[k]    = K

    return x_est, P_hist, K_hist, innov

def init_population(pop_size, n_genes, low, high):
    return np.random.uniform(low, high, size=(pop_size, n_genes))

def tournament_select(population, fitnesses, k=2):
    """Pick k individuals at random, return the best."""
    idx = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmax(fitnesses[idx])]
    return population[best].copy()

def crossover(parent1, parent2, p):
    """Single-point crossover."""
    if np.random.rand() < p:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(chromosome, p, std, low, high):
    """Gaussian mutation with boundary clipping."""
    chrom = chromosome.copy()
    for i in range(len(chrom)):
        if np.random.rand() < p:
            chrom[i] += np.random.normal(0, std)
    return np.clip(chrom, low, high)

def run_ga(pop_size, n_genes, n_gen, crossover_p, mutation_p,
           mutation_std, elite_count, bounds_low, bounds_high):
    """
    Main GA loop.
    Returns:
        best_chromosome  : optimal [log_q, log_r]
        best_fitness_hist: best fitness per generation
        mean_fitness_hist: mean fitness per generation
        population_hist  : population at each generation (for diversity plot)
    """
    population = init_population(pop_size, n_genes, bounds_low, bounds_high)
    best_fitness_hist = []
    mean_fitness_hist = []
    population_hist   = []

    print(f"\n{'─'*55}")
    print(f"  GA Optimisation — {n_gen} generations × {pop_size} individuals")
    print(f"{'─'*55}")
    print(f"{'Gen':>5}  {'Best MSE (ns²)':>16}  {'Best q_scale':>14}  {'Best r_scale':>14}")
    print(f"{'─'*55}")

    best_chromosome = None
    best_fitness    = -np.inf

    for gen in range(n_gen):
        # ── Evaluate fitness
        fitnesses = np.array([fitness(chrom) for chrom in population])

        # ── Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness    = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()

        best_mse_ns2 = (1.0 / best_fitness) * 1e18  # convert s² → ns²
        best_fitness_hist.append(best_fitness)
        mean_fitness_hist.append(np.mean(fitnesses))
        population_hist.append(population.copy())

        if gen % 10 == 0 or gen == n_gen - 1:
            qs = 10.0 ** best_chromosome[0]
            rs = 10.0 ** best_chromosome[1]
            print(f"{gen+1:>5}  {best_mse_ns2:>16.4f}  {qs:>14.4f}  {rs:>14.4f}")

        # ── Elitism: carry forward top individuals
        elite_idx  = np.argsort(fitnesses)[-elite_count:]
        elites     = population[elite_idx].copy()

        # ── Build new population
        new_population = []
        while len(new_population) < pop_size - elite_count:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            c1, c2 = crossover(p1, p2, crossover_p)
            c1 = mutate(c1, mutation_p, mutation_std, bounds_low, bounds_high)
            c2 = mutate(c2, mutation_p, mutation_std, bounds_low, bounds_high)
            new_population.extend([c1, c2])

        population = np.vstack([elites, np.array(new_population[:pop_size - elite_count])])

    print(f"{'─'*55}")
    return best_chromosome, best_fitness_hist, mean_fitness_hist, population_hist

# ─────────────────────────────────────────────────────────────
# FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────
def fitness(chromosome):
    """
    chromosome = [log10(q_scale), log10(r_scale)]
    Returns fitness = 1 / MSE  (higher is better)
    """
    q_scale = 10.0 ** chromosome[0]
    r_scale = 10.0 ** chromosome[1]

    Q_trial = Q_base * q_scale
    R_trial = R_base * r_scale

    # Ensure positive definiteness
    Q_trial += 1e-35 * np.eye(3)

    try:
        x_est, _, _, _ = run_kalman_filter(F, H, Q_trial, R_trial,
                                        z_meas, x0_init, P0_init)
        mse = np.mean((x_est[0] - x_true[0])**2)
        if np.isnan(mse) or mse <= 0:
            return 0.0
        return 1.0 / mse
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────
# SHARED MODEL SETUP
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

dt = 1.0
N = 500
t = np.arange(N) * dt

h0         = 2e-19
h_2        = 2e-20
sigma_meas = 10e-9

# ── Base Q (manual / nominal — from Objective 1)
q11 = h0 * dt + (h_2 * dt**3) / 3
q12 = (h_2 * dt**2) / 2
Q_base = np.array([
    [q11,       q12,   0.0  ],
    [q12,  h_2*dt,     0.0  ],
    [0.0,       0.0,   1e-24]
])

R_base = np.array([[sigma_meas**2]])   # 1×1

F = np.array([
    [1.0,  dt,  0.5 * dt**2],
    [0.0,  1.0,  dt         ],
    [0.0,  0.0,  1.0        ]
])
H = np.array([[1.0, 0.0, 0.0]])

# ── Simulate true trajectory + measurements (same seed as Obj 1 & 2)
L_chol       = np.linalg.cholesky(Q_base + 1e-30 * np.eye(3))
x_true       = np.zeros((3, N))
z_meas       = np.zeros((1, N))
x_true[:, 0] = np.array([50e-9, 1e-9, 0.0])

for k in range(1, N):
    w = L_chol @ np.random.randn(3)
    x_true[:, k] = F @ x_true[:, k-1] + w

for k in range(N):
    z_meas[0, k] = (H @ x_true[:, k])[0] + sigma_meas * np.random.randn()

x0_init = np.zeros(3)
P0_init = np.diag([1e-12, 1e-16, 1e-24])

# ─────────────────────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────────────────────

# ── GA Hyper-parameters
POP_SIZE      = 40       # population size
N_GENES       = 2        # chromosome length: [log_q, log_r]
N_GEN         = 60       # number of generations
CROSSOVER_P   = 0.80     # crossover probability
MUTATION_P    = 0.20     # mutation probability per gene
MUTATION_STD  = 0.40     # std of Gaussian mutation (log-space)
ELITE_COUNT   = 2        # number of elite individuals preserved

# ── Search bounds in log10 space
BOUNDS_LOW  = np.array([-4.0, -4.0])   # 0.0001× base values
BOUNDS_HIGH = np.array([ 4.0,  4.0])   # 10000× base values

def run_ga(pop_size, n_genes, n_gen, crossover_p, mutation_p,
           mutation_std, elite_count, bounds_low, bounds_high):
    """
    Main GA loop.
    Returns:
        best_chromosome  : optimal [log_q, log_r]
        best_fitness_hist: best fitness per generation
        mean_fitness_hist: mean fitness per generation
        population_hist  : population at each generation (for diversity plot)
    """
    population = init_population(pop_size, n_genes, bounds_low, bounds_high)
    best_fitness_hist = []
    mean_fitness_hist = []
    population_hist   = []

    print(f"\n{'─'*55}")
    print(f"  GA Optimisation — {n_gen} generations × {pop_size} individuals")
    print(f"{'─'*55}")
    print(f"{'Gen':>5}  {'Best MSE (ns²)':>16}  {'Best q_scale':>14}  {'Best r_scale':>14}")
    print(f"{'─'*55}")

    best_chromosome = None
    best_fitness    = -np.inf

    for gen in range(n_gen):
        # ── Evaluate fitness
        fitnesses = np.array([fitness(chrom) for chrom in population])

        # ── Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness    = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()

        best_mse_ns2 = (1.0 / best_fitness) * 1e18  # convert s² → ns²
        best_fitness_hist.append(best_fitness)
        mean_fitness_hist.append(np.mean(fitnesses))
        population_hist.append(population.copy())

        if gen % 10 == 0 or gen == n_gen - 1:
            qs = 10.0 ** best_chromosome[0]
            rs = 10.0 ** best_chromosome[1]
            print(f"{gen+1:>5}  {best_mse_ns2:>16.4f}  {qs:>14.4f}  {rs:>14.4f}")

        # ── Elitism: carry forward top individuals
        elite_idx  = np.argsort(fitnesses)[-elite_count:]
        elites     = population[elite_idx].copy()

        # ── Build new population
        new_population = []
        while len(new_population) < pop_size - elite_count:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            c1, c2 = crossover(p1, p2, crossover_p)
            c1 = mutate(c1, mutation_p, mutation_std, bounds_low, bounds_high)
            c2 = mutate(c2, mutation_p, mutation_std, bounds_low, bounds_high)
            new_population.extend([c1, c2])

        population = np.vstack([elites, np.array(new_population[:pop_size - elite_count])])

    print(f"{'─'*55}")
    return best_chromosome, best_fitness_hist, mean_fitness_hist, population_hist

# ─────────────────────────────────────────────────────────────
# RUN GA + COMPARE RESULTS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  OBJECTIVE 3 — GA-Optimised Kalman Filter")
print("=" * 60)

t_start = time.time()
best_chrom, best_fit_hist, mean_fit_hist, pop_hist = run_ga(
    POP_SIZE, N_GENES, N_GEN, CROSSOVER_P, MUTATION_P,
    MUTATION_STD, ELITE_COUNT, BOUNDS_LOW, BOUNDS_HIGH
)
elapsed = time.time() - t_start

# ── Derive optimised Q and R
q_scale_opt = 10.0 ** best_chrom[0]
r_scale_opt = 10.0 ** best_chrom[1]
Q_opt       = Q_base * q_scale_opt
R_opt       = R_base * r_scale_opt

# ── Run KF with manual (baseline) Q, R
x_est_manual, P_hist_manual, _, innov_manual = run_kalman_filter(
    F, H, Q_base, R_base, z_meas, x0_init, P0_init)

# ── Run KF with GA-optimised Q, R
x_est_ga, P_hist_ga, _, innov_ga = run_kalman_filter(
    F, H, Q_opt, R_opt, z_meas, x0_init, P0_init)

# ── Metrics
err_manual = x_est_manual[0] - x_true[0]
err_ga     = x_est_ga[0]     - x_true[0]

mse_manual = np.mean(err_manual**2)
mse_ga     = np.mean(err_ga**2)

rmse_manual = np.sqrt(mse_manual) * 1e9   # ns
rmse_ga     = np.sqrt(mse_ga)     * 1e9   # ns

# Convergence (first k where |error| < 3σ)
thresh      = 3 * sigma_meas
conv_manual = next((k for k in range(N) if abs(err_manual[k]) < thresh), None)
conv_ga     = next((k for k in range(N) if abs(err_ga[k])     < thresh), None)

# Steady-state MSE (last 20%)
ss = int(0.8 * N)
rmse_ss_manual = np.sqrt(np.mean(err_manual[ss:]**2)) * 1e9
rmse_ss_ga     = np.sqrt(np.mean(err_ga[ss:]**2))     * 1e9

print(f"\n{'─'*55}")
print(f"  GA RESULTS")
print(f"{'─'*55}")
print(f"GA runtime              : {elapsed:.1f} s")
print(f"Optimal q_scale         : {q_scale_opt:.6f}  (log10 = {best_chrom[0]:.4f})")
print(f"Optimal r_scale         : {r_scale_opt:.6f}  (log10 = {best_chrom[1]:.4f})")
print(f"\n{'─'*55}")
print(f"  COMPARISON:  Manual  vs  GA-Optimised")
print(f"{'─'*55}")
print(f"{'Metric':<30} {'Manual':>10} {'GA-Opt':>10}")
print(f"{'─'*55}")
print(f"{'RMSE (full run, ns)':<30} {rmse_manual:>10.4f} {rmse_ga:>10.4f}")
print(f"{'Steady-State RMSE (ns)':<30} {rmse_ss_manual:>10.4f} {rmse_ss_ga:>10.4f}")
print(f"{'Convergence step (k)':<30} {str(conv_manual):>10} {str(conv_ga):>10}")
print(f"{'MSE improvement':<30} {'—':>10} {rmse_manual/rmse_ga:>9.2f}×")
print(f"{'─'*55}")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "Objective 3 — GA-Optimised Kalman Filter  |  Q & R Covariance Tuning\n",
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── Plot 1: Bias estimate comparison (full run)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, x_true[0]*1e9,        color='steelblue',  lw=2.0, label='True Bias',           zorder=5)
ax1.plot(t, x_est_manual[0]*1e9,  color='tomato',     lw=1.4, linestyle='--', label=f'Manual KF  (RMSE={rmse_manual:.3f} ns)')
ax1.plot(t, x_est_ga[0]*1e9,      color='limegreen',  lw=1.4, linestyle='-.',  label=f'GA-Opt KF (RMSE={rmse_ga:.4f} ns)')
ax1.set_ylabel('Bias (ns)')
ax1.set_title('Clock Bias — Manual vs GA-Optimised Kalman Filter')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

# ── Plot 2: Estimation error comparison
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(t, err_manual*1e9, color='tomato',    lw=1.2, alpha=0.85, label=f'Manual KF Error  (RMSE={rmse_manual:.3f} ns)')
ax2.plot(t, err_ga*1e9,     color='limegreen', lw=1.2, alpha=0.85, label=f'GA-Opt KF Error  (RMSE={rmse_ga:.4f} ns)')
ax2.axhline(0, color='black', lw=0.5)
if conv_ga is not None:
    ax2.axvline(conv_ga*dt, color='green', lw=1.1, linestyle=':', label=f'GA Convergence @ t={conv_ga}s')
if conv_manual is not None:
    ax2.axvline(conv_manual*dt, color='red', lw=1.1, linestyle=':', label=f'Manual Convergence @ t={conv_manual}s')
ax2.set_ylabel('Estimation Error (ns)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Estimation Error — Manual vs GA-Optimised')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Plot 3: GA best fitness over generations
ax3 = fig.add_subplot(gs[2, 0])
gens = np.arange(1, N_GEN + 1)
ax3.semilogy(gens, [1/f if f > 0 else np.nan for f in best_fit_hist],
             color='royalblue', lw=2.0, label='Best MSE')
ax3.semilogy(gens, [1/f if f > 0 else np.nan for f in mean_fit_hist],
             color='gray', lw=1.2, linestyle='--', label='Mean MSE')
ax3.axhline(mse_manual, color='tomato', lw=1.2, linestyle=':', label='Manual MSE')
ax3.set_xlabel('Generation')
ax3.set_ylabel('MSE (s²)  — log scale')
ax3.set_title('GA Convergence — MSE vs Generation')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── Plot 4: Population evolution in parameter space
ax4 = fig.add_subplot(gs[2, 1])
# Plot population at Gen 0, Gen 20, Gen final
for gen_idx, label, alpha, ms in [(0, 'Gen 1', 0.25, 30),
                                    (19,'Gen 20',0.45, 35),
                                    (N_GEN-1,'Final',0.90, 50)]:
    pop = pop_hist[gen_idx]
    ax4.scatter(pop[:, 0], pop[:, 1], alpha=alpha, s=ms, label=label)
ax4.scatter(best_chrom[0], best_chrom[1], color='gold', s=150,
            marker='*', zorder=10, label=f'Best  ({q_scale_opt:.3f}, {r_scale_opt:.3f})')
ax4.set_xlabel('log₁₀(q_scale)')
ax4.set_ylabel('log₁₀(r_scale)')
ax4.set_title('Population Distribution Across Generations')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── Plot 5: Q diagonal comparison (manual vs GA)
ax5 = fig.add_subplot(gs[3, 0])
labels_q  = ['Q[0,0]', 'Q[1,1]', 'Q[2,2]']
q_manual  = [Q_base[i,i] for i in range(3)]
q_ga      = [Q_opt[i,i]  for i in range(3)]
x_pos     = np.arange(3)
w         = 0.35
ax5.bar(x_pos - w/2, q_manual, w, label='Manual Q diag', color='tomato',    alpha=0.8)
ax5.bar(x_pos + w/2, q_ga,     w, label='GA-Opt Q diag', color='limegreen', alpha=0.8)
ax5.set_yscale('log')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(labels_q)
ax5.set_ylabel('Value (log scale)')
ax5.set_title(f'Process Noise Q Diagonal\n(q_scale = {q_scale_opt:.4f}×)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')

# ── Plot 6: R comparison
ax6 = fig.add_subplot(gs[3, 1])
bars = ax6.bar(['Manual R', 'GA-Opt R'],
               [R_base[0,0], R_opt[0,0]],
               color=['tomato', 'limegreen'], alpha=0.85, width=0.4)
ax6.set_yscale('log')
ax6.set_ylabel('R value (log scale)')
ax6.set_title(f'Measurement Noise R\n(r_scale = {r_scale_opt:.4f}×)')
for bar, val in zip(bars, [R_base[0,0], R_opt[0,0]]):
    ax6.text(bar.get_x() + bar.get_width()/2, val * 1.5,
             f'{val:.2e}', ha='center', va='bottom', fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('ga_optimised_kf.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[✓] Plot saved successfully!!")