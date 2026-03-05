import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────
# Configuration
#
# We use 2 weights (d=2) so the loss surface
# can be drawn in 3D — same bowl as gradientDescent.py.
# NUM_SAMPLES is large enough that the difference
# between batch GD and SGD becomes obvious.
# ─────────────────────────────────────────────
TRUE_WEIGHTS  = np.array([2.0, -1.5])
NUM_SAMPLES   = 3_000
NOISE_STD     = 0.5
SEED          = 42

BATCH_ETA     = 0.05
BATCH_EPOCHS  = 20      # batch GD: each epoch scans ALL data for ONE update

SGD_ETA_INIT  = 1.0     # SGD: decreasing step size η = η₀ / sqrt(t)
SGD_EPOCHS    = 3       # SGD: each epoch = NUM_SAMPLES cheap updates

LOSS_EVAL_EVERY = 50    # how often to evaluate full train loss during SGD

np.random.seed(SEED)

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
d = len(TRUE_WEIGHTS)
X = np.random.randn(NUM_SAMPLES, d)
y = X @ TRUE_WEIGHTS + np.random.randn(NUM_SAMPLES) * NOISE_STD

# ─────────────────────────────────────────────
# Loss and gradients
#
# Per-example loss:    Loss(x, y, w) = ½(w·x − y)²
# Per-example grad:  ∇Loss(x, y, w) = (w·x − y)·x
#
# Full train loss:   TrainLoss(w) = (1/n) Σ ½(w·xᵢ − yᵢ)²
# Batch gradient:  ∇TrainLoss(w) = (1/n) Xᵀ(Xw − y)
# ─────────────────────────────────────────────
def train_loss(w):
    r = X @ w - y
    return np.mean(0.5 * r ** 2)

def batch_gradient(w):
    r = X @ w - y
    return (1 / len(y)) * X.T @ r

def per_example_gradient(w, x_i, y_i):
    return (w @ x_i - y_i) * x_i

# ─────────────────────────────────────────────
# BATCH GRADIENT DESCENT
#
# Cost per update: scans ALL n examples to compute
# one gradient → one weight update per epoch.
#
# After processing k·n examples:  k updates made.
# ─────────────────────────────────────────────
print("=" * 60)
print("BATCH GRADIENT DESCENT")
print(f"  {NUM_SAMPLES:,} examples scanned per update  |  η = {BATCH_ETA}")
print("=" * 60)

w_batch = np.zeros(d)
batch_examples_seen = [0]      # x-axis: total examples processed
batch_losses        = [train_loss(w_batch)]
batch_trajectory    = [w_batch.copy()]

t0 = time.time()
for epoch in range(1, BATCH_EPOCHS + 1):
    grad    = batch_gradient(w_batch)          # scans ALL n examples
    w_batch = w_batch - BATCH_ETA * grad
    loss    = train_loss(w_batch)

    batch_examples_seen.append(epoch * NUM_SAMPLES)
    batch_losses.append(loss)
    batch_trajectory.append(w_batch.copy())

    print(f"  Epoch {epoch:>3}  |  examples processed: {epoch*NUM_SAMPLES:>9,}"
          f"  |  loss: {loss:.5f}  |  w={w_batch.round(3)}")

batch_time = time.time() - t0
batch_trajectory = np.array(batch_trajectory)
print(f"  Time: {batch_time:.3f}s  |  Total updates: {BATCH_EPOCHS}")

# ─────────────────────────────────────────────
# STOCHASTIC GRADIENT DESCENT
#
# Cost per update: uses ONLY 1 example to compute
# one gradient → one weight update per example.
#
# After processing k·n examples:  k·n updates made.
# That is n× more updates for the same computation.
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STOCHASTIC GRADIENT DESCENT")
print(f"  1 example per update  |  η = {SGD_ETA_INIT}/√t  (decreasing)")
print("=" * 60)

w_sgd = np.zeros(d)
sgd_examples_seen = [0]
sgd_losses        = [train_loss(w_sgd)]
sgd_trajectory    = [w_sgd.copy()]

num_updates = 0
t0 = time.time()

for epoch in range(SGD_EPOCHS):
    indices = np.random.permutation(NUM_SAMPLES)
    for i in indices:
        num_updates += 1
        eta  = SGD_ETA_INIT / np.sqrt(num_updates)   # decreasing step size
        grad = per_example_gradient(w_sgd, X[i], y[i])
        w_sgd = w_sgd - eta * grad

        # record every LOSS_EVAL_EVERY steps (cheap enough for d=2)
        if num_updates % LOSS_EVAL_EVERY == 0:
            sgd_examples_seen.append(num_updates)
            sgd_losses.append(train_loss(w_sgd))
            sgd_trajectory.append(w_sgd.copy())

sgd_time = time.time() - t0
sgd_trajectory = np.array(sgd_trajectory)

print(f"  Total updates: {num_updates:,}  (vs {BATCH_EPOCHS} for batch GD)")
print(f"  Time: {sgd_time:.3f}s")
print(f"  Final w = {w_sgd.round(4)}  |  True w = {TRUE_WEIGHTS}")

# ─────────────────────────────────────────────
# Build loss surface grid
# ─────────────────────────────────────────────
w1_range = np.linspace(TRUE_WEIGHTS[0] - 4, TRUE_WEIGHTS[0] + 4, 80)
w2_range = np.linspace(TRUE_WEIGHTS[1] - 4, TRUE_WEIGHTS[1] + 4, 80)
W1, W2   = np.meshgrid(w1_range, w2_range)

J_surface = np.array([
    train_loss(np.array([w1, w2]))
    for w1, w2 in zip(W1.ravel(), W2.ravel())
]).reshape(W1.shape)

# ─────────────────────────────────────────────
# FIGURE 1 — The core efficiency argument
#
# Panel 1: 3D loss surface — both paths
#   Batch GD:  a few large clean steps
#   SGD:       many small noisy steps — but gets there faster
#
# Panel 2: Contour view — same story, top-down
#
# Panel 3: Loss vs examples processed  ← THE KEY PLOT
#   Same x-axis = same computational budget.
#   SGD reaches low loss far sooner.
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
fig.suptitle(
    "Batch GD vs SGD — same computation budget, very different progress",
    fontsize=13, fontweight='bold'
)

# ── 3D surface ────────────────────────────────
ax3d = fig.add_subplot(131, projection='3d')
ax3d.plot_surface(W1, W2, J_surface, cmap='viridis', alpha=0.45)

# Batch GD path
b_traj_losses = [train_loss(batch_trajectory[i]) for i in range(len(batch_trajectory))]
ax3d.plot(batch_trajectory[:, 0], batch_trajectory[:, 1], b_traj_losses,
          'o-', color='red', lw=2, markersize=5, label=f'Batch GD ({BATCH_EPOCHS} updates)')

# SGD path (subsample for readability)
step = max(1, len(sgd_trajectory) // 200)
s    = sgd_trajectory[::step]
s_losses = [train_loss(s[i]) for i in range(len(s))]
ax3d.plot(s[:, 0], s[:, 1], s_losses,
          color='orange', lw=1, alpha=0.7, label=f'SGD ({num_updates:,} updates)')

ax3d.scatter(*TRUE_WEIGHTS, train_loss(TRUE_WEIGHTS),
             color='cyan', s=80, marker='*', zorder=10, label='True minimum')
ax3d.set_xlabel('w₁'); ax3d.set_ylabel('w₂'); ax3d.set_zlabel('J(w)')
ax3d.set_title('3D Loss Surface\nRed=Batch GD  Orange=SGD')
ax3d.legend(fontsize=7)

# ── Contour view ──────────────────────────────
ax2d = fig.add_subplot(132)
ax2d.contourf(W1, W2, J_surface, levels=40, cmap='viridis')
ax2d.contour( W1, W2, J_surface, levels=40, colors='white', linewidths=0.3, alpha=0.4)

ax2d.plot(batch_trajectory[:, 0], batch_trajectory[:, 1],
          'o-', color='red', lw=2, markersize=6,
          label=f'Batch GD — {BATCH_EPOCHS} big steps\n(each costs {NUM_SAMPLES:,} examples)')
ax2d.plot(sgd_trajectory[::step, 0], sgd_trajectory[::step, 1],
          color='orange', lw=1, alpha=0.8,
          label=f'SGD — {num_updates:,} small steps\n(each costs 1 example)')
ax2d.scatter(*TRUE_WEIGHTS, color='cyan', s=100, marker='*', zorder=5, label='True min')
ax2d.scatter(*np.zeros(d),  color='white', s=60, marker='o', zorder=5, label='Start [0,0]')

ax2d.set_xlabel('w₁'); ax2d.set_ylabel('w₂')
ax2d.set_title('Contour view (top-down)\nBatch=few clean steps  SGD=many noisy steps')
ax2d.legend(fontsize=7)

# ── Loss vs examples processed ────────────────
ax_eff = fig.add_subplot(133)
ax_eff.plot(batch_examples_seen, batch_losses,
            'o-', color='red', lw=2, markersize=6,
            label=f'Batch GD\n({BATCH_EPOCHS} updates total)')
ax_eff.plot(sgd_examples_seen, sgd_losses,
            color='orange', lw=1.5, alpha=0.9,
            label=f'SGD\n({num_updates:,} updates total)')
ax_eff.set_xlabel('Examples processed  (= computational cost)')
ax_eff.set_ylabel('Training loss  J(w)')
ax_eff.set_title('THE KEY PLOT\nSame x = same compute budget\nSGD reaches low loss far sooner')
ax_eff.legend(fontsize=8)
ax_eff.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sgd_vs_batch.png', dpi=150, bbox_inches='tight')
print("\nFigure 1 saved → sgd_vs_batch.png")

# ─────────────────────────────────────────────
# FIGURE 2 — Snapshot comparison
#
# At each checkpoint (same examples processed),
# show where EACH algorithm's weights currently are.
# Batch GD has barely moved. SGD has nearly converged.
# This makes the efficiency gap concrete.
# ─────────────────────────────────────────────
checkpoints    = [100, 500, 1_000, 3_000, 6_000, 9_000]
fig2, axes2    = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle(
    "Snapshot comparison — where is each algorithm after N examples processed?",
    fontsize=13, fontweight='bold'
)

for ax, cp in zip(axes2.ravel(), checkpoints):
    ax.contourf(W1, W2, J_surface, levels=30, cmap='viridis')
    ax.contour( W1, W2, J_surface, levels=30, colors='white', linewidths=0.3, alpha=0.3)
    ax.scatter(*TRUE_WEIGHTS, color='cyan', s=100, marker='*', zorder=5, label='True min')

    # Batch GD position: cp examples = cp/NUM_SAMPLES epochs (may be <1 update)
    batch_updates_done = cp // NUM_SAMPLES          # integer division
    if batch_updates_done < len(batch_trajectory):
        w_b = batch_trajectory[batch_updates_done]
    else:
        w_b = batch_trajectory[-1]
    ax.scatter(w_b[0], w_b[1], color='red', s=120, zorder=6,
               label=f'Batch GD\nw=[{w_b[0]:.2f},{w_b[1]:.2f}]\n({batch_updates_done} updates)')

    # SGD position: find closest recorded step
    sgd_idx = np.searchsorted(sgd_examples_seen, cp)
    sgd_idx = min(sgd_idx, len(sgd_trajectory) - 1)
    w_s     = sgd_trajectory[sgd_idx]
    sgd_updates_done = sgd_examples_seen[sgd_idx]
    ax.scatter(w_s[0], w_s[1], color='orange', s=120, zorder=6,
               label=f'SGD\nw=[{w_s[0]:.2f},{w_s[1]:.2f}]\n({sgd_updates_done} updates)')

    ax.set_title(f'After {cp:,} examples processed', fontsize=9, fontweight='bold')
    ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
    ax.legend(fontsize=6.5, loc='upper right')

plt.tight_layout()
plt.savefig('sgd_snapshots.png', dpi=150, bbox_inches='tight')
print("Figure 2 saved → sgd_snapshots.png")
plt.show()
