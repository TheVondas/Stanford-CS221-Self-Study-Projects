import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Configuration
#
# We use 2 weights (d=2) so the loss surface
# J(w1, w2) can be drawn in 3D:
#   x-axis = w1,  y-axis = w2,  z-axis = loss
# This is exactly the "bowl" picture from the notes.
# ─────────────────────────────────────────────
TRUE_WEIGHTS   = np.array([2.0, -3.5])   # 2D so we can visualise the loss surface
NUM_SAMPLES    = 200
NOISE_STD      = 0.5
ETA            = 0.05
NUM_ITERATIONS = 300
PRINT_EVERY    = 30
SEED           = 42

# Snapshot iterations to show on the 3D surface plots
SNAPSHOTS = [1, 5, 20, 60, 150, 300]

np.random.seed(SEED)

# ─────────────────────────────────────────────
# Generate synthetic data
# ─────────────────────────────────────────────
d = len(TRUE_WEIGHTS)
X = np.random.randn(NUM_SAMPLES, d)
y = X @ TRUE_WEIGHTS + np.random.randn(NUM_SAMPLES) * NOISE_STD

# ─────────────────────────────────────────────
# Loss and gradient (from notes — Slide 9)
#   J(w)  = (1/2n) * sum_i (w·x_i - y_i)^2
#   ∇J(w) = (1/n)  * X^T (Xw - y)
#
# The gradient is a VECTOR in weight space.
# It points in the direction of steepest INCREASE of J.
# So we step in the NEGATIVE gradient direction to go downhill.
# ─────────────────────────────────────────────
def compute_loss(w, X, y):
    residuals = X @ w - y
    return (1 / (2 * len(y))) * np.sum(residuals ** 2)

def compute_gradient(w, X, y):
    residuals = X @ w - y
    return (1 / len(y)) * X.T @ residuals

# ─────────────────────────────────────────────
# Gradient Descent — record full trajectory
#   w ← w - η∇J(w)
# ─────────────────────────────────────────────
w = np.zeros(d)
trajectory = [w.copy()]   # store every weight vector
losses     = []

print("=" * 60)
print("Gradient Descent — Linear Regression (2D weights)")
print("=" * 60)
print(f"True weights  : {TRUE_WEIGHTS}")
print(f"Learning rate : {ETA}    Iterations: {NUM_ITERATIONS}")
print("=" * 60)
print(f"{'Iter':>6}  {'Loss':>12}  {'||gradient||':>14}  {'weights'}")
print("-" * 60)

for t in range(1, NUM_ITERATIONS + 1):
    grad = compute_gradient(w, X, y)
    w    = w - ETA * grad
    loss = compute_loss(w, X, y)

    trajectory.append(w.copy())
    losses.append(loss)

    if t % PRINT_EVERY == 0 or t == 1:
        grad_norm = np.linalg.norm(grad)
        w_str = "[" + ", ".join(f"{v:+.4f}" for v in w) + "]"
        print(f"{t:>6}  {loss:>12.6f}  {grad_norm:>14.6f}  {w_str}")

print("=" * 60)
print(f"Final weights : {w.round(4)}")
print(f"True weights  : {TRUE_WEIGHTS}")
print(f"Final loss    : {compute_loss(w, X, y):.6f}")

trajectory = np.array(trajectory)   # shape (NUM_ITERATIONS+1, 2)

# ─────────────────────────────────────────────
# Build the loss surface grid
#   Evaluate J(w1, w2) over a grid so we can
#   draw the bowl shape.
# ─────────────────────────────────────────────
w1_range = np.linspace(TRUE_WEIGHTS[0] - 4, TRUE_WEIGHTS[0] + 4, 80)
w2_range = np.linspace(TRUE_WEIGHTS[1] - 4, TRUE_WEIGHTS[1] + 4, 80)
W1, W2   = np.meshgrid(w1_range, w2_range)

J_surface = np.array([
    compute_loss(np.array([w1, w2]), X, y)
    for w1, w2 in zip(W1.ravel(), W2.ravel())
]).reshape(W1.shape)

# ─────────────────────────────────────────────
# Figure 1 — Full picture
#   Left:   3D loss surface + full GD trajectory
#   Middle: 2D contour view (top-down) + path + gradient arrows
#   Right:  Loss vs iteration
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
fig.suptitle(
    "Gradient Descent on the Loss Surface  J(w₁, w₂) = (1/2n)Σ(w·xᵢ − yᵢ)²",
    fontsize=13, fontweight='bold'
)

# ── 3D surface + trajectory ───────────────────
ax3d = fig.add_subplot(131, projection='3d')
ax3d.plot_surface(W1, W2, J_surface, cmap='viridis', alpha=0.6, zorder=0)

# trajectory on the surface
traj_losses = np.array([compute_loss(trajectory[i], X, y)
                         for i in range(len(trajectory))])
ax3d.plot(trajectory[:, 0], trajectory[:, 1], traj_losses,
          color='red', lw=2, zorder=5, label='GD path')
ax3d.scatter(trajectory[0, 0], trajectory[0, 1], traj_losses[0],
             color='orange', s=60, zorder=6, label='Start w=[0,0]')
ax3d.scatter(TRUE_WEIGHTS[0], TRUE_WEIGHTS[1],
             compute_loss(TRUE_WEIGHTS, X, y),
             color='cyan', s=80, marker='*', zorder=6, label='True minimum')

ax3d.set_xlabel('w₁')
ax3d.set_ylabel('w₂')
ax3d.set_zlabel('J(w)')
ax3d.set_title('3D Loss Surface\n(bowl = convex, one global minimum)')
ax3d.legend(fontsize=7)

# ── 2D contour + path + gradient arrows ───────
ax2d = fig.add_subplot(132)
contour = ax2d.contourf(W1, W2, J_surface, levels=40, cmap='viridis')
plt.colorbar(contour, ax=ax2d, label='J(w)')
ax2d.plot(trajectory[:, 0], trajectory[:, 1],
          'r-o', markersize=2, lw=1.5, label='GD path')
ax2d.scatter(*trajectory[0],  color='orange', s=80, zorder=5, label='Start')
ax2d.scatter(*TRUE_WEIGHTS,   color='cyan',   s=100, marker='*', zorder=5, label='True min')

# Draw negative gradient arrows every 20 steps
for i in range(0, NUM_ITERATIONS, 20):
    g = compute_gradient(trajectory[i], X, y)
    ax2d.annotate('', xy=trajectory[i] - 0.4 * g / (np.linalg.norm(g) + 1e-8),
                  xytext=trajectory[i],
                  arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

ax2d.set_xlabel('w₁')
ax2d.set_ylabel('w₂')
ax2d.set_title('Contour view (top-down)\nWhite arrows = −∇J  (step direction)')
ax2d.legend(fontsize=7)

# ── Loss vs iteration ─────────────────────────
ax_loss = fig.add_subplot(133)
ax_loss.plot(range(1, NUM_ITERATIONS + 1), losses, color='royalblue', lw=2)
ax_loss.set_xlabel('Iteration')
ax_loss.set_ylabel('J(w)')
ax_loss.set_title('Loss over iterations\n(should decrease monotonically)')
ax_loss.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gd_full_view.png', dpi=150, bbox_inches='tight')
print("\nFigure 1 saved → gd_full_view.png")

# ─────────────────────────────────────────────
# Figure 2 — Snapshot series
#   6 panels, one per snapshot iteration.
#   Each shows the 3D bowl with the current
#   weight position marked — you can watch
#   the red dot roll down into the minimum.
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9),
                            subplot_kw={'projection': '3d'})
fig2.suptitle(
    "Snapshot Series — watch the weights roll down to the minimum",
    fontsize=13, fontweight='bold'
)

for ax, snap in zip(axes2.ravel(), SNAPSHOTS):
    w_snap      = trajectory[snap]
    loss_snap   = compute_loss(w_snap, X, y)
    grad_snap   = compute_gradient(w_snap, X, y)

    ax.plot_surface(W1, W2, J_surface, cmap='viridis', alpha=0.5)

    # path up to this snapshot
    traj_so_far = trajectory[:snap + 1]
    losses_so_far = np.array([compute_loss(traj_so_far[i], X, y)
                               for i in range(len(traj_so_far))])
    ax.plot(traj_so_far[:, 0], traj_so_far[:, 1], losses_so_far,
            color='red', lw=1.5)

    # current position
    ax.scatter(w_snap[0], w_snap[1], loss_snap,
               color='red', s=60, zorder=10)

    # gradient arrow: from current point in direction of -grad (projected onto surface)
    arrow_scale = 0.3
    neg_g = -grad_snap / (np.linalg.norm(grad_snap) + 1e-8)
    ax.quiver(w_snap[0], w_snap[1], loss_snap,
              neg_g[0] * arrow_scale, neg_g[1] * arrow_scale, 0,
              color='yellow', lw=2, arrow_length_ratio=0.3)

    ax.set_xlabel('w₁', fontsize=7)
    ax.set_ylabel('w₂', fontsize=7)
    ax.set_zlabel('J(w)', fontsize=7)
    ax.set_title(
        f"Iteration {snap}\n"
        f"w=[{w_snap[0]:.2f}, {w_snap[1]:.2f}]  J={loss_snap:.3f}\n"
        f"‖∇J‖={np.linalg.norm(grad_snap):.3f}",
        fontsize=8
    )

plt.tight_layout()
plt.savefig('gd_snapshots.png', dpi=150, bbox_inches='tight')
print("Figure 2 saved → gd_snapshots.png")
plt.show()
