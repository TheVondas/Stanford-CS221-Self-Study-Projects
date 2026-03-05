import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ─────────────────────────────────────────────
# Configuration
#
# From notes (lecture example):
#   φ(x) = [x₁, x₂, x₁² + x₂²]
#   f(x)  = sign([2, 2, -1] · φ(x))
#
# This gives a CIRCULAR decision boundary in input space:
#   (x₁ - 1)² + (x₂ - 1)² = 2
# ...but a LINEAR separator (hyperplane) in feature space.
#
# Key message from notes:
#   "Linear in w? YES.  Linear in φ(x)? YES.  Linear in x? NO."
# ─────────────────────────────────────────────
TRUE_WEIGHTS  = np.array([2.0, 2.0, -1.0])   # from notes
NUM_SAMPLES   = 700
ETA           = 0.005
NUM_ITERS     = 800
SEED          = 42

np.random.seed(SEED)

# ─────────────────────────────────────────────
# Non-linear feature map (from notes)
#   φ(x) = [x₁, x₂, x₁² + x₂²]
#
# The learning algorithm only ever sees φ(x).
# It has no idea φ was constructed non-linearly from x.
# ─────────────────────────────────────────────
def phi(X):
    return np.column_stack([X[:, 0], X[:, 1],
                             X[:, 0]**2 + X[:, 1]**2])

# ─────────────────────────────────────────────
# Generate data
#   Circular boundary: (x₁-1)² + (x₂-1)² = 2
#   Inside  → y = +1
#   Outside → y = -1
# ─────────────────────────────────────────────
X_raw      = np.random.uniform(-2.5, 4.5, (NUM_SAMPLES, 2))
Phi_data   = phi(X_raw)
y          = np.sign(Phi_data @ TRUE_WEIGHTS)
y[y == 0]  = 1

# ─────────────────────────────────────────────
# Hinge loss + gradient (same machinery as linearClassification.py)
#
#   margin          = (w · φ(x)) · y
#   Loss_hinge      = max(1 − margin, 0)
#   ∇Loss_hinge     = −φ(x)·y   if margin < 1,  else 0
#   ∇TrainLoss(w)   = (1/n) Σ ∇Loss_hinge
#
# The ONLY change from linear classification: φ(x) is now non-linear.
# The gradient descent update rule w ← w − η∇J(w) is IDENTICAL.
# ─────────────────────────────────────────────
def train_loss(w, Phi, y):
    margins = (Phi @ w) * y
    return np.mean(np.maximum(1 - margins, 0))

def train_gradient(w, Phi, y):
    margins  = (Phi @ w) * y
    violated = (margins < 1).astype(float)
    return np.mean(-Phi * (y * violated)[:, np.newaxis], axis=0)

def zero_one_loss(w, Phi, y):
    return np.mean(np.sign(Phi @ w) != y)

# ─────────────────────────────────────────────
# Gradient Descent — record full trajectory
# ─────────────────────────────────────────────
w          = np.zeros(3)
trajectory = [w.copy()]
losses, zo_losses = [], []

print("=" * 60)
print("Non-Linear Features — Quadratic Classifier")
print(f"  φ(x) = [x₁, x₂, x₁² + x₂²]   (from notes)")
print(f"  True weights: {TRUE_WEIGHTS}")
print("=" * 60)
print(f"{'Iter':>6}  {'HingeLoss':>10}  {'0-1 Loss':>10}  {'weights'}")
print("-" * 60)

for t in range(1, NUM_ITERS + 1):
    grad = train_gradient(w, Phi_data, y)
    w    = w - ETA * grad
    losses.append(train_loss(w, Phi_data, y))
    zo_losses.append(zero_one_loss(w, Phi_data, y))
    trajectory.append(w.copy())

    if t % 100 == 0 or t == 1:
        w_str = "[" + ", ".join(f"{v:+.3f}" for v in w) + "]"
        print(f"{t:>6}  {losses[-1]:>10.5f}  {zo_losses[-1]:>10.4f}  {w_str}")

trajectory = np.array(trajectory)
print("-" * 60)
print(f"Final weights : {w.round(4)}")
print(f"True weights  : {TRUE_WEIGHTS}")
print(f"Final hinge loss : {losses[-1]:.5f}")
print(f"Final 0-1 loss   : {zo_losses[-1]:.4f}")

# ─────────────────────────────────────────────
# FIGURE 1 — The classifier in two spaces
#
# Panel 1 (2D input space):
#   Scatter of data + learned circular decision boundary
#   Shows the NON-LINEAR boundary in input space
#
# Panel 2 (3D feature space):
#   Data lifted to φ-space + linear separator PLANE
#   Shows that in feature space the boundary is LINEAR
#
# Panel 3: Loss + 0-1 loss over iterations
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
fig.suptitle(
    "Non-Linear Features: circular decision boundary in input space "
    "= linear separator in feature space",
    fontsize=12, fontweight='bold'
)

# ── Panel 1: Input space ──────────────────────
ax1 = fig.add_subplot(131)
pos = y ==  1
neg = y == -1
ax1.scatter(X_raw[pos, 0], X_raw[pos, 1], c='royalblue', s=15, alpha=0.6, label='+1 (inside)')
ax1.scatter(X_raw[neg, 0], X_raw[neg, 1], c='tomato',    s=15, alpha=0.6, label='−1 (outside)')

# Learned decision boundary: solve w·φ(x) = 0 on a grid
x1g = np.linspace(-2.5, 4.5, 300)
x2g = np.linspace(-2.5, 4.5, 300)
X1G, X2G = np.meshgrid(x1g, x2g)
Phi_grid  = phi(np.column_stack([X1G.ravel(), X2G.ravel()]))
scores    = (Phi_grid @ w).reshape(X1G.shape)
ax1.contour(X1G, X2G, scores, levels=[0], colors='black', linewidths=2)

# True circular boundary: (x₁-1)² + (x₂-1)² = 2
true_circle = Circle((1, 1), np.sqrt(2), fill=False,
                      edgecolor='green', linewidth=2, linestyle='--')
ax1.add_patch(true_circle)
ax1.plot([], [], 'k-',  lw=2, label='Learned boundary')
ax1.plot([], [], 'g--', lw=2, label='True boundary (circle)')
ax1.set_xlim(-2.5, 4.5); ax1.set_ylim(-2.5, 4.5)
ax1.set_xlabel('x₁'); ax1.set_ylabel('x₂')
ax1.set_title('Input space (x₁, x₂)\nNon-linear boundary in x')
ax1.legend(fontsize=7); ax1.set_aspect('equal')

# ── Panel 2: Feature space (3D) ───────────────
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(Phi_data[pos, 0], Phi_data[pos, 1], Phi_data[pos, 2],
            c='royalblue', s=8, alpha=0.5, label='+1')
ax2.scatter(Phi_data[neg, 0], Phi_data[neg, 1], Phi_data[neg, 2],
            c='tomato',    s=8, alpha=0.5, label='−1')

# Draw the separating hyperplane w·[φ₁,φ₂,φ₃] = 0
# → φ₃ = -(w₁φ₁ + w₂φ₂) / w₃
phi1g = np.linspace(-2.5, 4.5, 30)
phi2g = np.linspace(-2.5, 4.5, 30)
P1G, P2G = np.meshgrid(phi1g, phi2g)
if abs(w[2]) > 1e-6:
    P3G = -(w[0] * P1G + w[1] * P2G) / w[2]
    ax2.plot_surface(P1G, P2G, P3G, alpha=0.25, color='yellow')

ax2.set_xlabel('φ₁ = x₁'); ax2.set_ylabel('φ₂ = x₂')
ax2.set_zlabel('φ₃ = x₁²+x₂²')
ax2.set_title('Feature space (φ₁, φ₂, φ₃)\nLinear separator becomes a PLANE here')
ax2.legend(fontsize=7)

# ── Panel 3: Loss over iterations ─────────────
ax3 = fig.add_subplot(133)
iters = range(1, NUM_ITERS + 1)
ax3.plot(iters, losses,    color='royalblue', lw=2, label='Hinge loss')
ax3.plot(iters, zo_losses, color='tomato',    lw=2, label='0-1 loss')
ax3.set_xlabel('Iteration'); ax3.set_ylabel('Loss')
ax3.set_title('Gradient descent still works perfectly\nwith non-linear features (same algorithm!)')
ax3.legend(); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nonlinear_classifier.png', dpi=150, bbox_inches='tight')
print("\nFigure 1 saved → nonlinear_classifier.png")

# ─────────────────────────────────────────────
# FIGURE 2 — Convex vs Non-convex loss surface
#
# LEFT: Convex loss surface from our quadratic classifier
#   One smooth bowl. Gradient descent always finds the
#   global minimum regardless of starting point.
#
# RIGHT: Non-convex surface (what complex models look like)
#   Multiple local minima. Gradient descent can get STUCK
#   depending on where you start.
#
# This contrast explains WHY convex objectives (linear models
# + hinge/squared loss) are so desirable.
# ─────────────────────────────────────────────

# Build convex surface: fix w₃ = true w₃, vary w₁, w₂
w3_fixed = w[2]
w1r = np.linspace(-1, 5, 70)
w2r = np.linspace(-1, 5, 70)
W1C, W2C = np.meshgrid(w1r, w2r)
J_convex = np.array([
    train_loss(np.array([a, b, w3_fixed]), Phi_data, y)
    for a, b in zip(W1C.ravel(), W2C.ravel())
]).reshape(W1C.shape)

# Build non-convex surface (educational demo)
# J(w1,w2) = (w1² − 2.5)²·0.3 + 0.5·w2² − 0.2·w1
#
# This creates:
#   Global minimum near w1 ≈ +1.6, w2 ≈ 0  (right well)
#   Local minimum  near w1 ≈ −1.6, w2 ≈ 0  (left well)
#
# Gradient descent from different starts hits different minima.
w1r_nc = np.linspace(-3, 3, 80)
w2r_nc = np.linspace(-3, 3, 80)
W1NC, W2NC = np.meshgrid(w1r_nc, w2r_nc)

def nonconvex(w1, w2):
    return 0.3 * (w1**2 - 2.5)**2 + 0.5 * w2**2 - 0.2 * w1

def nonconvex_gradient(w1, w2):
    dw1 = 0.3 * 4 * w1 * (w1**2 - 2.5) - 0.2
    dw2 = w2
    return np.array([dw1, dw2])

J_nc = nonconvex(W1NC, W2NC)

# Run gradient descent from 3 starting points on non-convex surface
starts_nc   = [np.array([-2.0,  1.5]),
               np.array([ 0.0,  2.5]),
               np.array([ 2.0,  1.5])]
colors_nc   = ['red', 'orange', 'lime']
labels_nc   = ['Start A → local min  (STUCK)',
               'Start B → depends on path',
               'Start C → global min (SUCCESS)']
eta_nc      = 0.05
iters_nc    = 300
paths_nc    = []
for start in starts_nc:
    wp = start.copy()
    path = [wp.copy()]
    for _ in range(iters_nc):
        g  = nonconvex_gradient(wp[0], wp[1])
        wp = wp - eta_nc * g
        path.append(wp.copy())
    paths_nc.append(np.array(path))

# Run gradient descent from 3 starting points on convex surface
starts_cvx = [np.array([-0.5,  4.0]),
              np.array([ 4.5,  4.0]),
              np.array([ 2.0, -0.5])]
colors_cvx = ['red', 'orange', 'lime']
eta_cvx    = 0.05
iters_cvx  = 200
paths_cvx  = []
for start in starts_cvx:
    wp   = np.array([start[0], start[1], w3_fixed])
    path = [wp[:2].copy()]
    for _ in range(iters_cvx):
        g  = train_gradient(wp, Phi_data, y)
        wp = wp - eta_cvx * g
        path.append(wp[:2].copy())
    paths_cvx.append(np.array(path))

fig2, axes = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle(
    "Convex vs Non-Convex Loss Surface\n"
    "Why convex objectives (linear models + hinge/squared loss) are so desirable",
    fontsize=12, fontweight='bold'
)

# ── Top-left: Convex 3D surface ───────────────
ax_c3d = fig2.add_subplot(221, projection='3d')
ax_c3d.plot_surface(W1C, W2C, J_convex, cmap='viridis', alpha=0.6)
ax_c3d.set_xlabel('w₁'); ax_c3d.set_ylabel('w₂'); ax_c3d.set_zlabel('J(w)')
ax_c3d.set_title('CONVEX loss surface\n(quadratic features + hinge loss)\nOne smooth bowl — one global minimum')

# ── Top-right: Non-convex 3D surface ──────────
ax_nc3d = fig2.add_subplot(222, projection='3d')
ax_nc3d.plot_surface(W1NC, W2NC, J_nc, cmap='plasma', alpha=0.6)
ax_nc3d.set_xlabel('w₁'); ax_nc3d.set_ylabel('w₂'); ax_nc3d.set_zlabel('J(w)')
ax_nc3d.set_title('NON-CONVEX loss surface\n(complex models e.g. neural networks)\nMultiple valleys — local minima trap GD')

# ── Bottom-left: Convex contour + GD paths ────
ax_cc = fig2.add_subplot(223)
cf = ax_cc.contourf(W1C, W2C, J_convex, levels=40, cmap='viridis')
ax_cc.contour(W1C, W2C, J_convex, levels=40, colors='white', linewidths=0.3, alpha=0.4)
plt.colorbar(cf, ax=ax_cc, label='J(w)')

for path, col in zip(paths_cvx, colors_cvx):
    ax_cc.plot(path[:, 0], path[:, 1], color=col, lw=2)
    ax_cc.scatter(*path[0],  color=col,   s=80, zorder=5, marker='o')
    ax_cc.scatter(*path[-1], color='cyan', s=80, zorder=6, marker='*')

ax_cc.scatter([], [], color='cyan', marker='*', s=80,
              label='All paths find the SAME minimum')
ax_cc.set_xlabel('w₁'); ax_cc.set_ylabel('w₂')
ax_cc.set_title('Convex: every starting point → global minimum\n(GD always wins on convex surfaces)')
ax_cc.legend(fontsize=8)

# ── Bottom-right: Non-convex contour + GD paths
ax_ncc = fig2.add_subplot(224)
cf2 = ax_ncc.contourf(W1NC, W2NC, J_nc, levels=40, cmap='plasma')
ax_ncc.contour(W1NC, W2NC, J_nc, levels=40, colors='white', linewidths=0.3, alpha=0.4)
plt.colorbar(cf2, ax=ax_ncc, label='J(w)')

for path, col, lbl in zip(paths_nc, colors_nc, labels_nc):
    ax_ncc.plot(path[:, 0], path[:, 1], color=col, lw=2, label=lbl)
    ax_ncc.scatter(*path[0],  color=col,    s=80, zorder=5, marker='o')
    ax_ncc.scatter(*path[-1], color='white', s=80, zorder=6, marker='*')

# Mark the two minima
ax_ncc.scatter( 1.6, 0, color='cyan',  s=120, marker='*', zorder=7, label='Global minimum')
ax_ncc.scatter(-1.6, 0, color='yellow', s=120, marker='X', zorder=7, label='Local minimum (TRAP)')
ax_ncc.set_xlabel('w₁'); ax_ncc.set_ylabel('w₂')
ax_ncc.set_title('Non-convex: starting point determines outcome\nSome paths get STUCK in a local minimum')
ax_ncc.legend(fontsize=7)

plt.tight_layout()
plt.savefig('convex_vs_nonconvex.png', dpi=150, bbox_inches='tight')
print("Figure 2 saved → convex_vs_nonconvex.png")
plt.show()
