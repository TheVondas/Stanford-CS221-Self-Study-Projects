import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Exact data from lecture notes
#   Group A: 2 examples  (minority group)
#   Group B: 4 examples  (majority group)
#
#   Predictor: f_w(x) = w * x   (scalar w, no intercept)
#   φ(x) = x  (identity feature map)
#   NOTE: group label g is NOT used by the predictor,
#         only by the learning objective.
# ─────────────────────────────────────────────
X_A = np.array([1.0, 2.0])
y_A = np.array([4.0, 8.0])

X_B = np.array([5.0, 6.0, 7.0, 8.0])
y_B = np.array([5.0, 6.0, 7.0, 8.0])

X_all = np.concatenate([X_A, X_B])
y_all = np.concatenate([y_A, y_B])

ETA            = 0.001    # step size
NUM_ITERATIONS = 2000     # iterations (more needed — scalar problem is sensitive)
PRINT_EVERY    = 200

# ─────────────────────────────────────────────
# Loss functions (from notes)
#
# Per-example loss:
#   Loss(x, y, w) = (w*x - y)^2
#
# Per-group loss:
#   TrainLoss_g(w) = (1/|D_g|) * sum_{i in g} (w*x_i - y_i)^2
#
# Average loss:
#   TrainLoss(w) = (1/n) * sum_i (w*x_i - y_i)^2
#
# Maximum group loss:
#   TrainLoss_max(w) = max_g TrainLoss_g(w)
# ─────────────────────────────────────────────
def group_loss(w, X_g, y_g):
    return np.mean((w * X_g - y_g) ** 2)

def avg_loss(w):
    return np.mean((w * X_all - y_all) ** 2)

def max_group_loss(w):
    return max(group_loss(w, X_A, y_A), group_loss(w, X_B, y_B))

# ─────────────────────────────────────────────
# Gradients
#
# Per-group gradient:
#   ∇TrainLoss_g(w) = (2/|D_g|) * sum_{i in g} (w*x_i - y_i) * x_i
#
# Average gradient:
#   ∇TrainLoss(w) = (2/n) * sum_i (w*x_i - y_i) * x_i
#
# Group DRO gradient (from notes):
#   g* = argmax_g TrainLoss_g(w)
#   ∇TrainLoss_max(w) = ∇TrainLoss_g*(w)
#
# Intuition: update using gradient from the worst group —
# "help the group that needs it most"
# ─────────────────────────────────────────────
def group_gradient(w, X_g, y_g):
    return np.mean(2 * (w * X_g - y_g) * X_g)

def avg_gradient(w):
    return np.mean(2 * (w * X_all - y_all) * X_all)

def dro_gradient(w):
    loss_A = group_loss(w, X_A, y_A)
    loss_B = group_loss(w, X_B, y_B)
    if loss_A >= loss_B:
        return group_gradient(w, X_A, y_A), 'A'
    else:
        return group_gradient(w, X_B, y_B), 'B'

# ─────────────────────────────────────────────
# Standard Gradient Descent (minimize average loss)
#   w ← w - η * ∇TrainLoss(w)
# ─────────────────────────────────────────────
print("=" * 65)
print("STANDARD GRADIENT DESCENT  (minimise average loss)")
print("=" * 65)
print(f"{'Iter':>6}  {'AvgLoss':>10}  {'Loss_A':>10}  {'Loss_B':>10}  {'w':>8}")
print("-" * 65)

w_std = 0.0
std_avg_losses, std_lossA, std_lossB, std_ws = [], [], [], []

for t in range(1, NUM_ITERATIONS + 1):
    grad = avg_gradient(w_std)
    w_std = w_std - ETA * grad

    std_avg_losses.append(avg_loss(w_std))
    std_lossA.append(group_loss(w_std, X_A, y_A))
    std_lossB.append(group_loss(w_std, X_B, y_B))
    std_ws.append(w_std)

    if t % PRINT_EVERY == 0 or t == 1:
        print(f"{t:>6}  {avg_loss(w_std):>10.4f}  "
              f"{group_loss(w_std, X_A, y_A):>10.4f}  "
              f"{group_loss(w_std, X_B, y_B):>10.4f}  "
              f"{w_std:>8.4f}")

print("-" * 65)
print(f"Standard GD final w = {w_std:.4f}  (lecture says ~1.09)")
print(f"  Avg loss   : {avg_loss(w_std):.4f}")
print(f"  Loss_A     : {group_loss(w_std, X_A, y_A):.4f}")
print(f"  Loss_B     : {group_loss(w_std, X_B, y_B):.4f}")
print(f"  Max loss   : {max_group_loss(w_std):.4f}")

# ─────────────────────────────────────────────
# Group DRO Gradient Descent (minimize max group loss)
#   g* = argmax_g TrainLoss_g(w)
#   w  ← w - η * ∇TrainLoss_g*(w)
# ─────────────────────────────────────────────
print()
print("=" * 65)
print("GROUP DRO GRADIENT DESCENT  (minimise max group loss)")
print("=" * 65)
print(f"{'Iter':>6}  {'MaxLoss':>10}  {'Loss_A':>10}  {'Loss_B':>10}  {'w':>8}  {'worst g'}")
print("-" * 65)

w_dro = 0.0
dro_max_losses, dro_lossA, dro_lossB, dro_ws = [], [], [], []

for t in range(1, NUM_ITERATIONS + 1):
    grad, worst_g = dro_gradient(w_dro)
    w_dro = w_dro - ETA * grad

    dro_max_losses.append(max_group_loss(w_dro))
    dro_lossA.append(group_loss(w_dro, X_A, y_A))
    dro_lossB.append(group_loss(w_dro, X_B, y_B))
    dro_ws.append(w_dro)

    if t % PRINT_EVERY == 0 or t == 1:
        print(f"{t:>6}  {max_group_loss(w_dro):>10.4f}  "
              f"{group_loss(w_dro, X_A, y_A):>10.4f}  "
              f"{group_loss(w_dro, X_B, y_B):>10.4f}  "
              f"{w_dro:>8.4f}  {worst_g}")

print("-" * 65)
print(f"Group DRO final w  = {w_dro:.4f}  (lecture says ~1.58)")
print(f"  Avg loss   : {avg_loss(w_dro):.4f}")
print(f"  Loss_A     : {group_loss(w_dro, X_A, y_A):.4f}")
print(f"  Loss_B     : {group_loss(w_dro, X_B, y_B):.4f}")
print(f"  Max loss   : {max_group_loss(w_dro):.4f}")

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
x_line = np.linspace(0, 9, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Group DRO vs Standard Learning", fontsize=14, fontweight='bold')

# ── Left: data + trend lines ──────────────────
ax = axes[0]
ax.scatter(X_A, y_A, color='royalblue',  s=100, zorder=5, label='Group A (minority)')
ax.scatter(X_B, y_B, color='tomato',     s=100, zorder=5, label='Group B (majority)')
ax.plot(x_line, w_std * x_line, color='green',  lw=2,
        label=f'Standard GD  w={w_std:.3f}')
ax.plot(x_line, w_dro * x_line, color='purple', lw=2, linestyle='--',
        label=f'Group DRO    w={w_dro:.3f}')
ax.set_xlabel('x')
ax.set_ylabel('y  /  prediction')
ax.set_title('Data & Fitted Lines\n(predictor: f(x) = w·x)')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Right: per-group loss over iterations ────
ax2 = axes[1]
iters = np.arange(1, NUM_ITERATIONS + 1)
ax2.plot(iters, std_lossA,   color='royalblue', lw=1.5,
         label='Std GD — Loss_A (minority)')
ax2.plot(iters, std_lossB,   color='tomato',    lw=1.5,
         label='Std GD — Loss_B (majority)')
ax2.plot(iters, dro_lossA,   color='royalblue', lw=1.5, linestyle='--',
         label='DRO — Loss_A (minority)')
ax2.plot(iters, dro_lossB,   color='tomato',    lw=1.5, linestyle='--',
         label='DRO — Loss_B (majority)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Per-group loss')
ax2.set_title('Per-group Loss Over Training\n(solid = Standard GD, dashed = Group DRO)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('groupDRO_plot.png', dpi=150, bbox_inches='tight')
print()
print("Plot saved to groupDRO_plot.png")
plt.show()
