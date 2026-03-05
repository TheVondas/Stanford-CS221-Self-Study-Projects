import numpy as np

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
TRUE_WEIGHTS   = np.array([2.0, -3.5, 1.2])   # ground truth weights we want to recover
NUM_SAMPLES    = 200                            # number of data points to generate
NOISE_STD      = 0.5                            # std dev of Gaussian noise added to labels
ETA            = 0.01                           # step size (learning rate)
NUM_ITERATIONS = 300                            # number of gradient descent steps
PRINT_EVERY    = 30                             # print progress every N iterations
SEED           = 42

np.random.seed(SEED)

# ─────────────────────────────────────────────
# Generate synthetic data
#   X : (n, d)  feature matrix, drawn from N(0, 1)
#   y : (n,)    labels = X @ w_true + noise
# ─────────────────────────────────────────────
d = len(TRUE_WEIGHTS)
X = np.random.randn(NUM_SAMPLES, d)
y = X @ TRUE_WEIGHTS + np.random.randn(NUM_SAMPLES) * NOISE_STD

# ─────────────────────────────────────────────
# Loss and gradient (from notes — Slide 9)
#   J(w)  = (1/2n) * sum_i (w·x_i - y_i)^2
#   ∇J(w) = (1/n)  * sum_i (w·x_i - y_i) * x_i
#         = (1/n)  * X^T (Xw - y)
#
# The 1/2 factor cancels the 2 from the chain rule,
# giving a clean gradient with no leading constant.
# ─────────────────────────────────────────────
def compute_loss(w, X, y):
    residuals = X @ w - y
    return (1 / (2 * len(y))) * np.sum(residuals ** 2)

def compute_gradient(w, X, y):
    residuals = X @ w - y
    return (1 / len(y)) * X.T @ residuals

# ─────────────────────────────────────────────
# Gradient Descent
#   w ← w - η∇J(w)
# ─────────────────────────────────────────────
w = np.zeros(d)   # initialise weights at zero

print("=" * 60)
print("Gradient Descent — Linear Regression")
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

    if t % PRINT_EVERY == 0 or t == 1:
        grad_norm = np.linalg.norm(grad)
        w_str = "[" + ", ".join(f"{v:+.4f}" for v in w) + "]"
        print(f"{t:>6}  {loss:>12.6f}  {grad_norm:>14.6f}  {w_str}")

print("=" * 60)
print(f"Final weights : {w.round(4)}")
print(f"True weights  : {TRUE_WEIGHTS}")
print(f"Final loss    : {compute_loss(w, X, y):.6f}")
