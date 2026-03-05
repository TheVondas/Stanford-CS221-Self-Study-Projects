import numpy as np
import time

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
TRUE_WEIGHTS    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # ground truth to recover
NUM_SAMPLES     = 1_000_000                               # large dataset (as in lecture demo)
NOISE_STD       = 0.1                                     # Gaussian noise on labels
SEED            = 42

# Batch GD settings
BATCH_EPOCHS    = 5        # batch GD is slow — only run a few epochs
BATCH_ETA       = 0.01     # constant step size for batch GD
BATCH_PRINT_EVERY = 1      # print every epoch

# SGD settings
SGD_EPOCHS      = 1        # one pass through the data is enough (as shown in lecture)
SGD_PRINT_EVERY = 100_000  # print every 100k updates

np.random.seed(SEED)

# ─────────────────────────────────────────────
# Generate synthetic data
#   x_i ~ N(0, 1),  y_i = w_true · x_i + noise
# ─────────────────────────────────────────────
d = len(TRUE_WEIGHTS)
X = np.random.randn(NUM_SAMPLES, d)
y = X @ TRUE_WEIGHTS + np.random.randn(NUM_SAMPLES) * NOISE_STD

# ─────────────────────────────────────────────
# Loss functions (matching notes)
#
# Training loss (full batch):
#   TrainLoss(w) = (1/n) * sum_i Loss(x_i, y_i, w)
#
# Per-example loss:
#   Loss(x, y, w) = (1/2)(w·x - y)^2
#
# Per-example gradient:
#   ∇Loss(x, y, w) = (w·x - y) * x
# ─────────────────────────────────────────────
def per_example_loss(w, x, y_i):
    return 0.5 * (w @ x - y_i) ** 2

def per_example_gradient(w, x, y_i):
    return (w @ x - y_i) * x

def train_loss(w, X, y):
    residuals = X @ w - y
    return np.mean(0.5 * residuals ** 2)

def batch_gradient(w, X, y):
    residuals = X @ w - y
    return (1 / len(y)) * X.T @ residuals

# ─────────────────────────────────────────────
# Batch Gradient Descent
#   w ← w - η * ∇TrainLoss(w)
#   One update per epoch — slow on large datasets
# ─────────────────────────────────────────────
print("=" * 65)
print("BATCH GRADIENT DESCENT")
print("=" * 65)
print(f"  Dataset size : {NUM_SAMPLES:,}  |  d = {d}  |  η = {BATCH_ETA}")
print(f"  True weights : {TRUE_WEIGHTS}")
print("=" * 65)
print(f"{'Epoch':>6}  {'TrainLoss':>12}  {'||gradient||':>14}  {'weights'}  {'time(s)'}")
print("-" * 65)

w_batch = np.zeros(d)
for epoch in range(1, BATCH_EPOCHS + 1):
    t0   = time.time()
    grad = batch_gradient(w_batch, X, y)
    w_batch = w_batch - BATCH_ETA * grad
    loss = train_loss(w_batch, X, y)
    elapsed = time.time() - t0

    if epoch % BATCH_PRINT_EVERY == 0:
        w_str = "[" + ", ".join(f"{v:+.4f}" for v in w_batch) + "]"
        print(f"{epoch:>6}  {loss:>12.6f}  {np.linalg.norm(grad):>14.6f}  {w_str}  {elapsed:.2f}s")

print("-" * 65)
print(f"  Final weights : {w_batch.round(4)}")
print(f"  True weights  : {TRUE_WEIGHTS}")

# ─────────────────────────────────────────────
# Stochastic Gradient Descent
#   For each example (x_i, y_i):
#     w ← w - η * ∇Loss(x_i, y_i, w)
#
#   Decreasing step size (schedule from notes):
#     η = 1 / sqrt(num_updates_so_far)
#
#   Many cheap updates vs few expensive ones.
#   "Not about quality — it's about quantity."
# ─────────────────────────────────────────────
print()
print("=" * 65)
print("STOCHASTIC GRADIENT DESCENT")
print("=" * 65)
print(f"  Dataset size : {NUM_SAMPLES:,}  |  d = {d}  |  η = 1/sqrt(t)")
print(f"  True weights : {TRUE_WEIGHTS}")
print("=" * 65)
print(f"{'Update':>10}  {'AvgLoss(last 1k)':>18}  {'η':>8}  {'weights'}")
print("-" * 65)

w_sgd      = np.zeros(d)
num_updates = 0
rolling_loss = []

t0_sgd = time.time()
for epoch in range(SGD_EPOCHS):
    indices = np.random.permutation(NUM_SAMPLES)  # shuffle each epoch
    for i in indices:
        num_updates += 1
        eta = 1.0 / np.sqrt(num_updates)           # decreasing step size

        x_i  = X[i]
        y_i  = y[i]
        grad = per_example_gradient(w_sgd, x_i, y_i)
        w_sgd = w_sgd - eta * grad

        rolling_loss.append(per_example_loss(w_sgd, x_i, y_i))

        if num_updates % SGD_PRINT_EVERY == 0:
            avg_loss = np.mean(rolling_loss[-1000:])
            w_str = "[" + ", ".join(f"{v:+.4f}" for v in w_sgd) + "]"
            print(f"{num_updates:>10,}  {avg_loss:>18.6f}  {eta:>8.6f}  {w_str}")

elapsed_sgd = time.time() - t0_sgd

print("-" * 65)
print(f"  Final weights : {w_sgd.round(4)}")
print(f"  True weights  : {TRUE_WEIGHTS}")
print(f"  Total updates : {num_updates:,}  |  Time: {elapsed_sgd:.2f}s")
print("=" * 65)
