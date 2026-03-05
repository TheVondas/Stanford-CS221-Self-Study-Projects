import numpy as np

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
TRUE_WEIGHTS   = np.array([-0.6, 0.6])   # true decision boundary (from lecture example)
NUM_SAMPLES    = 500                      # synthetic training examples
ETA            = 0.01                     # step size (learning rate)
NUM_ITERATIONS = 300                      # gradient descent steps
PRINT_EVERY    = 30                       # print every N iterations
SEED           = 42

np.random.seed(SEED)

# ─────────────────────────────────────────────
# Generate synthetic data
#   x_i ~ N(0, 1)  in R^2
#   y_i = sign(w_true · x_i)   labels in {+1, -1}
#
# Feature map φ(x) = x  (identity, as in lecture)
# ─────────────────────────────────────────────
d = len(TRUE_WEIGHTS)
X = np.random.randn(NUM_SAMPLES, d)
y = np.sign(X @ TRUE_WEIGHTS)
y[y == 0] = 1   # resolve the rare sign(0) edge case

# ─────────────────────────────────────────────
# Score and margin (from notes)
#   score  = w · φ(x)
#   margin = (w · φ(x)) * y
#
# Interpretation:
#   score  → confidence in predicting +1
#   margin → how correct we are (positive = correct)
# ─────────────────────────────────────────────
def score(w, X):
    return X @ w

def margin(w, X, y):
    return score(w, X) * y

# ─────────────────────────────────────────────
# Zero-one loss (from notes)
#   Loss_0-1(x, y, w) = 1[margin ≤ 0]
#
# Cannot be optimized with gradient descent —
# gradient is zero almost everywhere.
# ─────────────────────────────────────────────
def zero_one_loss(w, X, y):
    return np.mean(margin(w, X, y) <= 0)

# ─────────────────────────────────────────────
# Hinge loss — surrogate for zero-one loss
#   Loss_hinge(x, y, w) = max(1 - margin, 0)
#
# Upper-bounds zero-one loss.
# If hinge loss = 0 → zero-one loss = 0.
#
# TrainLoss(w) = (1/n) * sum_i Loss_hinge(x_i, y_i, w)
# ─────────────────────────────────────────────
def hinge_loss_per_example(w, X, y):
    return np.maximum(1 - margin(w, X, y), 0)

def train_loss(w, X, y):
    return np.mean(hinge_loss_per_example(w, X, y))

# ─────────────────────────────────────────────
# Gradient of hinge loss (from notes — piecewise rule)
#
#   ∇Loss_hinge(x, y, w) = -φ(x) * y   if margin < 1
#                         =  0           otherwise
#
# Interpretation:
#   - margin ≥ 1 ("safe"): no update — example already correct with confidence
#   - margin < 1 (violated): push weights to increase margin
#
# Batch gradient:
#   ∇TrainLoss(w) = (1/n) * sum_i ∇Loss_hinge(x_i, y_i, w)
# ─────────────────────────────────────────────
def train_gradient(w, X, y):
    margins   = margin(w, X, y)
    violated  = (margins < 1).astype(float)           # 1 if margin violated, 0 if safe
    # per-example gradient contribution: -φ(x_i) * y_i  when violated
    grad_matrix = -X * (y * violated)[:, np.newaxis]  # shape (n, d)
    return np.mean(grad_matrix, axis=0)

# ─────────────────────────────────────────────
# Classifier: f_w(x) = sign(w · φ(x))
# ─────────────────────────────────────────────
def predict(w, X):
    return np.sign(score(w, X))

def accuracy(w, X, y):
    return np.mean(predict(w, X) == y)

# ─────────────────────────────────────────────
# Gradient Descent
#   w ← w - η * ∇TrainLoss(w)
# ─────────────────────────────────────────────
w = np.zeros(d)

print("=" * 70)
print("Gradient Descent — Linear Classification (Hinge Loss)")
print("=" * 70)
print(f"True weights   : {TRUE_WEIGHTS}")
print(f"Learning rate  : {ETA}    Iterations: {NUM_ITERATIONS}")
print(f"n = {NUM_SAMPLES}  |  d = {d}  |  φ(x) = x  (identity feature map)")
print("=" * 70)
print(f"{'Iter':>6}  {'HingeLoss':>10}  {'0-1 Loss':>10}  {'Accuracy':>10}  {'||grad||':>10}  {'weights'}")
print("-" * 70)

for t in range(1, NUM_ITERATIONS + 1):
    grad = train_gradient(w, X, y)
    w    = w - ETA * grad

    if t % PRINT_EVERY == 0 or t == 1:
        h_loss  = train_loss(w, X, y)
        zo_loss = zero_one_loss(w, X, y)
        acc     = accuracy(w, X, y)
        g_norm  = np.linalg.norm(grad)
        w_str   = "[" + ", ".join(f"{v:+.4f}" for v in w) + "]"
        print(f"{t:>6}  {h_loss:>10.6f}  {zo_loss:>10.4f}  {acc:>10.4f}  {g_norm:>10.6f}  {w_str}")

print("=" * 70)
print(f"Final weights  : {w.round(4)}")
print(f"True weights   : {TRUE_WEIGHTS}")
print(f"Final hinge loss : {train_loss(w, X, y):.6f}")
print(f"Final 0-1 loss   : {zero_one_loss(w, X, y):.4f}")
print(f"Final accuracy   : {accuracy(w, X, y):.4f}")
print()
print("Note: hinge loss = 0 implies zero-one loss = 0 (all examples correctly")
print("classified with margin >= 1). The algorithm is generic — only the loss")
print("and gradient changed from regression; gradient descent stayed the same.")
