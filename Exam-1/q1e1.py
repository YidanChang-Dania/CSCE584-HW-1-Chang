import numpy as np
import matplotlib.pyplot as plt
import time
import os

# CONFIG
SEED = 123                     # single unified seed for reproducibility
DEBUG_SHORT_RUN = True         # True -> quick debug run; False -> full paper run
MAX_STEPS = 5000 if DEBUG_SHORT_RUN else 50000
LEARNING_RATE = 0.25
SAVE_FIG = True                # whether to save final figure to disk
OUT_FIG_PATH = "example1_repro.png"


# determinism / seed
np.random.seed(SEED)

# Neural Network
class NeuralNetwork:
    def __init__(self, layers, seed=None):
        """
        layers: list [in, hidden1, hidden2, ..., out]
        Weight shape: (in_dim, out_dim)
        Use unified random seed provided by outer scope (we already set np.random.seed(SEED))
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            in_dim = layers[i]
            out_dim = layers[i+1]
            # Xavier-like initialization (symmetric)
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            w = np.random.randn(in_dim, out_dim) * scale
            b = np.zeros((1, out_dim), dtype=np.float64)
            self.weights.append(w.astype(np.float64))
            self.biases.append(b)

    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative_from_activation(self, a):
        return a * (1.0 - a)

    def forward(self, x):
        """
        x: (batch, input_dim)
        returns: (batch, out_dim)
        stores activations and z_values
        """
        self.activations = [x.astype(np.float64)]
        self.z_values = []
        a = x.astype(np.float64)
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                # linear output layer
                a = z.copy()
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, x, y_true, y_pred, learning_rate=0.25):
        """
        Single-step / per-batch gradient descent (no external optimizer).
        x: (batch, in_dim)
        y_true: (batch, out_dim)
        y_pred: (batch, out_dim)
        """
        m = x.shape[0]
        L = len(self.weights)
        deltas = [None] * L

        # output layer (linear): delta = (a_L - y)
        deltas[-1] = (y_pred - y_true)

        # backprop through hidden layers
        for l in range(L-2, -1, -1):
            next_delta = deltas[l+1]            # (batch, out_{l+1})
            W_next = self.weights[l+1]          # (in_{l+1}, out_{l+1})
            # propagate error
            err = np.dot(next_delta, W_next.T)  # (batch, out_l)
            a_l = self.activations[l+1]         # activation of layer l (post-nonlinearity)
            deltas[l] = err * self.sigmoid_derivative_from_activation(a_l)

        # update weights & biases
        for i in range(L):
            a_prev = self.activations[i]       # (batch, in_dim)
            delta = deltas[i]                  # (batch, out_dim)
            grad_W = np.dot(a_prev.T, delta) / m
            grad_b = np.mean(delta, axis=0, keepdims=True)
            # optional gradient clipping (small safeguard) - keep harmless but active
            # max_norm = 10.0
            # norm = np.linalg.norm(grad_W)
            # if norm > max_norm:
            #     grad_W = grad_W * (max_norm / norm)
            self.weights[i] -= learning_rate * grad_W
            self.biases[i]  -= learning_rate * grad_b


# Plant definition
def plant_function(u):
    # f(u) = 0.6 sin(pi u) + 0.3 sin(3 pi u) + 0.1 sin(5 pi u)
    return 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u)

def simulate_plant(y_k, y_k_minus_1, u_k):
    # y_p(k+1) = 0.3 y_p(k) + 0.6 y_p(k-1) + f(u(k))
    return 0.3 * y_k + 0.6 * y_k_minus_1 + plant_function(u_k)


# Training routine
def train_neural_network(max_steps=50000, learning_rate=0.25, verbose=True):
    # instantiate network
    nn = NeuralNetwork([1, 20, 10, 1])

    # compute final index we'll touch: Phase1 uses up to k=501; Phase2 uses k=502..(502+max_steps-1)
    final_k = 502 + max_steps - 1
    buffer = 5
    total_len = final_k + 1 + buffer

    y_p = np.zeros(total_len, dtype=np.float64)
    y_model = np.zeros_like(y_p)
    training_error = []

    # Phase 1: sinusoidal input (k=2..501)
    if verbose: print("Phase 1: Training with sinusoidal input (steps 2-501)...")
    for k in range(2, 502):
        u_k = np.sin(2 * np.pi * k / 250.0)
        y_p[k] = simulate_plant(y_p[k-1], y_p[k-2], u_k)

        nn_output = nn.forward(np.array([[u_k]]))           # shape (1,1)
        y_model[k] = 0.3 * y_p[k-1] + 0.6 * y_p[k-2] + nn_output[0, 0]

        target = np.array([[y_p[k] - (0.3 * y_p[k-1] + 0.6 * y_p[k-2])]])
        err = y_p[k] - y_model[k]
        training_error.append(err**2)
        nn.backward(np.array([[u_k]]), target, nn_output, learning_rate=learning_rate)

    # store phase1 snapshots
    y_p_phase1 = y_p.copy()
    y_model_phase1 = y_model.copy()
    if verbose:
        mse_p1 = np.mean(training_error[-100:]) if len(training_error) >= 100 else np.mean(training_error)
        print(f"Phase 1 complete. Final MSE (last100): {mse_p1:.6e}")

    # Phase 2: random input training
    if verbose: print(f"Phase 2: Training with random input (steps 502..{502+max_steps-1})...")
    np.random.seed(SEED)   # re-seed to keep Phase2 inputs deterministic across runs
    start_time = time.time()
    for step_idx, k in enumerate(range(502, 502 + max_steps)):
        # random input in [-1, 1] (paper setting)
        u_k = np.random.uniform(-1, 1)
        y_p[k] = simulate_plant(y_p[k-1], y_p[k-2], u_k)

        nn_output = nn.forward(np.array([[u_k]]))
        y_model[k] = 0.3 * y_p[k-1] + 0.6 * y_p[k-2] + nn_output[0, 0]

        target = np.array([[y_p[k] - (0.3 * y_p[k-1] + 0.6 * y_p[k-2])]])
        err = y_p[k] - y_model[k]
        training_error.append(err**2)
        nn.backward(np.array([[u_k]]), target, nn_output, learning_rate=learning_rate)

        # logging
        if verbose and ((step_idx + 1) % 5000 == 0 or step_idx == 0):
            recent1000 = np.mean(training_error[-1000:]) if len(training_error) >= 1000 else np.mean(training_error)
            recent100 = np.mean(training_error[-100:]) if len(training_error) >= 100 else np.mean(training_error)
            elapsed = time.time() - start_time
            print(f"  Step {step_idx+1}/{max_steps}, global k={k}, MSE(last1000)={recent1000:.6e}, MSE(last100)={recent100:.6e}, elapsed={elapsed:.1f}s")

    # final train stats
    if verbose:
        final_mse_100 = np.mean(training_error[-100:]) if len(training_error) >= 100 else np.mean(training_error)
        final_mse_1000 = np.mean(training_error[-1000:]) if len(training_error) >= 1000 else np.mean(training_error)
        print(f"Phase 2 complete. Final MSE (last100)={final_mse_100:.6e}, (last1000)={final_mse_1000:.6e}")

    # Testing with sum-of-sinusoids
    if verbose: print("Testing trained network with sum-of-sinusoids (1000 steps)...")
    test_len = 1000
    y_p_test = np.zeros(test_len, dtype=np.float64)
    y_model_test = np.zeros(test_len, dtype=np.float64)

    final_k_train = 502 + max_steps - 1
    y_p_test[0] = y_p[final_k_train]
    y_p_test[1] = y_p[final_k_train - 1]
    y_model_test[0] = y_model[final_k_train]
    y_model_test[1] = y_model[final_k_train - 1]

    for k in range(2, test_len):
        if k < 250:
            u_k = np.sin(2 * np.pi * k / 250.0)
        else:
            u_k = np.sin(2 * np.pi * k / 250.0) + np.sin(2 * np.pi * k / 25.0)
        y_p_test[k] = simulate_plant(y_p_test[k-1], y_p_test[k-2], u_k)
        nn_output = nn.forward(np.array([[u_k]]))
        y_model_test[k] = 0.3 * y_p_test[k-1] + 0.6 * y_p_test[k-2] + nn_output[0, 0]

    test_mse = np.mean((y_p_test - y_model_test)**2)
    test_mae = np.mean(np.abs(y_p_test - y_model_test))
    test_maxabs = np.max(np.abs(y_p_test - y_model_test))

    if verbose:
        print(f"Test MSE: {test_mse:.6e}, MAE: {test_mae:.6e}, MaxAbs: {test_maxabs:.6e}")

    return y_p_phase1, y_model_phase1, y_p_test, y_model_test, training_error


# Plotting & Main
def make_plots(y_p_phase1, y_model_phase1, y_p_test, y_model_test, training_error):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) outputs when adaptation stops at k=500: plot 300..700
    ax1 = axes[0, 0]
    k1 = np.arange(300, 700)
    ax1.plot(k1, y_p_phase1[300:700], label='Plant output', linewidth=2)
    ax1.plot(k1, y_model_phase1[300:700], '--', label='Model output', linewidth=2)
    ax1.set_xlabel('k')
    ax1.set_ylabel('y_p and ŷ_p')
    ax1.set_title('(a) Outputs when adaptation stops at k = 500')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) response after identification using random input: test 0..499
    ax2 = axes[0, 1]
    k2 = np.arange(0, 500)
    ax2.plot(k2, y_p_test[:500], label='Plant output', linewidth=2)
    ax2.plot(k2, y_model_test[:500], '--', label='Model output', linewidth=2)
    ax2.set_xlabel('k')
    ax2.set_ylabel('y_p and ŷ_p')
    ax2.set_title('(b) Response after identification using random input')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (c) training error over time (log scale). show first N points or all if smaller
    ax3 = axes[1, 0]
    to_plot = training_error[:10000] if len(training_error) >= 10000 else training_error
    ax3.semilogy(to_plot, linewidth=0.8)
    ax3.set_xlabel('Training step')
    ax3.set_ylabel('Squared error (log scale)')
    ax3.set_title('Training error evolution (first 10000 steps)')
    ax3.grid(True, alpha=0.3)

    # (d) absolute error on test input first 500 steps
    ax4 = axes[1, 1]
    test_error = np.abs(y_p_test - y_model_test)
    ax4.plot(test_error[:500], linewidth=0.9)
    ax4.set_xlabel('k')
    ax4.set_ylabel('|y_p - ŷ_p|')
    ax4.set_title('Absolute error on test set')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # run training
    y_p_phase1, y_model_phase1, y_p_test, y_model_test, training_error = train_neural_network(
        max_steps=MAX_STEPS, learning_rate=LEARNING_RATE, verbose=True
    )

    # plotting
    fig = make_plots(y_p_phase1, y_model_phase1, y_p_test, y_model_test, training_error)
    plt.show()

    if SAVE_FIG:
        fig.savefig(OUT_FIG_PATH, dpi=200)
        print(f"Figure saved to {os.path.abspath(OUT_FIG_PATH)}")

    # print summarized stats
    final_mse_100 = np.mean(training_error[-100:]) if len(training_error) >= 100 else np.mean(training_error)
    final_mse_1000 = np.mean(training_error[-1000:]) if len(training_error) >= 1000 else np.mean(training_error)
    test_mse = np.mean((y_p_test - y_model_test)**2)
    test_mae = np.mean(np.abs(y_p_test - y_model_test))
    test_maxabs = np.max(np.abs(y_p_test - y_model_test))

    print("SUMMARY")
    print(f"SEED = {SEED}, MAX_STEPS = {MAX_STEPS}, LEARNING_RATE = {LEARNING_RATE}")
    print(f"Final training MSE (last100): {final_mse_100:.6e}, (last1000): {final_mse_1000:.6e}")
    print(f"Test MSE: {test_mse:.6e}, MAE: {test_mae:.6e}, MaxAbs: {test_maxabs:.6e}")
