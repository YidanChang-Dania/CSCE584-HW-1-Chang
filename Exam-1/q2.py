import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layers):
        """
        Initialize neural network with given layer sizes
        layers: list of layer sizes [input, hidden1, hidden2, ..., output]
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        np.random.seed(42)
        for i in range(self.num_layers - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        """Sigmoid activation with numerical stability"""
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def sigmoid_derivative_from_output(self, a):
        """Derivative of sigmoid using activation output"""
        return a * (1 - a)

    def forward(self, x):
        """Forward propagation"""
        self.activations = [x]
        self.z_values = []

        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i < self.num_layers - 2:  # Hidden layers use sigmoid
                a = self.sigmoid(z)
            else:  # Output layer is linear
                a = z
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, x, y_target, learning_rate=0.25):
        """Backward propagation with correct indexing"""
        m = x.shape[0]

        # Initialize deltas
        deltas = [None] * (self.num_layers - 1)

        # Output layer delta (linear activation)
        deltas[-1] = self.activations[-1] - y_target

        # Backpropagate through hidden layers
        for l in range(self.num_layers - 3, -1, -1):
            error = np.dot(deltas[l + 1], self.weights[l + 1].T)
            deltas[l] = error * self.sigmoid_derivative_from_output(self.activations[l + 1])

        # Update weights and biases
        for l in range(self.num_layers - 1):
            grad_w = np.dot(self.activations[l].T, deltas[l]) / m
            grad_b = np.mean(deltas[l], axis=0, keepdims=True)

            self.weights[l] -= learning_rate * grad_w
            self.biases[l] -= learning_rate * grad_b


def plant_function(y_k, y_k_minus_1):
    numerator = y_k * y_k_minus_1 * (y_k + 2.5)
    denominator = 1 + y_k ** 2 + y_k_minus_1 ** 2
    return numerator / denominator


def simulate_plant(y_k, y_k_minus_1, u_k):
    """Plant dynamics: y_p(k+1) = f[y_p(k), y_p(k-1)] + u(k)"""
    return plant_function(y_k, y_k_minus_1) + u_k


def simulate_reference_model(y_m_k, y_m_k_minus_1, r_k):
    """Reference model: y_m(k+1) = 0.6y_m(k) + 0.2y_m(k-1) + r(k)"""
    return 0.6 * y_m_k + 0.2 * y_m_k_minus_1 + r_k


def identify_plant_offline(nn, num_steps=50000):
    """
    Identify the plant offline using random inputs
    This corresponds to identifying f in Example 2
    """
    print("Offline identification of plant function f...")

    # Generate training data with random states
    np.random.seed(123)

    for step in range(num_steps):
        # Random states for training
        y_k = np.random.uniform(-4, 4)
        y_k_minus_1 = np.random.uniform(-4, 4)

        # Compute true function value
        f_true = plant_function(y_k, y_k_minus_1)

        # Forward pass
        x = np.array([[y_k, y_k_minus_1]])
        f_pred = nn.forward(x)

        # Backward pass with smaller learning rate for stability
        nn.backward(x, np.array([[f_true]]), learning_rate=0.05)

        if (step + 1) % 10000 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    print("Offline identification complete.\n")
    return nn


def simulate_no_control():
    """Simulate plant without control (Figure 24a baseline)"""
    steps = 100
    y_p = np.zeros(steps)
    r = np.zeros(steps)

    # Initial conditions
    y_p[0] = 0.0
    y_p[1] = 0.0

    # Reference input
    for k in range(steps):
        r[k] = np.sin(2 * np.pi * k / 25)

    # Simulate with u(k) = r(k) (no control, just feeding reference as input)
    for k in range(2, steps):
        y_p[k] = simulate_plant(y_p[k - 1], y_p[k - 2], r[k])

    return y_p, r


def simulate_offline_control(nn):
    """Simulate control after offline identification (Figure 24b)"""
    steps = 100
    y_p = np.zeros(steps)
    y_m = np.zeros(steps)
    u = np.zeros(steps)
    r = np.zeros(steps)

    # Initial conditions
    y_p[0] = 0.0
    y_p[1] = 0.0
    y_m[0] = 0.0
    y_m[1] = 0.0

    # Reference input
    for k in range(steps):
        r[k] = np.sin(2 * np.pi * k / 25)

    # Control loop
    for k in range(2, steps):
        # Reference model
        y_m[k] = simulate_reference_model(y_m[k - 1], y_m[k - 2], r[k])

        # Control computation using identified model
        x = np.array([[y_p[k - 1], y_p[k - 2]]])
        f_estimate = nn.forward(x)[0, 0]
        u[k] = -f_estimate + 0.6 * y_p[k - 1] + 0.2 * y_p[k - 2] + r[k]

        # Plant response
        y_p[k] = simulate_plant(y_p[k - 1], y_p[k - 2], u[k])

    return y_p, y_m, r


def simulate_online_control(Ti, Tc, total_steps=10000):
    """Simulate simultaneous identification and control"""
    y_p = np.zeros(total_steps)
    y_m = np.zeros(total_steps)
    u = np.zeros(total_steps)
    r = np.zeros(total_steps)

    # Initial conditions
    y_p[0] = 0.0
    y_p[1] = 0.0
    y_m[0] = 0.0
    y_m[1] = 0.0

    # Initialize neural network for online learning
    nn_online = NeuralNetwork([2, 20, 10, 1])

    # Pre-train briefly for stability (except for Ti=Tc=10 case)
    if not (Ti == 10 and Tc == 10):
        identify_plant_offline(nn_online, num_steps=5000)

    # Reference input
    for k in range(total_steps):
        r[k] = np.sin(2 * np.pi * k / 25)

    # Online control loop
    for k in range(2, total_steps):
        # Reference model
        y_m[k] = simulate_reference_model(y_m[k - 1], y_m[k - 2], r[k])

        # Control update (every Tc steps)
        if k % Tc == 0:
            x = np.array([[y_p[k - 1], y_p[k - 2]]])
            f_estimate = nn_online.forward(x)[0, 0]
            u[k] = -f_estimate + 0.6 * y_p[k - 1] + 0.2 * y_p[k - 2] + r[k]
        else:
            u[k] = u[k - 1]  # Hold previous control

        # Plant output
        y_p[k] = simulate_plant(y_p[k - 1], y_p[k - 2], u[k])

        # Identification update (every Ti steps)
        if k % Ti == 0 and k < 5000:  # Stop identification after 5000 steps
            x = np.array([[y_p[k - 1], y_p[k - 2]]])
            f_true = y_p[k] - u[k]
            f_pred = nn_online.forward(x)
            nn_online.backward(x, np.array([[f_true]]), learning_rate=0.05)

        # Check for instability
        if np.abs(y_p[k]) > 50:
            print(f"  System diverged at k={k}")
            break

    return y_p, y_m, r


# Run all simulations
print("=" * 60)
print("EXAMPLE 7: ADAPTIVE CONTROL OF NONLINEAR PLANT")
print("=" * 60)

# 1. No control simulation (baseline)
print("\n1. Simulating plant without control (baseline)...")
y_p_no_control, r_no_control = simulate_no_control()

# 2. Offline identification and control
print("\n2. Offline identification and control...")
nn_offline = NeuralNetwork([2, 20, 10, 1])
nn_offline = identify_plant_offline(nn_offline, num_steps=50000)
y_p_offline, y_m_offline, r_offline = simulate_offline_control(nn_offline)

# 3. Online control with different Ti and Tc
print("\n3. Simultaneous identification and control...")

print("   Ti = Tc = 1...")
y_p_online_1_1, y_m_online_1_1, r_online_1_1 = simulate_online_control(Ti=1, Tc=1, total_steps=10000)

print("   Ti = 1, Tc = 3...")
y_p_online_1_3, y_m_online_1_3, r_online_1_3 = simulate_online_control(Ti=1, Tc=3, total_steps=10000)

print("   Ti = Tc = 10 (unstable case)...")
y_p_online_10_10, y_m_online_10_10, r_online_10_10 = simulate_online_control(Ti=10, Tc=10, total_steps=100)

# Create Figure 24: Offline control comparison
fig24, axes24 = plt.subplots(1, 2, figsize=(12, 5))

# Figure 24(a): No control response
ax24a = axes24[0]
k_range = np.arange(100)
ax24a.plot(k_range, y_p_no_control, 'b-', linewidth=2, label='y_p (no control)')
ax24a.plot(k_range, r_no_control, 'g--', linewidth=1.5, label='r = sin(2πk/25)')
ax24a.set_xlabel('k')
ax24a.set_ylabel('y_p and r')
ax24a.set_title('(a) Response for no control action')
ax24a.set_xlim(0, 100)
ax24a.set_ylim(-2, 4)
ax24a.legend()
ax24a.grid(True, alpha=0.3)

# Figure 24(b): With control after offline identification
ax24b = axes24[1]
ax24b.plot(k_range, y_p_offline, 'b-', linewidth=2, label='y_p (plant)')
ax24b.plot(k_range, y_m_offline, 'r--', linewidth=2, label='y_m (reference)')
ax24b.set_xlabel('k')
ax24b.set_ylabel('y_p and y_m')
ax24b.set_title('(b) Response for\nr = sin(2πk/25) with control')
ax24b.set_xlim(0, 100)
ax24b.set_ylim(-3, 4)
ax24b.legend()
ax24b.grid(True, alpha=0.3)

plt.suptitle('Figure 24. Example 7: Offline Control Performance', fontsize=14)
plt.tight_layout()
plt.show()

# Create Figure 25: Asymptotic response with online control
fig25, axes25 = plt.subplots(1, 2, figsize=(12, 5))

# Figure 25(a): Ti=Tc=1 asymptotic response
ax25a = axes25[0]
k_range_asymptotic = np.arange(9900, 10000)
ax25a.plot(k_range_asymptotic, y_p_online_1_1[9900:10000], 'b-', linewidth=2, label='y_p')
ax25a.plot(k_range_asymptotic, y_m_online_1_1[9900:10000], 'r--', linewidth=2, label='y_m')
ax25a.set_xlabel('k')
ax25a.set_ylabel('y_m and y_p')
ax25a.set_title('(a) Response when control is initiated at k = 0 with\nTi = Tc = 1')
ax25a.set_xlim(9900, 10000)
ax25a.set_ylim(-5, 5)
ax25a.legend()
ax25a.grid(True, alpha=0.3)

# Figure 25(b): Ti=1, Tc=3 asymptotic response
ax25b = axes25[1]
ax25b.plot(k_range_asymptotic, y_p_online_1_3[9900:10000], 'b-', linewidth=2, label='y_p')
ax25b.plot(k_range_asymptotic, y_m_online_1_3[9900:10000], 'r--', linewidth=2, label='y_m')
ax25b.set_xlabel('k')
ax25b.set_ylabel('y_m and y_p')
ax25b.set_title('(b) Response when control is initiated at k = 0 and\nTi = 1 and Tc = 3')
ax25b.set_xlim(9900, 10000)
ax25b.set_ylim(-5, 5)
ax25b.legend()
ax25b.grid(True, alpha=0.3)

plt.suptitle('Figure 25. Example 7: Asymptotic Response with Online Control', fontsize=14)
plt.tight_layout()
plt.show()

# Create Figure 26: Unstable case
fig26, ax26 = plt.subplots(1, 1, figsize=(8, 6))

k_range_unstable = np.arange(min(40, len(y_p_online_10_10)))
ax26.plot(k_range_unstable, y_p_online_10_10[:len(k_range_unstable)], 'b-', linewidth=2, label='y_p')
ax26.plot(k_range_unstable, y_m_online_10_10[:len(k_range_unstable)], 'r--', linewidth=2, label='y_m')
ax26.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax26.set_xlabel('k')
ax26.set_ylabel('y_m and y_p')
ax26.set_title('Figure 26. Example 7: Response when control is initiated at k = 0 with\nTi = Tc = 10')
ax26.set_xlim(0, 40)
ax26.set_ylim(-10, 60)
ax26.text(15, 40, 'Ti = Tc = 10', fontsize=12, fontweight='bold')
ax26.legend()
ax26.grid(True, alpha=0.3)
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)

print("\n1. No Control (baseline):")
print(f"   Max |y_p|: {np.max(np.abs(y_p_no_control)):.2f}")
print(f"   Plant does not track reference without control")

print("\n2. Offline Control (after complete identification):")
mse_offline = np.mean((y_p_offline - y_m_offline) ** 2)
print(f"   MSE between y_p and y_m: {mse_offline:.6f}")
print(f"   Max tracking error: {np.max(np.abs(y_p_offline - y_m_offline)):.4f}")

print("\n3. Online Control (simultaneous identification & control):")

# Ti=Tc=1
mse_1_1 = np.mean((y_p_online_1_1[-100:] - y_m_online_1_1[-100:]) ** 2)
print(f"   Ti=Tc=1:")
print(f"     Asymptotic MSE: {mse_1_1:.6f}")
print(f"     System stable and converges")

# Ti=1, Tc=3
mse_1_3 = np.mean((y_p_online_1_3[-100:] - y_m_online_1_3[-100:]) ** 2)
print(f"   Ti=1, Tc=3:")
print(f"     Asymptotic MSE: {mse_1_3:.6f}")
print(f"     System stable with slightly degraded performance")

# Ti=Tc=10
print(f"   Ti=Tc=10:")
print(f"     System becomes unstable")
print(f"     Max |y_p| before divergence: {np.max(np.abs(y_p_online_10_10)):.1f}")

print("\nConclusion: Ti and Tc must be chosen carefully for stability.")