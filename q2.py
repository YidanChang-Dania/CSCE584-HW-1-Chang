"""
q2 (a)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Define activation functions
# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hard_limit(z):
    return np.where(z >= 0, 1, 0)

def rbf(z):
    return np.exp(-z**2)

# Activation function dictionary
activations = {
    "Sigmoid": sigmoid,
    "Hard Limit": hard_limit,
    "RBF": rbf
}

# ---------------------------
# 2. Perceptron model parameters
# ---------------------------
def perceptron(x1, x2, activation):
    z = -4.79 * x1 + 5.90 * x2 - 0.93
    return activation(z)

# ---------------------------
# 3. Plot output surface
# ---------------------------
def plot_surface(n_points, activation_name, activation_func):
    # Generate input points
    x1 = np.linspace(-2, 2, int(np.sqrt(n_points)))
    x2 = np.linspace(-2, 2, int(np.sqrt(n_points)))
    X1, X2 = np.meshgrid(x1, x2)

    # Compute output
    Y = perceptron(X1, X2, activation_func)

    # Plot 3D surface
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap='viridis')

    ax.set_title(f"{activation_name} Activation, {n_points} points")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    plt.tight_layout()
    plt.show()

# ---------------------------
# 4. Main program: three activation functions × three sample sizes
# ---------------------------
sample_sizes = [100, 5000, 10000]

for act_name, act_func in activations.items():
    for n in sample_sizes:
        plot_surface(n, act_name, act_func)






"""
q2 (b)
"""

# 1. Activation functions
# ---------------------------
def sigmoid(z):
    """Standard Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def hard_limit(z):
    """Hard limit activation function (step function)"""
    return np.where(z >= 0, 1, 0)

def rbf(z):
    """
    Radial Basis Function (RBF)
    Note: Here we use an element-wise definition exp(-z^2).
    This differs from some norm-based RBF definitions,
    but it matches the requirement of the assignment.
    """
    return np.exp(-z ** 2)

# Dictionary of activations
activations = {
    "Sigmoid": sigmoid,
    "Hard Limit": hard_limit,
    "RBF": rbf
}

# ---------------------------
# 2. Two-layer neural network parameters (strictly from assignment)
# ---------------------------
# Hidden layer weight matrix V^T (2x2)
V_T = np.array([[-2.69, -2.80],
                [-3.39, -4.56]])

# Hidden layer bias vector (2,)
b_v = np.array([-2.21, 4.76])

# Output layer weight vector (2,)
W = np.array([-4.91, 4.95])

# Output layer bias (scalar)
b_w = -2.28

# ---------------------------
# 3. Forward propagation
# ---------------------------
def two_layer_nn(x1, x2, activation):
    """
    Forward propagation of the two-layer neural network
    Args:
        x1, x2: input grids
        activation: activation function

    Returns:
        y: network output surface
    """
    # Input stacking (2 x N)
    inputs = np.vstack([x1.ravel(), x2.ravel()])

    # Hidden layer: V^T * x + b_v
    hidden_linear = np.dot(V_T, inputs) + b_v.reshape(-1, 1)

    # Apply activation
    hidden_output = activation(hidden_linear)

    # Output layer: W^T * hidden_output + b_w
    y = np.dot(W, hidden_output) + b_w

    return y.reshape(x1.shape)

# ---------------------------
# 4. Plotting function
# ---------------------------
def plot_surface(n_points, activation_name, activation_func):
    """
    Plot the output surface of the neural network

    Args:
        n_points: number of sample points
        activation_name: name of the activation function
        activation_func: activation function
    """
    # Ensure square grid, adjust if not a perfect square
    n = int(np.sqrt(n_points))
    actual_points = n * n
    if actual_points != n_points:
        print(f"⚠️ Note: {n_points} is not a perfect square, actual plotted points = {actual_points}")

    # Generate input grid
    x1 = np.linspace(-2, 2, n)
    x2 = np.linspace(-2, 2, n)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute output
    Y = two_layer_nn(X1, X2, activation_func)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X1, X2, Y, cmap='viridis', edgecolor='none', alpha=0.8)

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Output Value (y)', fontsize=12)

    # Labels and title
    ax.set_xlabel("x₁", fontsize=12, labelpad=10)
    ax.set_ylabel("x₂", fontsize=12, labelpad=10)
    ax.set_zlabel("y", fontsize=12, labelpad=10)

    # More descriptive title
    title = f"Two-Layer NN Output Surface\n({activation_name} Activation, {actual_points} Points)"
    ax.set_title(title, fontsize=14, pad=20)

    # View angle
    ax.view_init(elev=20, azim=45)

    # Grid lines
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"nn_final_{activation_name.lower().replace(' ', '_')}_{actual_points}.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    return Y

# ---------------------------
# 5. Verification function
# ---------------------------
def verify_calculation():
    """
    Verify calculations at several fixed points to ensure correctness
    """
    test_points = [
        (0, 0),
        (1, 1),
        (-1, -1),
        (2, 2),
        (-2, -2)
    ]

    print("Verification of calculations:")
    print("Point\t\tSigmoid\t\tHard Limit\tRBF")
    print("-" * 50)

    for x1, x2 in test_points:
        inputs = np.array([x1, x2])
        hidden_linear = np.dot(V_T, inputs) + b_v

        sigmoid_out = np.dot(W, sigmoid(hidden_linear)) + b_w
        hard_limit_out = np.dot(W, hard_limit(hidden_linear)) + b_w
        rbf_out = np.dot(W, rbf(hidden_linear)) + b_w

        print(f"({x1}, {x2})\t{sigmoid_out:.4f}\t\t{hard_limit_out:.4f}\t\t{rbf_out:.4f}")

# ---------------------------
# 6. Main program
# ---------------------------
def main():
    # Verify some test points
    verify_calculation()

    # Sample sizes
    sample_sizes = [100, 5000, 10000]

    # Store results
    all_results = {}

    # Generate plots
    for act_name, act_func in activations.items():
        print(f"\nProcessing {act_name} activation...")
        act_results = {}
        for n in sample_sizes:
            print(f"  Generating plot with {n} points...")
            Y = plot_surface(n, act_name, act_func)
            act_results[n] = Y
        all_results[act_name] = act_results

    return all_results


if __name__ == "__main__":
    results = main()
