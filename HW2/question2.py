import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

# Plot style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TwoLayerNN:
    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1, learning_rate=0.01, activation='tanh'):
        # Initialize network dimensions and hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.activation = activation

        # Xavier initialization for weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        self.training_errors = []

    # Activation functions and their derivatives
    def sigmoid(self, z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def sigmoid_derivative(self, a): return a * (1 - a)
    def tanh(self, z): return np.tanh(z)
    def tanh_derivative(self, a): return 1 - a ** 2
    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z): return (z > 0).astype(float)
    def logsig(self, z): return self.sigmoid(z)
    def logsig_derivative(self, a): return self.sigmoid_derivative(a)
    def tansig(self, z): return 2 / (1 + np.exp(-2 * np.clip(z, -500, 500))) - 1
    def tansig_derivative(self, a): return 1 - a ** 2
    def radialbasis(self, z, gamma=1.0): return np.exp(-gamma * np.square(z))
    def radialbasis_derivative(self, z, gamma=1.0): return -2 * gamma * z * np.exp(-gamma * np.square(z))

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.activation == 'sigmoid': self.a1 = self.sigmoid(self.z1)
        elif self.activation == 'tanh': self.a1 = self.tanh(self.z1)
        elif self.activation == 'relu': self.a1 = self.relu(self.z1)
        elif self.activation == 'logsig': self.a1 = self.logsig(self.z1)
        elif self.activation == 'tansig': self.a1 = self.tansig(self.z1)
        elif self.activation == 'radialbasis': self.a1 = self.radialbasis(self.z1)
        else: raise ValueError(f"Unknown activation: {self.activation}")
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear output for regression
        return self.a2

    def backward(self, X, Y):
        # Backward pass
        m = X.shape[0]
        self.dz2 = self.a2 - Y
        self.dW2 = np.dot(self.a1.T, self.dz2) / m
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True) / m
        da1 = np.dot(self.dz2, self.W2.T)

        if self.activation == 'sigmoid': self.dz1 = da1 * self.sigmoid_derivative(self.a1)
        elif self.activation == 'tanh': self.dz1 = da1 * self.tanh_derivative(self.a1)
        elif self.activation == 'relu': self.dz1 = da1 * self.relu_derivative(self.z1)
        elif self.activation == 'logsig': self.dz1 = da1 * self.logsig_derivative(self.a1)
        elif self.activation == 'tansig': self.dz1 = da1 * self.tansig_derivative(self.a1)
        elif self.activation == 'radialbasis': self.dz1 = da1 * self.radialbasis_derivative(self.z1)

        self.dW1 = np.dot(X.T, self.dz1) / m
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

    def calculate_loss(self, Y, predictions):
        return np.mean(np.square(Y - predictions))

    def train(self, X, Y, epochs, verbose=True):
        # Training loop
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = self.calculate_loss(Y, predictions)
            self.training_errors.append(loss)
            self.backward(X, Y)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


def plot_function_approximation(X_train, Y_train, models, activation_names, epochs_list):
    # Plot approximation results at different epochs
    X_dense = np.linspace(-1, 1, 200).reshape(-1, 1)
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.35)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B4513']

    # First 4 plots: show function approximation at specific epochs
    for idx, num_epochs in enumerate(epochs_list[:-1]):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(X_train, Y_train, color='black', s=50, alpha=0.7, edgecolors='white', linewidth=1.5, zorder=5)
        for act, name, color in zip(activation_names, activation_names, colors):
            model = TwoLayerNN(hidden_dim=15, learning_rate=0.1, activation=act)
            model.train(X_train, Y_train, num_epochs, verbose=False)
            ax.plot(X_dense, model.predict(X_dense), label=name.upper(), linewidth=2.2, alpha=0.85, color=color)
        ax.set_xlabel('x', fontsize=13, fontweight='bold')
        ax.set_ylabel('f(x)', fontsize=13, fontweight='bold')
        ax.set_title(f'Function Approximation after {num_epochs} epochs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(-1.05, 1.05)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True, labelspacing=1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Last plot: 1000 epochs with error visualization
    ax = fig.add_subplot(gs[2, :])
    ax.scatter(X_train, Y_train, color='black', s=60, alpha=0.75, edgecolors='white', linewidth=2, zorder=5)
    best_model = TwoLayerNN(hidden_dim=15, learning_rate=0.1, activation='tanh')
    best_model.train(X_train, Y_train, 1000, verbose=False)
    y_best = best_model.predict(X_dense)
    ax.plot(X_dense, y_best, label='NN Approximation (Tanh)', linewidth=3.5, alpha=0.95, color='#2E86AB')

    # Show error lines for training points
    y_pred_train = best_model.predict(X_train)
    for xi, yi, yp in zip(X_train.flat, Y_train.flat, y_pred_train.flat):
        ax.plot([xi, xi], [yi, yp], 'r--', alpha=0.35, linewidth=1)

    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=14, fontweight='bold')
    ax.set_title('Best Function Approximation (1000 epochs) with Error Visualization', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, shadow=True, labelspacing=1.2)
    ax.set_xlim(-1.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Neural Network Function Approximation Progress', fontsize=20, fontweight='bold', y=0.98)
    desktop = os.path.expanduser("~/Desktop/function_approximation.png")
    fig.savefig(desktop, dpi=300, bbox_inches='tight')
    print(f"Saved: {desktop}")
    plt.show()


def plot_activation_comparison(X_train, Y_train, activation_names):
    # Plot comparison of different activation functions
    X_dense = np.linspace(-1, 1, 200).reshape(-1, 1)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.ravel()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B4513']

    # First 5 subplots: each activation function with its MSE
    for idx, (act, color) in enumerate(zip(activation_names, colors)):
        if idx >= 5: break
        ax = axes[idx]
        model = TwoLayerNN(hidden_dim=15, learning_rate=0.1, activation=act)
        model.train(X_train, Y_train, 1000, verbose=False)
        y_pred = model.predict(X_dense)
        y_pred_train = model.predict(X_train)
        mse = np.mean((Y_train - y_pred_train) ** 2)
        ax.scatter(X_train, Y_train, color='black', s=45, alpha=0.65, edgecolors='white', linewidth=1.5, zorder=5)
        ax.plot(X_dense, y_pred, linewidth=2.8, alpha=0.9, color=color)
        ax.set_xlabel('x', fontsize=12, fontweight='bold')
        ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
        ax.set_title(f'{act.upper()} (MSE: {mse:.6f})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(-1.05, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Last subplot: convergence comparison
    ax = axes[5]
    for act, color in zip(activation_names, colors):
        model = TwoLayerNN(hidden_dim=15, learning_rate=0.1, activation=act)
        model.train(X_train, Y_train, 1000, verbose=False)
        ax.plot(model.training_errors, label=act.upper(), linewidth=2.3, alpha=0.85, color=color)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Comparison', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True, shadow=True, labelspacing=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Activation Functions Performance Comparison', fontsize=18, fontweight='bold', y=0.98)
    desktop = os.path.expanduser("~/Desktop/activation_comparison.png")
    fig.savefig(desktop, dpi=300, bbox_inches='tight')
    print(f"Saved: {desktop}")
    plt.show()


if __name__ == "__main__":
    # Training data
    X_train = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(-1, 1)
    Y_train = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134,
                        -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396,
                        0.345, 0.182, -0.031, -0.219, -0.321]).reshape(-1, 1)

    # Activation functions to evaluate
    activation_functions = ['tanh', 'relu', 'sigmoid', 'logsig', 'tansig', 'radialbasis']
    models = []

    for activation in activation_functions:
        print(f"\nTraining with {activation.upper()} activation...")
        model = TwoLayerNN(hidden_dim=15, learning_rate=0.1, activation=activation)
        model.train(X_train, Y_train, epochs=1000, verbose=False)
        models.append(model)
        Y_pred = model.predict(X_train)
        mse = np.mean((Y_train - Y_pred) ** 2)
        print(f"  Final MSE: {mse:.6f}")
        print(f"  Final Loss: {model.training_errors[-1]:.6f}")

    epochs_list = [10, 100, 200, 400, 1000]
    plot_function_approximation(X_train, Y_train, models, activation_functions, epochs_list)
    plot_activation_comparison(X_train, Y_train, activation_functions)
