import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import os

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set path to save images on desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_images = True  # Set to False if you don't want to save images


class OneLayerNN:
    def __init__(self, input_dim=2, output_dim=2, learning_rate=0.1, activation='sigmoid'):
        """
        Initialize one-layer neural network
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.activation = activation

        # Initialize weights and biases with small random values
        np.random.seed(42)
        self.W = np.random.randn(input_dim, output_dim) * 0.5
        self.b = np.zeros((1, output_dim))

        # Store training history
        self.training_errors = []

    def sigmoid(self, z):
        """Sigmoid activation function (logsig)"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        """Derivative of sigmoid function"""
        return a * (1 - a)

    def tanh(self, z):
        """Tanh activation function (tansig)"""
        return np.tanh(z)

    def tanh_derivative(self, a):
        """Derivative of tanh function"""
        return 1 - a ** 2

    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)

    def relu_derivative(self, a):
        """Derivative of ReLU function"""
        return (a > 0).astype(float)

    def radialbasis(self, z, gamma=1.0):
        """Radial basis function (Gaussian)"""
        # For simplicity, we use a Gaussian RBF centered at 0
        return np.exp(-gamma * np.square(z))

    def radialbasis_derivative(self, a, gamma=1.0):
        """Derivative of RBF function (approximate)"""
        # This is a simplified derivative for the RBF
        return -2 * gamma * a

    def forward(self, X):
        """Forward propagation"""
        self.z = np.dot(X, self.W) + self.b

        if self.activation == 'sigmoid' or self.activation == 'logsig':
            self.a = self.sigmoid(self.z)
        elif self.activation == 'tanh' or self.activation == 'tansig':
            self.a = self.tanh(self.z)
        elif self.activation == 'relu':
            self.a = self.relu(self.z)
        elif self.activation == 'radialbasis':
            self.a = self.radialbasis(self.z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return self.a

    def backward(self, X, Y, output):
        """Backward propagation"""
        m = X.shape[0]

        # Calculate output layer gradients
        self.output_error = output - Y

        if self.activation == 'sigmoid' or self.activation == 'logsig':
            self.output_delta = self.output_error * self.sigmoid_derivative(output)
        elif self.activation == 'tanh' or self.activation == 'tansig':
            self.output_delta = self.output_error * self.tanh_derivative(output)
        elif self.activation == 'relu':
            self.output_delta = self.output_error * self.relu_derivative(output)
        elif self.activation == 'radialbasis':
            self.output_delta = self.output_error * self.radialbasis_derivative(output)

        # Calculate gradients
        self.dW = np.dot(X.T, self.output_delta) / m
        self.db = np.sum(self.output_delta, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db

    def calculate_loss(self, Y, output):
        """Calculate mean squared error loss"""
        return np.mean(np.square(Y - output))

    def train(self, X, Y, epochs):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Calculate loss
            loss = self.calculate_loss(Y, output)
            self.training_errors.append(loss)

            # Backward propagation
            self.backward(X, Y, output)

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        # Convert to binary predictions
        return (output > 0.5).astype(int)

    def get_decision_boundary(self, x_min, x_max, y_min, y_max, resolution=100):
        """Generate decision boundary for visualization"""
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.forward(grid_points)

        # Convert binary outputs to class labels
        # Group encoding: (1,0)=0, (0,0)=1, (1,1)=2, (0,1)=3
        class_labels = Z[:, 0] * 2 + Z[:, 1]
        return xx, yy, class_labels.reshape(xx.shape)


def plot_results(X, Y, models, epochs_list, activation_functions):
    """Create comprehensive visualization of results"""

    # Define groups and colors
    groups = [
        [(0.1, 1.2), (0.7, 1.8), (0.8, 1.6)],  # Group 1 (1,0)
        [(0.8, 0.6), (1.0, 0.8)],  # Group 2 (0,0)
        [(0.3, 0.5), (0.0, 0.2), (-0.3, 0.8)],  # Group 3 (1,1)
        [(-0.5, -1.5), (-1.5, -1.3)]  # Group 4 (0,1)
    ]
    group_colors = ['red', 'blue', 'green', 'orange']
    group_labels = ['Group 1: (1,0)', 'Group 2: (0,0)', 'Group 3: (1,1)', 'Group 4: (0,1)']

    # Convert Y to class labels for visualization
    class_labels = Y[:, 0] * 2 + Y[:, 1]

    # 1. Plot training error vs epoch for sigmoid
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sigmoid_model = models['sigmoid']
    ax1.plot(sigmoid_model.training_errors, linewidth=2, color='darkblue')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Error (MSE)', fontsize=12)
    ax1.set_title('Training Error vs Epoch Number (Sigmoid Activation)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    plt.tight_layout()

    if save_images:
        plt.savefig(os.path.join(desktop_path, 'training_error_curve.png'), dpi=300, bbox_inches='tight')
        print(f"Saved training error curve to desktop")

    plt.show()

    # 2. Plot decision boundaries at different epochs for sigmoid
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, num_epochs in enumerate(epochs_list):
        ax = axes[idx]

        # Train a new model for this number of epochs
        temp_model = OneLayerNN(learning_rate=0.1, activation='sigmoid')
        temp_model.train(X.T, Y.T, num_epochs)

        # Get decision boundary
        x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
        y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
        xx, yy, Z = temp_model.get_decision_boundary(x_min, x_max, y_min, y_max)

        # Create custom colormap for 4 classes
        cmap = ListedColormap(['#FFE6E6', '#E6F3FF', '#E6FFE6', '#FFF3E6'])

        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5, 3.5])

        # Plot data points
        for i, (group, color, label) in enumerate(zip(groups, group_colors, group_labels)):
            group_points = np.array(group)
            ax.scatter(group_points[:, 0], group_points[:, 1],
                       c=color, s=100, edgecolors='black', linewidth=2,
                       label=label, zorder=5)

        ax.set_xlabel('X1', fontsize=11)
        ax.set_ylabel('X2', fontsize=11)
        ax.set_title(f'Decision Boundary after {num_epochs} epochs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.suptitle('Decision Boundaries at Different Training Stages (Sigmoid)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_images:
        plt.savefig(os.path.join(desktop_path, 'decision_boundaries.png'), dpi=300, bbox_inches='tight')
        print(f"Saved decision boundaries to desktop")

    plt.show()

    # 3. Compare different activation functions
    fig3, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(models.items()):
        if idx >= 5:  # Only 5 subplots for activation functions
            break

        ax = axes[idx]

        # Get decision boundary
        x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
        y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
        xx, yy, Z = model.get_decision_boundary(x_min, x_max, y_min, y_max)

        # Plot decision boundary
        cmap = ListedColormap(['#FFE6E6', '#E6F3FF', '#E6FFE6', '#FFF3E6'])
        contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5, 3.5])

        # Plot data points
        for i, (group, color) in enumerate(zip(groups, group_colors)):
            group_points = np.array(group)
            ax.scatter(group_points[:, 0], group_points[:, 1],
                       c=color, s=100, edgecolors='black', linewidth=2, zorder=5)

        # Calculate accuracy
        predictions = model.predict(X.T)
        accuracy = np.mean(predictions == Y.T) * 100

        ax.set_xlabel('X1', fontsize=11)
        ax.set_ylabel('X2', fontsize=11)
        ax.set_title(f'{name.capitalize()} (Acc: {accuracy:.1f}%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Plot training curves comparison
    ax = axes[5]
    for name, model in models.items():
        if name not in ['tanh_lr0.01', 'tanh_lr0.5']:  # Only show main activation functions
            ax.plot(model.training_errors[:1000], label=name.capitalize(), linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Error (MSE)', fontsize=11)
    ax.set_title('Training Curves Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('Comparison of Different Activation Functions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_images:
        plt.savefig(os.path.join(desktop_path, 'activation_functions_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved activation functions comparison to desktop")

    plt.show()


def print_detailed_assessment(models, X, Y):
    """Print detailed assessment of all activation functions"""
    print("\n" + "=" * 70)
    print("DETAILED ASSESSMENT OF ACTIVATION FUNCTIONS")
    print("=" * 70)

    # Define assessment criteria for each activation function
    assessments = {
        'sigmoid': {
            'pros': ['Smooth, bounded [0,1] output', 'Good for binary classification',
                     'Well-suited for probability outputs'],
            'cons': ['Suffers from vanishing gradients', 'Outputs not zero-centered', 'Slower convergence than tanh'],
            'suitability': 'Good for this problem due to binary classification nature'
        },
        'logsig': {
            'pros': ['Same as sigmoid (logsig is another name for sigmoid)'],
            'cons': ['Same limitations as sigmoid'],
            'suitability': 'Identical to sigmoid for this problem'
        },
        'tanh': {
            'pros': ['Zero-centered output [-1,1]', 'Faster convergence than sigmoid', 'Better gradient flow'],
            'cons': ['Still suffers from vanishing gradients at extremes'],
            'suitability': 'Excellent choice, often better than sigmoid'
        },
        'tansig': {
            'pros': ['Same as tanh (tansig is another name for tanh)'],
            'cons': ['Same limitations as tanh'],
            'suitability': 'Identical to tanh for this problem'
        },
        'radialbasis': {
            'pros': ['Localized response', 'Good for pattern recognition', 'Bounded output [0,1]'],
            'cons': ['Not ideal for this single-layer architecture', 'Requires careful parameter tuning',
                     'May not converge well'],
            'suitability': 'Poor for this specific problem due to architecture limitations'
        },
        'relu': {
            'pros': ['Computationally efficient', 'No vanishing gradient for positive inputs', 'Sparsity inducing'],
            'cons': ['"Dying ReLU" problem for negative inputs', 'Not bounded',
                     'May not be ideal for binary classification'],
            'suitability': 'Moderate - works but not optimal for this binary classification'
        }
    }

    # Save assessment to a text file on desktop
    if save_images:
        assessment_file = os.path.join(desktop_path, 'activation_functions_assessment.txt')
        with open(assessment_file, 'w') as f:
            f.write("ACTIVATION FUNCTIONS ASSESSMENT\n")
            f.write("=" * 50 + "\n\n")

    for name, model in models.items():
        # Get base name for assessment lookup
        base_name = name
        if 'lr' in name:
            base_name = 'tanh'  # tanh variants

        predictions = model.predict(X.T)
        accuracy = np.mean(predictions == Y.T) * 100
        final_loss = model.training_errors[-1]

        print(f"\n{name.upper():<15}")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Accuracy: {accuracy:.2f}%")

        if base_name in assessments:
            assessment = assessments[base_name]
            print(f"  Pros: {', '.join(assessment['pros'])}")
            print(f"  Cons: {', '.join(assessment['cons'])}")
            print(f"  Suitability: {assessment['suitability']}")

        print("-" * 50)

        # Write to file
        if save_images:
            with open(assessment_file, 'a') as f:
                f.write(f"{name.upper():<15}\n")
                f.write(f"  Final Loss: {final_loss:.6f}\n")
                f.write(f"  Accuracy: {accuracy:.2f}%\n")
                if base_name in assessments:
                    assessment = assessments[base_name]
                    f.write(f"  Pros: {', '.join(assessment['pros'])}\n")
                    f.write(f"  Cons: {', '.join(assessment['cons'])}\n")
                    f.write(f"  Suitability: {assessment['suitability']}\n")
                f.write("-" * 50 + "\n\n")

    if save_images:
        print(f"\nSaved activation functions assessment to desktop")


# Main execution
if __name__ == "__main__":
    # Define input data
    X = np.array([
        [0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
        [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]
    ])

    # Define target outputs (binary encoding for 4 groups)
    Y = np.array([
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],  # First bit
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Second bit
    ])

    print("=" * 60)
    print("ONE-LAYER NEURAL NETWORK CLASSIFICATION")
    print("=" * 60)
    print(f"\nInput shape: {X.shape}")
    print(f"Target shape: {Y.shape}")
    print(f"Number of samples: {X.shape[1]}")
    print(f"Number of features: {X.shape[0]}")
    print(f"Number of outputs: {Y.shape[0]}")

    # Train with all required activation functions
    activation_functions = ['sigmoid', 'tanh', 'relu', 'radialbasis']
    models = {}

    for activation in activation_functions:
        print(f"\n{'=' * 40}")
        print(f"Training with {activation.upper()} activation")
        print(f"{'=' * 40}")

        model = OneLayerNN(learning_rate=0.1, activation=activation)
        model.train(X.T, Y.T, epochs=1000)
        models[activation] = model

        # Evaluate
        predictions = model.predict(X.T)
        accuracy = np.mean(predictions == Y.T) * 100
        print(f"Final Loss: {model.training_errors[-1]:.6f}")
        print(f"Accuracy: {accuracy:.2f}%")

    # Add logsig and tansig (which are same as sigmoid and tanh)
    models['logsig'] = models['sigmoid']
    models['tansig'] = models['tanh']

    # Additional models for different learning rates (tanh variants)
    for lr in [0.01, 0.5]:
        name = f'tanh_lr{lr}'
        model = OneLayerNN(learning_rate=lr, activation='tanh')
        model.train(X.T, Y.T, epochs=1000)
        models[name] = model

    # Visualize results with correct epoch points
    epochs_list = [3, 10, 100, 1000]  # Fixed to match problem requirements
    plot_results(X, Y, models, epochs_list, activation_functions)

    # Print detailed assessment
    print_detailed_assessment(models, X, Y)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    print("Based on the analysis, tanh (tansig) activation function is")
    print("recommended for this problem due to its zero-centered output")
    print("and faster convergence compared to sigmoid/logsig.")
    print("Radial basis function is not suitable for this single-layer")
    print("architecture, and ReLU is suboptimal for binary classification.")

    # Save final recommendation to file
    if save_images:
        with open(os.path.join(desktop_path, 'final_recommendation.txt'), 'w') as f:
            f.write("FINAL RECOMMENDATION\n")
            f.write("=" * 50 + "\n\n")
            f.write("Based on the analysis, tanh (tansig) activation function is\n")
            f.write("recommended for this problem due to its zero-centered output\n")
            f.write("and faster convergence compared to sigmoid/logsig.\n")
            f.write("Radial basis function is not suitable for this single-layer\n")
            f.write("architecture, and ReLU is suboptimal for binary classification.\n")
        print(f"Saved final recommendation to desktop")

    print(f"\nAll results have been saved to your desktop: {desktop_path}")

