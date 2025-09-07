"""
q1 (b)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. Load data
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

x_train = np.array(train_data.iloc[:, 1:]) / 255.0
y_train_raw = np.array(train_data.iloc[:, 0])
x_test = np.array(test_data.iloc[:, 1:]) / 255.0
y_test = np.array(test_data.iloc[:, 0])

# One-hot encode
encoded_labels = np.eye(10)[y_train_raw]


# 2. DNN implementation
class DNN:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.lr = learning_rate
        self.weights = []

        # Xavier initialization
        for i in range(len(layers) - 1):
            limit = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            w = np.random.randn(layers[i + 1], layers[i] + 1) * limit
            self.weights.append(w)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            a_prev = np.concatenate([activations[-1], [1]])  # add bias
            z = np.dot(self.weights[i], a_prev)
            if i == len(self.weights) - 1:
                activation = self.softmax(z)  # output layer
            else:
                activation = self.sigmoid(z)  # hidden layers
            activations.append(activation)
        return activations

    def train(self, x, y_true):
        # Forward
        activations = self.forward(x)

        # Backward
        deltas = [None] * len(self.weights)
        deltas[-1] = activations[-1] - y_true  # output error

        for i in range(len(self.weights) - 2, -1, -1):
            w_no_bias = self.weights[i + 1][:, :-1]
            error = np.dot(w_no_bias.T, deltas[i + 1])
            delta = error * self.sigmoid_derivative(activations[i + 1])
            deltas[i] = delta

        # Update weights
        for i in range(len(self.weights)):
            a_with_bias = np.concatenate([activations[i], [1]])
            grad = np.outer(deltas[i], a_with_bias)
            self.weights[i] -= self.lr * grad

        # Cross-entropy loss
        loss = -np.sum(y_true * np.log(activations[-1] + 1e-8))
        return loss

    def predict(self, x):
        return self.forward(x)[-1]

    def evaluate(self, x_data, y_data):
        correct = 0
        for i in range(len(x_data)):
            pred = np.argmax(self.predict(x_data[i]))
            if pred == y_data[i]:
                correct += 1
        return correct / len(y_data)


# 3. Train with mini-batch
model = DNN(layers=[784, 128, 10], learning_rate=0.01)

errors = []
accuracies = []
steps = 5000
batch_size = 64

print("Starting training...")
for n in range(steps):
    # mini-batch
    batch_indices = np.random.choice(len(x_train), batch_size, replace=False)
    batch_loss = 0
    for idx in batch_indices:
        batch_loss += model.train(x_train[idx], encoded_labels[idx])
    batch_loss /= batch_size
    errors.append(batch_loss)

    # evaluate every 100 steps
    if n % 100 == 0:
        test_acc = model.evaluate(x_test[:1000], y_test[:1000])
        accuracies.append(test_acc)
        print(f"Step {n}, Loss: {batch_loss:.4f}, Accuracy: {test_acc:.4f}")


# 4. Plot training results
plt.figure(figsize=(15, 6))

# Left plot: raw + smoothed loss
plt.subplot(1, 2, 1)
plt.plot(errors, linewidth=0.5, alpha=0.4, label="Raw Loss")

window_size = 50
smoothed_errors = [np.mean(errors[max(0, i - window_size):i + 1]) for i in range(len(errors))]
plt.plot(smoothed_errors, color='red', linewidth=1.5, label=f"Smoothed ({window_size}-step)")
plt.title('Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Right plot: Test accuracy with improved annotations
plt.subplot(1, 2, 2)
accuracy_steps = list(range(0, steps, 100))
plt.plot(accuracy_steps, accuracies, 'o-', linewidth=2, markersize=5, color='green', markerfacecolor='white')
plt.title('Test Accuracy Progress', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Improved annotations with two decimal places and staggered placement
for i, (step, acc) in enumerate(zip(accuracy_steps, accuracies)):
    if i % 3 == 0:  # Annotate every 3rd point to reduce clutter
        # Alternate vertical offset to prevent overlap
        vertical_offset = 10 if i % 2 == 0 else -15
        plt.annotate(f'{acc:.2f}',
                     xy=(step, acc),
                     xytext=(5, vertical_offset),
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color='gray'))

plt.tight_layout()
plt.savefig("training_results_stable.pdf", dpi=300, bbox_inches='tight')
plt.show()

# 5. Final evaluation
final_accuracy = model.evaluate(x_test, y_test)
print(f"\nFinal Test Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")




"""
q1 (d)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. Load data
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

x_train = np.array(train_data.iloc[:, 1:]) / 255.0
y_train_raw = np.array(train_data.iloc[:, 0])
x_test = np.array(test_data.iloc[:, 1:]) / 255.0
y_test = np.array(test_data.iloc[:, 0])

# One-hot encode
encoded_labels = np.eye(10)[y_train_raw]


# 2. Define DNN class (single hidden layer)
class DNN:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.lr = learning_rate
        self.weights = []
        for i in range(len(layers) - 1):
            limit = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            w = np.random.randn(layers[i + 1], layers[i] + 1) * limit
            self.weights.append(w)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            a_prev = np.concatenate([activations[-1], [1]])
            z = np.dot(self.weights[i], a_prev)
            if i == len(self.weights) - 1:
                activation = self.softmax(z)
            else:
                activation = self.sigmoid(z)
            activations.append(activation)
        return activations

    def train_batch(self, x_batch, y_batch):
        grad_sum = [np.zeros_like(w) for w in self.weights]
        total_loss = 0
        for i in range(len(x_batch)):
            activations = self.forward(x_batch[i])
            deltas = [None] * len(self.weights)
            deltas[-1] = activations[-1] - y_batch[i]
            for j in range(len(self.weights) - 2, -1, -1):
                w_no_bias = self.weights[j + 1][:, :-1]
                error = np.dot(w_no_bias.T, deltas[j + 1])
                deltas[j] = error * self.sigmoid_derivative(activations[j + 1])
            for j in range(len(self.weights)):
                a_with_bias = np.concatenate([activations[j], [1]])
                grad_sum[j] += np.outer(deltas[j], a_with_bias)
            total_loss += -np.sum(y_batch[i] * np.log(activations[-1] + 1e-8))
        for j in range(len(self.weights)):
            self.weights[j] -= self.lr * grad_sum[j] / len(x_batch)
        return total_loss / len(x_batch)

    def predict(self, x):
        return self.forward(x)[-1]

    def evaluate(self, x_data, y_data):
        correct = 0
        for i in range(len(x_data)):
            pred = np.argmax(self.predict(x_data[i]))
            if pred == y_data[i]:
                correct += 1
        return correct / len(y_data)


# 3. Experiment with different hidden layer neuron counts - using more extreme contrasts
neuron_counts = [32, 128, 512]  # smaller minimum value and larger maximum value
steps = 3000  # increase training steps
batch_size = 64

results = {}
for neurons in neuron_counts:
    print(f"\nTraining with hidden layer = {neurons} neurons...")
    model = DNN([784, neurons, 10], learning_rate=0.05)
    accuracies = []
    for n in range(steps):
        batch_indices = np.random.choice(len(x_train), batch_size, replace=False)
        x_batch, y_batch = x_train[batch_indices], encoded_labels[batch_indices]
        model.train_batch(x_batch, y_batch)
        if n % 100 == 0:
            # use the entire test set for more stable evaluation
            acc = model.evaluate(x_test, y_test)
            accuracies.append(acc)
            if n % 500 == 0:
                print(f"Step {n}, Accuracy = {acc:.4f}")
    results[neurons] = accuracies

# 4. Plot comparison chart - enhanced visualization
plt.figure(figsize=(10, 7))

# use different line styles and colors
markers = ['o', 's', '^']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (neurons, accs) in enumerate(results.items()):
    x_points = range(0, steps, 100)
    plt.plot(x_points, accs,
             marker=markers[i],
             markersize=4,
             color=colors[i],
             linewidth=2,
             label=f"{neurons} neurons")

    # add value annotations at key points
    if i == len(results) - 1:  # only annotate the largest network to avoid clutter
        for j in range(0, len(accs), 5):  # annotate every 5 points
            plt.annotate(f'{accs[j]:.3f}',
                         xy=(x_points[j], accs[j]),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=8,
                         color=colors[i])

plt.xlabel("Training Step", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Effect of Hidden Layer Size on Model Performance", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# set y-axis range to highlight differences
min_acc = min([min(accs) for accs in results.values()])
max_acc = max([max(accs) for accs in results.values()])
plt.ylim(min_acc * 0.95, max_acc * 1.01)

plt.tight_layout()
plt.savefig("hidden_neurons_comparison_enhanced.pdf", dpi=300, bbox_inches='tight')
plt.show()

# 5. Add final performance comparison table
print("\nFinal Performance Comparison:")
print("Neurons | Final Accuracy | Improvement from 32 neurons")
print("--------|----------------|----------------------------")
base_acc = results[32][-1]
for neurons in neuron_counts:
    final_acc = results[neurons][-1]
    improvement = final_acc - base_acc
    print(f"{neurons:6d} | {final_acc:.4f}        | {improvement:.4f} ({improvement / base_acc * 100:.1f}%)")