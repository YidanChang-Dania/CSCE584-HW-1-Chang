"""
Reproduce Example 3

Two single-input neural networks N_f and N_g:
 - architecture: [1, 20, 10, 1]
 - hidden layers: sigmoid
 - output layer: linear

Training:
 - Sample pairs (y, u) from ranges covering the plant behaviour:
    y in [-10,10] (for f), u in [-2,2] (for g)
 - Use SGD (batch size = 1), learning rate eta
 - Number of iterations: TRAIN_STEPS (default 100000)

Testing:
 - Use u(k) = sin(2πk/25) + sin(2πk/10), simulate plant with true f,g,
 - Compare plant y_p(k) vs identification model output y_hat(k) where
   y_hat(k+1) = N_f(y_p(k)) + N_g(u(k))   (series-parallel style used in identification)
"""
import numpy as np
import matplotlib.pyplot as plt


# Neural network class
class NeuralNetwork:
    def __init__(self, layer_sizes, seed=None):
        """
        layer_sizes: list, e.g. [1,20,10,1]
        """
        if seed is not None:
            np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # number of weight layers

        # Xavier / Glorot-ish init for weights, zero biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            w = np.random.randn(in_dim, out_dim) * np.sqrt(1.0 / in_dim)
            b = np.zeros((1, out_dim))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        # numerically safe sigmoid
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime_from_activation(self, a):
        # derivative using activation
        return a * (1.0 - a)

    def forward(self, x):
        """
        x: shape (batch, input_dim)
        returns activation of last layer (batch, output_dim)
        stores activations and zs for backprop
        NOTE: All hidden layers -> sigmoid, last layer -> linear
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.activations = [x]
        self.zs = []
        for i in range(self.num_layers):
            z = self.activations[-1].dot(self.weights[i]) + self.biases[i]
            self.zs.append(z)
            # hidden layers: sigmoid; output layer: linear
            if i < self.num_layers - 1:
                a = self.sigmoid(z)
            else:
                a = z  # linear output
            self.activations.append(a)
        return self.activations[-1]

    def sgd_update(self, x, y_target, lr):
        """
        Single-sample (or small-batch) gradient descent update.
        x: (1, input_dim)
        y_target: (1, output_dim)
        lr: learning rate
        """
        # forward pass
        self.forward(x)

        # compute delta for output layer: linear output -> delta = (a_L - y)
        a_L = self.activations[-1]              # shape (1, out)
        delta_L = (a_L - y_target)              # shape (1, out)
        deltas = [None] * self.num_layers
        deltas[-1] = delta_L

        # backpropagate through hidden layers
        for l in range(self.num_layers - 2, -1, -1):
            # delta_l = (delta_{l+1} dot W_{l+1}^T) * sigma'(a_l)
            a_l = self.activations[l+1]  # activation of layer l+1 (hidden)
            delta_next = deltas[l+1]
            w_next = self.weights[l+1]
            delta_l = delta_next.dot(w_next.T) * self.sigmoid_prime_from_activation(a_l)
            deltas[l] = delta_l

        # gradient descent weight updates
        m = x.shape[0]  # batch size (here usually 1)
        for l in range(self.num_layers):
            grad_w = self.activations[l].T.dot(deltas[l]) / m   # shape (in, out)
            grad_b = np.mean(deltas[l], axis=0, keepdims=True)
            # gradient descent step
            self.weights[l] -= lr * grad_w
            self.biases[l]  -= lr * grad_b

        # return squared error for logging (MSE per sample)
        return float(np.mean((a_L - y_target)**2))



# True plant functions
def f_true(y):
    # f[y] = y/(1 + y^2)
    return y / (1.0 + y**2)

def g_true(u):
    # g[u] = u^3
    return u**3

# Training & simulation routine
def run_example3(seed=123,
                 TRAIN_STEPS=100000,
                 lr_f=0.1,
                 lr_g=0.1,
                 show_plots=True,
                 save_fig=False,
                 figname="example3_repro.png"):
    np.random.seed(seed)

    # create networks N_f and N_g
    nn_f = NeuralNetwork([1, 20, 10, 1], seed=seed+1)
    nn_g = NeuralNetwork([1, 20, 10, 1], seed=seed+2)

    # training logging
    training_errors = []   # combined squared error per iteration (f+g vs target)
    log_every = 1000

    # sample training points independently:
    # y ~ Uniform(-10, 10)  (f domain of interest)
    # u ~ Uniform(-2, 2)    (g domain)
    for k in range(TRAIN_STEPS):
        # sample random y and u
        y_samp = np.random.uniform(-10.0, 10.0)
        u_samp = np.random.uniform(-2.0, 2.0)

        # true targets
        t_f = np.array([[f_true(y_samp)]])   # shape (1,1)
        t_g = np.array([[g_true(u_samp)]])   # shape (1,1)

        # forward & update nn_f on (y -> f(y))
        x_f = np.array([[y_samp]])
        err_f = nn_f.sgd_update(x_f, t_f, lr_f)

        # forward & update nn_g on (u -> g(u))
        x_g = np.array([[u_samp]])
        err_g = nn_g.sgd_update(x_g, t_g, lr_g)

        training_errors.append(err_f + err_g)

        if (k+1) % log_every == 0:
            recent = np.mean(training_errors[-log_every:])
            print(f"Step {k+1}/{TRAIN_STEPS}  recent MSE (avg of last {log_every}) = {recent:.6e}")

    print("Training complete.")


    # Prepare test: sinusoidal input and plant simulation
    test_steps = 100
    y_p = np.zeros(test_steps)
    y_hat = np.zeros(test_steps)   # identification model output (series-parallel style)
    u_test = np.zeros(test_steps)

    # initial condition
    y_p[0] = 0.0
    # run for test_steps-1 transitions
    for k in range(test_steps-1):
        u = np.sin(2*np.pi*k/25.0) + np.sin(2*np.pi*k/10.0)
        u_test[k] = u
        # plant true next
        y_next = f_true(y_p[k]) + g_true(u)
        y_p[k+1] = y_next

        # identification model predicted output in series-parallel:
        # use actual plant y_p[k] as input to N_f (series-parallel identification)
        f_est = nn_f.forward(np.array([[y_p[k]]]))[0,0]
        g_est = nn_g.forward(np.array([[u]]))[0,0]
        y_hat[k+1] = f_est + g_est

    # final u entry
    u_test[-1] = np.sin(2*np.pi*(test_steps-1)/25.0) + np.sin(2*np.pi*(test_steps-1)/10.0)


    # Prepare function plots: evaluate true vs estimate on grid
    y_grid = np.linspace(-10, 10, 1000)
    u_grid = np.linspace(-2, 2, 1000)

    f_true_vals = f_true(y_grid)
    g_true_vals = g_true(u_grid)

    f_est_vals = np.array([nn_f.forward(np.array([[yy]]))[0,0] for yy in y_grid])
    g_est_vals = np.array([nn_g.forward(np.array([[uu]]))[0,0] for uu in u_grid])


    # Plot results (Fig.15 style)
    if show_plots:
        fig = plt.figure(figsize=(14,9))

        # (a) f and f_hat
        ax1 = plt.subplot(2,2,1)
        ax1.plot(y_grid, f_true_vals, 'b-', label=r"$f[y]=y/(1+y^2)$")
        ax1.plot(y_grid, f_est_vals, 'r--', label=r"Estimate $\hat f$")
        ax1.set_xlim(-10,10)
        ax1.set_ylim(-0.6,0.6)
        ax1.set_title("(a) Plots of the functions f and $\hat f$")
        ax1.grid(True)
        ax1.legend()

        # (b) g and g_hat
        ax2 = plt.subplot(2,2,2)
        ax2.plot(u_grid, g_true_vals, 'b-', label=r"$g[u]=u^3$")
        ax2.plot(u_grid, g_est_vals, 'r--', label=r"Estimate $\hat g$")
        ax2.set_xlim(-2,2)
        ax2.set_ylim(-10,10)
        ax2.set_title("(b) Plots of the functions g and $\hat g$")
        ax2.grid(True)
        ax2.legend()

        # (c) outputs of plant and identification model (test)
        ax3 = plt.subplot(2,2,3)
        ax3.plot(y_p, 'b-', label=r"$y_p$ (plant)")
        ax3.plot(y_hat, 'r--', label=r"$\hat y_p$ (ident. model)")
        ax3.set_title("(c) Outputs of the plant and the identification model")
        ax3.set_xlabel("k")
        ax3.grid(True)
        ax3.legend()
        # annotate input used for test
        plt.figtext(0.5, 0.01, "u(k) = sin(2πk/25) + sin(2πk/10)", ha='center')

        # (d) training error (log scale)
        ax4 = plt.subplot(2,2,4)
        # plot running-average of training_errors to smooth
        te = np.array(training_errors)
        window = 1000
        if te.size >= window:
            # moving mean
            mov = np.convolve(te, np.ones(window)/window, mode='valid')
            ax4.semilogy(mov, '-')
            ax4.set_xlabel("Training iterations (thousands)")
        else:
            ax4.semilogy(te, '-')
            ax4.set_xlabel("Training iterations")
        ax4.set_title("Mean Square Error (log scale)")
        ax4.grid(True)

        plt.tight_layout()
        if save_fig:
            plt.savefig(figname, dpi=200)
        plt.show()

    # return objects for further analysis if needed
    return {
        "nn_f": nn_f,
        "nn_g": nn_g,
        "training_errors": training_errors,
        "y_p_test": y_p,
        "y_hat_test": y_hat,
        "u_test": u_test,
        "f_true_vals": f_true_vals,
        "f_est_vals": f_est_vals,
        "g_true_vals": g_true_vals,
        "g_est_vals": g_est_vals
    }

if __name__ == "__main__":
    results = run_example3(seed=123, TRAIN_STEPS=100000, lr_f=0.1, lr_g=0.1,
                           show_plots=True, save_fig=False)
