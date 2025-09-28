"""
q6 GANs from Scratch 1: A deep introduction.
With code in PyTorch and TensorFlow

GAN
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from utils import Logger

# ----------------------
# Data Preparation
# ----------------------
def mnist_data():
    """
    Download and normalize the MNIST dataset.
    Normalization: scale pixel values to [-1, 1].
    """
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


data = mnist_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)


# ----------------------
# Discriminator Network
# ----------------------
class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer discriminative neural network.
    Input: Flattened 28x28 image (784 values).
    Output: Single probability (real or fake).
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),  # LeakyReLU helps avoid "dying ReLU" problem
            nn.Dropout(0.3)     # Dropout prevents overfitting
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()  # Output is probability [0,1]
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


# ----------------------
# Generator Network
# ----------------------
class GeneratorNet(nn.Module):
    """
    A three hidden-layer generative neural network.
    Input: Random noise vector of length 100.
    Output: Flattened 28x28 image (784 values).
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()  # Outputs are scaled to [-1, 1] (consistent with normalization)
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


# ----------------------
# Utility Functions
# ----------------------
def images_to_vectors(images):
    """Convert batch of images into 1D vectors"""
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    """Reshape vectors back into image format (batch, 1, 28, 28)"""
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    """Generate random Gaussian noise vectors"""
    return Variable(torch.randn(size, 100))

def ones_target(size):
    """Tensor of ones (used for real data labels)"""
    return Variable(torch.ones(size, 1))

def zeros_target(size):
    """Tensor of zeros (used for fake data labels)"""
    return Variable(torch.zeros(size, 1))


# ----------------------
# Model Initialization
# ----------------------
discriminator = DiscriminatorNet()
generator = GeneratorNet()

# Adam optimizers for both networks
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Binary cross entropy loss
loss = nn.BCELoss()


# ----------------------
# Training Functions
# ----------------------
def train_discriminator(optimizer, real_data, fake_data):
    """
    Train the discriminator on both real and fake data.
    Goal: Maximize ability to distinguish real from fake.
    """
    N = real_data.size(0)
    optimizer.zero_grad()

    # Train on real data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # Train on fake data
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    """
    Train the generator to fool the discriminator.
    Goal: Maximize D(G(z)) → close to 1 (real).
    """
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, ones_target(N))  # Pretend generated data is real
    error.backward()
    optimizer.step()
    return error


# ----------------------
# Main Training Loop
# ----------------------
num_test_samples = 16
test_noise = noise(num_test_samples)

logger = Logger(model_name='VGAN', data_name='MNIST')
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        N = real_batch.size(0)

        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        fake_data = generator(noise(N)).detach()  # detach so generator is not updated here
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        fake_data = generator(noise(N))
        g_error = train_generator(g_optimizer, fake_data)

        # Logging
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Every 100 batches, log sample images
        if n_batch % 100 == 0:
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
            logger.display_status(epoch, num_epochs, n_batch, num_batches,
                                  d_error, g_error, d_pred_real, d_pred_fake)




