
"""
q6 DCGAN
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import Logger

# --------------------------------------
# Load MNIST dataset (grayscale 1×28×28)
# Resize to 64×64 for DCGAN
# --------------------------------------
def mnist_data():
    transform_pipeline = transforms.Compose([
        transforms.Resize(64),  # DCGAN typically uses 64×64 input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    out_dir = 'data/images/VGAN/MNIST/dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=transform_pipeline, download=True)

data = mnist_data()
data_loader = DataLoader(data, batch_size=64, shuffle=True)
num_batches = len(data_loader)


# --------------------------------------
# DCGAN Discriminator (Convolutional)
# --------------------------------------
class DiscriminatorDCGAN(nn.Module):
    def __init__(self):
        super(DiscriminatorDCGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),        # 1×64×64 → 64×32×32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),      # 64×32×32 → 128×16×16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),     # 128×16×16 → 256×8×8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),     # 256×8×8 → 512×4×4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),       # 512×4×4 → 1×1×1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


# --------------------------------------
# DCGAN Generator (Transpose Convolutional)
# --------------------------------------
class GeneratorDCGAN(nn.Module):
    def __init__(self):
        super(GeneratorDCGAN, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0),   # 100×1×1 → 512×4×4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # 512×4×4 → 256×8×8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 256×8×8 → 128×16×16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 128×16×16 → 64×32×32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),      # 64×32×32 → 1×64×64
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------------
# Utility functions
# --------------------------------------
def noise(size):
    """Generate random noise input for the generator"""
    return Variable(torch.randn(size, 100, 1, 1))  # DCGAN expects 100×1×1 noise vector

def ones_target(size):
    """Return tensor of ones (for real samples)"""
    return Variable(torch.ones(size, 1))

def zeros_target(size):
    """Return tensor of zeros (for fake samples)"""
    return Variable(torch.zeros(size, 1))


# --------------------------------------
# Initialize models and optimizers
# --------------------------------------
discriminator = DiscriminatorDCGAN()
generator = GeneratorDCGAN()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss = nn.BCELoss()


# --------------------------------------
# Training functions
# --------------------------------------
def train_discriminator(optimizer, real_data, fake_data):
    """Train the discriminator with both real and fake data"""
    N = real_data.size(0)
    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    """Train the generator to fool the discriminator"""
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, ones_target(N))  # Generator wants D(G(z)) → 1
    error.backward()
    optimizer.step()
    return error


# --------------------------------------
# Training loop
# --------------------------------------
num_epochs = 50
num_test_samples = 16
test_noise = noise(num_test_samples)

logger = Logger(model_name='DCGAN', data_name='MNIST')

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        N = real_batch.size(0)
        real_data = Variable(real_batch)

        # 1. Train Discriminator
        fake_data = generator(noise(N)).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        fake_data = generator(noise(N))
        g_error = train_generator(g_optimizer, fake_data)

        # Logging and monitoring
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        if n_batch % 100 == 0:
            test_images = generator(test_noise).data
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
            logger.display_status(epoch, num_epochs, n_batch, num_batches,
                                  d_error, g_error, d_pred_real, d_pred_fake)
