"""
q3 Building Convolutional Neural Network using NumPy from Scratch
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
import sys

# Set random seed for reproducible results
np.random.seed(42)


# ------------------------------
# 1. Convolution Helper Functions
# ------------------------------
def conv_(img, conv_filter):
    """Perform convolution on a single-channel image"""
    filter_size = conv_filter.shape[0]
    result = np.zeros((img.shape))

    # Loop through the image to apply convolution
    for r in np.uint16(np.arange(filter_size / 2.0, img.shape[0] - filter_size / 2.0 + 1)):
        for c in np.uint16(np.arange(filter_size / 2.0, img.shape[1] - filter_size / 2.0 + 1)):
            # Get the current region to multiply with the filter
            curr_region = img[r - np.uint16(np.floor(filter_size / 2.0)):r + np.uint16(np.ceil(filter_size / 2.0)),
            c - np.uint16(np.floor(filter_size / 2.0)):c + np.uint16(np.ceil(filter_size / 2.0))]
            # Element-wise multiplication between region and filter
            curr_result = curr_region * conv_filter
            # Sum the results
            conv_sum = np.sum(curr_result)
            # Save the result in the feature map
            result[r, c] = conv_sum

    # Clip the outliers of the result matrix
    final_result = result[np.uint16(filter_size / 2.0):result.shape[0] - np.uint16(filter_size / 2.0),
    np.uint16(filter_size / 2.0):result.shape[1] - np.uint16(filter_size / 2.0)]
    return final_result


def conv(img, conv_filter):
    """Perform convolution on multi-channel images"""
    # Check if number of channels matches
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in image and filter must match")
            sys.exit()

    # Check if filter is square
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print('Error: Filter must be a square matrix')
        sys.exit()

    # Check if filter has odd size
    if conv_filter.shape[1] % 2 == 0:
        print('Error: Filter must have an odd size')
        sys.exit()

    # Initialize feature maps
    feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                             img.shape[1] - conv_filter.shape[1] + 1,
                             conv_filter.shape[0]))

    # Convolve the image with each filter
    for filter_num in range(conv_filter.shape[0]):
        print(f"Filter {filter_num + 1}")
        curr_filter = conv_filter[filter_num, :]  # Get a filter from the bank

        # Check if filter has multiple channels
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])  # Array holding sum of feature maps
            for ch_num in range(1, curr_filter.shape[-1]):  # Convolve each channel and sum results
                conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
        else:
            conv_map = conv_(img, curr_filter)  # Single channel filter

        feature_maps[:, :, filter_num] = conv_map  # Save feature map for current filter

    return feature_maps  # Return all feature maps


# ------------------------------
# 2. Pooling Function
# ------------------------------
def pooling(feature_map, size=2, stride=2):
    """Perform max pooling operation"""
    # Prepare output of pooling operation
    pool_out = np.zeros((np.uint16((feature_map.shape[0] - size + 1) / stride + 1),
                         np.uint16((feature_map.shape[1] - size + 1) / stride + 1),
                         feature_map.shape[-1]))

    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0] - size + 1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1] - size + 1, stride):
                pool_out[r2, c2, map_num] = np.max(feature_map[r:r + size, c:c + size, map_num])
                c2 = c2 + 1
            r2 = r2 + 1

    return pool_out


# ------------------------------
# 3. ReLU Activation Function
# ------------------------------
def relu(feature_map):
    """Apply ReLU activation function"""
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out


# ------------------------------
# 4. Main Program
# ------------------------------
# Read image (using the same image as in the original article)
print("Loading image...")
img = data.chelsea()  # Use the standard test image from the article

# Convert to grayscale
img = color.rgb2gray(img)

# Display input image
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.title("Input Image")
plt.axis("off")
plt.savefig("input_image.png", bbox_inches="tight")
plt.close()

# ------------------------------
# First Convolutional Layer
# ------------------------------
print("\n**Working with Conv Layer 1**")
l1_filter = np.zeros((2, 3, 3))
l1_filter[0, :, :] = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])  # Vertical edge detector
l1_filter[1, :, :] = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])  # Horizontal edge detector

l1_feature_map = conv(img, l1_filter)
print("\n**Applying ReLU activation**")
l1_feature_map_relu = relu(l1_feature_map)
print("\n**Applying pooling operation**")
l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)
print("**Conv Layer 1 complete**\n")

# Visualize Layer 1 results
fig1, ax1 = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
ax1[0, 0].imshow(l1_feature_map[:, :, 0], cmap="gray")
ax1[0, 0].set_title("L1-FeatureMap1")
ax1[0, 0].axis("off")

ax1[0, 1].imshow(l1_feature_map[:, :, 1], cmap="gray")
ax1[0, 1].set_title("L1-FeatureMap2")
ax1[0, 1].axis("off")

ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0], cmap="gray")
ax1[1, 0].set_title("L1-FeatureMap1(ReLU)")
ax1[1, 0].axis("off")

ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1], cmap="gray")
ax1[1, 1].set_title("L1-FeatureMap2(ReLU)")
ax1[1, 1].axis("off")

ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0], cmap="gray")
ax1[2, 0].set_title("L1-FeatureMap1(ReLU+Pool)")
ax1[2, 0].axis("off")

ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1], cmap="gray")
ax1[2, 1].set_title("L1-FeatureMap2(ReLU+Pool)")
ax1[2, 1].axis("off")

plt.tight_layout()
plt.savefig("L1_results.png", bbox_inches="tight")
plt.close(fig1)

# ------------------------------
# Second Convolutional Layer
# ------------------------------
print("\n**Working with Conv Layer 2**")
l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
print("\n**Applying ReLU activation**")
l2_feature_map_relu = relu(l2_feature_map)
print("\n**Applying pooling operation**")
l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)
print("**Conv Layer 2 complete**\n")

# Visualize Layer 2 results
fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
for i in range(3):
    ax2[0, i].imshow(l2_feature_map[:, :, i], cmap="gray")
    ax2[0, i].set_title(f"L2-FeatureMap{i + 1}")
    ax2[0, i].axis("off")

    ax2[1, i].imshow(l2_feature_map_relu[:, :, i], cmap="gray")
    ax2[1, i].set_title(f"L2-FeatureMap{i + 1}(ReLU)")
    ax2[1, i].axis("off")

    ax2[2, i].imshow(l2_feature_map_relu_pool[:, :, i], cmap="gray")
    ax2[2, i].set_title(f"L2-FeatureMap{i + 1}(ReLU+Pool)")
    ax2[2, i].axis("off")

plt.tight_layout()
plt.savefig("L2_results.png", bbox_inches="tight")
plt.close(fig2)

# ------------------------------
# Third Convolutional Layer
# ------------------------------
print("\n**Working with Conv Layer 3**")
l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
print("\n**Applying ReLU activation**")
l3_feature_map_relu = relu(l3_feature_map)
print("\n**Applying pooling operation**")
l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
print("**Conv Layer 3 complete**\n")

# Visualize Layer 3 results
fig3, ax3 = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
ax3[0].imshow(l3_feature_map[:, :, 0], cmap="gray")
ax3[0].set_title("L3-FeatureMap1")
ax3[0].axis("off")

ax3[1].imshow(l3_feature_map_relu[:, :, 0], cmap="gray")
ax3[1].set_title("L3-FeatureMap1(ReLU)")
ax3[1].axis("off")

ax3[2].imshow(l3_feature_map_relu_pool[:, :, 0], cmap="gray")
ax3[2].set_title("L3-FeatureMap1(ReLU+Pool)")
ax3[2].axis("off")

plt.tight_layout()
plt.savefig("L3_results.png", bbox_inches="tight")
plt.close(fig3)

print(" CNN pipeline complete. Results saved as L1_results.png, L2_results.png, L3_results.png")