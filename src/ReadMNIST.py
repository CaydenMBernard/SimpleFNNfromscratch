import os
import numpy as np
import struct
import matplotlib.pyplot as plt

def read_idx_images(file_path):
    with open(file_path, 'rb') as f:
        # Read the header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in IDX image file!")
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num_images, rows, cols))
    return images

def read_idx_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read the header
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in IDX label file!")
        
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Base directory for the MNIST files relative to the src folder
base_dir = os.path.join("FNN", "src", "MNIST digits")

# File paths
train_images_file = os.path.join(base_dir, "train-images.idx3-ubyte")
train_labels_file = os.path.join(base_dir, "train-labels.idx1-ubyte")

# Load training data
train_images = read_idx_images(train_images_file)
train_labels = read_idx_labels(train_labels_file)

# Print dataset info
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")

# Display a random image
random_index = np.random.randint(0, train_images.shape[0])
random_image = train_images[random_index]
random_label = train_labels[random_index]

plt.imshow(random_image, cmap='gray')
plt.title(f"Label: {random_label}")
plt.axis('off')
plt.show()
