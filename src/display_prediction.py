import numpy as np
import matplotlib.pyplot as plt
from ReadMNIST import return_test_images, return_test_labels
from FNN import FNN

# Load test images and labels
test_images = return_test_images()
test_labels = return_test_labels()

# Normalize images
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

# Initialize the FNN
fnn = FNN()

# Select a random image
random_index = np.random.randint(0, test_images.shape[0])
random_image = test_images[random_index]
random_label = test_labels[random_index]

# Get FNN prediction
activations, _ = fnn.FeedForward(random_image)
predicted_label = np.argmax(activations[-1])

# Reshape the random image back to its original 28x28 shape for display
reshaped_image = random_image.reshape(28, 28)

# Display the image and FNN prediction
plt.imshow(reshaped_image, cmap='gray')
plt.title(f"True Label: {random_label}, Predicted Label: {predicted_label}")
plt.axis('off')
plt.show()
