import os
import numpy as np

# Network parameters
input_size = 784
num_hidden = 2
hidden_size = 128
output_size = 10

# Initialize weights
weights = [np.random.randn(hidden_size, input_size)]
for _ in range(num_hidden - 1):
    weights.append(np.random.randn(hidden_size, hidden_size))
weights.append(np.random.randn(output_size, hidden_size))  # Output layer weights

# Initialize biases
biases = [np.zeros(hidden_size) for _ in range(num_hidden)]
biases.append(np.zeros(output_size))  # Output layer biases

# Create the "Weights and Biases" folder
folder_path = os.path.join(os.path.dirname(__file__), "Weights and Biases")
os.makedirs(folder_path, exist_ok=True)

# Save weights and biases to the "Weights and Biases" folder
for i, w in enumerate(weights):
    np.save(os.path.join(folder_path, f'weight_layer_{i}.npy'), w)

for i, b in enumerate(biases):
    np.save(os.path.join(folder_path, f'bias_layer_{i}.npy'), b)

print(f"Weights and biases have been initialized and saved in {folder_path}.")
