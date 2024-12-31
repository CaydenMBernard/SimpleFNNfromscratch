import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from ReadMNIST import return_test_images, return_test_labels
from FNN import FNN
import sys

# Load test images and labels
test_images = return_test_images()
test_labels = return_test_labels()

# Normalize images
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

# Initialize the FNN
fnn = FNN()

# Function to display a random number
def display_random_number():
    random_index = np.random.randint(0, test_images.shape[0])
    random_image = test_images[random_index]
    random_label = test_labels[random_index]

    # Get FNN prediction
    activations, _ = fnn.FeedForward(random_image)
    predicted_label = np.argmax(activations[-1])

    # Reshape the random image back to its original 28x28 shape
    reshaped_image = random_image.reshape(28, 28)

    # Clear the canvas and redraw the new image
    ax.clear()
    ax.imshow(reshaped_image, cmap='gray')
    ax.set_title(f"True Label: {random_label}, Predicted Label: {predicted_label}", fontsize=20)
    ax.axis('off')
    canvas.draw()

# Properly stop the program when the window is closed
def on_closing():
    root.destroy()
    sys.exit()

# Create the main tkinter window
root = tk.Tk()
root.title("MNIST Predictor")
root.protocol("WM_DELETE_WINDOW", on_closing)
root.geometry("1000x1000") 

# Create a matplotlib figure with a larger size
fig, ax = plt.subplots(figsize=(8, 8)) 
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=20)
button = tk.Button(root, text="Display New Number", command=display_random_number, font=("Helvetica", 16))
button.pack(pady=20) 

display_random_number()
root.mainloop()
