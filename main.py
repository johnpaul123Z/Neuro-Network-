import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk

# ----------------------------
# 1. Neural Network Functions
# ----------------------------

def softmax(x):
    """
    Compute the softmax of input x along the columns.
    """
    ex = np.exp(x - np.max(x, axis=0, keepdims=True))
    return ex / np.sum(ex, axis=0, keepdims=True)

def ReLU(x):
    """
    Apply the ReLU activation function.
    """
    return np.maximum(0, x)

def onehot(y, num_classes=10):
    """
    Convert label vector y into a one-hot encoded matrix with a fixed number of classes.
    """
    onehot_vec = np.zeros((y.size, num_classes))
    onehot_vec[np.arange(y.size), y] = 1
    return onehot_vec.T  # (num_classes, m)

def init_params():
    """
    Initialize parameters for a network with two hidden layers.
      - Layer 1: 256 neurons (He initialization)
      - Layer 2: 128 neurons (He initialization)
      - Output: 10 neurons (Xavier initialization)
    """
    n_x = 784      # 28x28 input
    n_h1 = 256     # First hidden layer size
    n_h2 = 128     # Second hidden layer size
    n_y = 10       # 10 output classes

    w1 = np.random.randn(n_h1, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h1, 1))
    w2 = np.random.randn(n_h2, n_h1) * np.sqrt(2.0 / n_h1)
    b2 = np.zeros((n_h2, 1))
    w3 = np.random.randn(n_y, n_h2) * np.sqrt(1.0 / n_h2)  # Xavier for output layer
    b3 = np.zeros((n_y, 1))
    return w1, b1, w2, b2, w3, b3

def forward_propagation(w1, b1, w2, b2, w3, b3, x):
    """
    Perform forward propagation through the network.
    Returns all intermediate values needed for backpropagation.
    """
    z1 = np.dot(w1, x) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = ReLU(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def deriv_relu(x):
    """
    Derivative of the ReLU activation.
    """
    return (x > 0).astype(float)

def backward_propagation(z1, a1, z2, a2, z3, a3, w3, w2, w1, x, y, lambda_reg):
    """
    Backpropagation for a three-layer network.
    L2 regularization is added to the weight gradients.
    Now includes w1 for computing the gradient for the first hidden layer.
    """
    m_samples = y.size
    onehot_y = onehot(y, num_classes=10)
    
    # Output layer gradients
    dz3 = a3 - onehot_y
    dw3 = (1/m_samples) * np.dot(dz3, a2.T) + (lambda_reg/m_samples) * w3
    db3 = (1/m_samples) * np.sum(dz3, axis=1, keepdims=True)
    
    # Second hidden layer gradients
    da2 = np.dot(w3.T, dz3)
    dz2 = da2 * deriv_relu(z2)
    dw2 = (1/m_samples) * np.dot(dz2, a1.T) + (lambda_reg/m_samples) * w2
    db2 = (1/m_samples) * np.sum(dz2, axis=1, keepdims=True)
    
    # First hidden layer gradients
    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * deriv_relu(z1)
    dw1 = (1/m_samples) * np.dot(dz1, x.T) + (lambda_reg/m_samples) * w1
    db1 = (1/m_samples) * np.sum(dz1, axis=1, keepdims=True)
    
    return dw1, db1, dw2, db2, dw3, db3

def get_accuracy(a3, y):
    """
    Calculate accuracy by comparing predictions with true labels.
    """
    predictions = np.argmax(a3, axis=0)
    return np.mean(predictions == y)

def compute_loss(a3, y, w1, w2, w3, lambda_reg):
    """
    Compute cross-entropy loss with L2 regularization.
    """
    m_samples = y.size
    onehot_y = onehot(y, num_classes=10)
    cross_entropy = -np.sum(onehot_y * np.log(a3 + 1e-8)) / m_samples
    reg_loss = (lambda_reg / (2 * m_samples)) * (np.sum(w1**2) + np.sum(w2**2) + np.sum(w3**2))
    return cross_entropy + reg_loss

def adam_update(param, dparam, m, v, t, learning_rate, beta1, beta2, epsilon):
    """
    Update parameters using Adam optimization.
    """
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

def gradient_descent(x, y, epochs, learning_rate, batch_size=128, lambda_reg=0.001,
                     beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Train the neural network using mini-batch gradient descent with Adam optimization.
    Uses full MNIST training data and applies L2 regularization.
    """
    w1, b1, w2, b2, w3, b3 = init_params()
    
    # Initialize Adam moment estimates for all parameters
    m_w1, v_w1 = np.zeros_like(w1), np.zeros_like(w1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_w2, v_w2 = np.zeros_like(w2), np.zeros_like(w2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    m_w3, v_w3 = np.zeros_like(w3), np.zeros_like(w3)
    m_b3, v_b3 = np.zeros_like(b3), np.zeros_like(b3)
    
    m_examples = y.size
    t = 0  # Adam time step
    for epoch in range(epochs):
        # Shuffle training data
        permutation = np.random.permutation(m_examples)
        x_shuffled = x[:, permutation]
        y_shuffled = y[permutation]
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, m_examples, batch_size):
            t += 1
            x_batch = x_shuffled[:, i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            z1, a1, z2, a2, z3, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, x_batch)
            loss = compute_loss(a3, y_batch, w1, w2, w3, lambda_reg)
            epoch_loss += loss
            batch_count += 1
            
            dw1, db1, dw2, db2, dw3, db3 = backward_propagation(z1, a1, z2, a2, z3, a3,
                                                                 w3, w2, w1, x_batch, y_batch, lambda_reg)
            
            # Update parameters with Adam
            w1, m_w1, v_w1 = adam_update(w1, dw1, m_w1, v_w1, t, learning_rate, beta1, beta2, epsilon)
            b1, m_b1, v_b1 = adam_update(b1, db1, m_b1, v_b1, t, learning_rate, beta1, beta2, epsilon)
            w2, m_w2, v_w2 = adam_update(w2, dw2, m_w2, v_w2, t, learning_rate, beta1, beta2, epsilon)
            b2, m_b2, v_b2 = adam_update(b2, db2, m_b2, v_b2, t, learning_rate, beta1, beta2, epsilon)
            w3, m_w3, v_w3 = adam_update(w3, dw3, m_w3, v_w3, t, learning_rate, beta1, beta2, epsilon)
            b3, m_b3, v_b3 = adam_update(b3, db3, m_b3, v_b3, t, learning_rate, beta1, beta2, epsilon)
        
        avg_loss = epoch_loss / batch_count
        _, _, _, _, _, a3_full = forward_propagation(w1, b1, w2, b2, w3, b3, x)
        acc = get_accuracy(a3_full, y)
        print(f"Epoch: {epoch+1}, Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    
    return w1, b1, w2, b2, w3, b3

# ----------------------------
# 2. Train the Neural Network
# ----------------------------

# Load full MNIST training data (60,000 examples)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
m = train_images.shape[0]
x = train_images.reshape(m, -1).T  # (784, m)
x = x / 255.0                     # Normalize
y = train_labels                  # (m,)

# Hyperparameters for training
epochs = 30
learning_rate = 0.001
w1, b1, w2, b2, w3, b3 = gradient_descent(x, y, epochs, learning_rate, batch_size=128, lambda_reg=0.001)

# ----------------------------
# 3A. Evaluate on MNIST Test Set
# ----------------------------

m_test = test_images.shape[0]
x_test = test_images.reshape(m_test, -1).T / 255.0  # (784, m_test)
_, _, _, _, _, a3_test = forward_propagation(w1, b1, w2, b2, w3, b3, x_test)
test_acc = get_accuracy(a3_test, test_labels)
print("Test Accuracy:", test_acc)

# ----------------------------
# 3B. Drawing Interface for Testing
# ----------------------------

class DrawDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas_width = 280  # Enlarged canvas for drawing
        self.canvas_height = 280
        # Set canvas background to black
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.last_x, self.last_y = None, None
        # Create a PIL image with a black background
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw_image = ImageDraw.Draw(self.image)
        self.button = tk.Button(root, text="Predict", command=self.predict)
        self.button.pack()
        self.label = tk.Label(root, text="Draw a digit and click Predict")
        self.label.pack()
        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

    def draw(self, event):
        """
        Draw white lines on both the Tkinter canvas and the PIL image.
        """
        if self.last_x is not None and self.last_y is not None:
            # Draw white line
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill='white', width=8, capstyle=tk.ROUND, smooth=True)
            self.draw_image.line((self.last_x, self.last_y, event.x, event.y),
                                 fill='white', width=8)
        self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        # Reinitialize the PIL image with a black background
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw_image = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a digit and click Predict")

    def predict(self):
        """
        Process the drawing, resize the image, and predict the digit.
        Assumes the drawing is white on a black background.
        """
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.ANTIALIAS

        img = self.image.resize((28, 28), resample_filter)
        # No need to invert because the image is already white on black
        img_array = np.array(img) / 255.0
        img_flat = img_array.flatten().reshape(784, 1)
        _, _, _, _, _, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, img_flat)
        pred = np.argmax(a3, axis=0)[0]
        self.label.config(text=f"Prediction: {pred} (Confidence: {a3[pred][0]:.4f})")

if __name__ == '__main__':
    root = tk.Tk()
    app = DrawDigitApp(root)
    root.mainloop()
