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
    onehot_vec = np.zeros((y.size, num_classes), dtype=np.float32)
    onehot_vec[np.arange(y.size), y] = 1
    return onehot_vec.T  # (num_classes, m)

def init_params():
    """
    Initialize parameters for a simplified network with one hidden layer.
    - Layer 1: 128 neurons (He initialization)
    - Output: 10 neurons (Xavier initialization)
    All parameters are stored as float32.
    """
    n_x = 784      # 28x28 input
    n_h1 = 128     # Single hidden layer with reduced size
    n_y = 10       # 10 output classes

    w1 = (np.random.randn(n_h1, n_x) * np.sqrt(2.0 / n_x)).astype(np.float32)
    b1 = np.zeros((n_h1, 1), dtype=np.float32)
    w2 = (np.random.randn(n_y, n_h1) * np.sqrt(1.0 / n_h1)).astype(np.float32)  # Xavier for output layer
    b2 = np.zeros((n_y, 1), dtype=np.float32)
    return w1, b1, w2, b2

def forward_propagation(w1, b1, w2, b2, x):
    """
    Perform forward propagation through the simplified network.
    Returns all intermediate values needed for backpropagation.
    """
    z1 = np.dot(w1, x) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def deriv_relu(x):
    """
    Derivative of the ReLU activation.
    """
    return (x > 0).astype(np.float32)

def backward_propagation(z1, a1, z2, a2, w2, w1, x, y, lambda_reg):
    """
    Backpropagation for a two-layer network.
    L2 regularization is added to the weight gradients.
    """
    m_samples = y.size
    onehot_y = onehot(y, num_classes=10)
    
    # Output layer gradients
    dz2 = a2 - onehot_y
    dw2 = (1/m_samples) * np.dot(dz2, a1.T) + (lambda_reg/m_samples) * w2
    db2 = (1/m_samples) * np.sum(dz2, axis=1, keepdims=True)
    
    # Hidden layer gradients
    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * deriv_relu(z1)
    dw1 = (1/m_samples) * np.dot(dz1, x.T) + (lambda_reg/m_samples) * w1
    db1 = (1/m_samples) * np.sum(dz1, axis=1, keepdims=True)
    
    return dw1, db1, dw2, db2

def get_accuracy(a2, y):
    """
    Calculate accuracy by comparing predictions with true labels.
    """
    predictions = np.argmax(a2, axis=0)
    return np.mean(predictions == y)

def compute_loss(a2, y, w1, w2, lambda_reg):
    """
    Compute cross-entropy loss with L2 regularization.
    """
    m_samples = y.size
    onehot_y = onehot(y, num_classes=10)
    cross_entropy = -np.sum(onehot_y * np.log(a2 + 1e-8)) / m_samples
    reg_loss = (lambda_reg / (2 * m_samples)) * (np.sum(w1**2) + np.sum(w2**2))
    return cross_entropy + reg_loss

def sgd_update(param, dparam, learning_rate):
    """
    Update parameters using standard SGD.
    """
    return param - learning_rate * dparam

def gradient_descent(x, y, epochs, learning_rate, batch_size=512, lambda_reg=0.0005):
    """
    Train the neural network using mini-batch gradient descent with standard SGD.
    Uses a smaller subset of MNIST data for faster training.
    """
    # Use only a subset of the data for faster training
    subset_size = 20000  
    indices = np.random.permutation(y.size)[:subset_size]
    x = x[:, indices]
    y = y[indices]
    
    w1, b1, w2, b2 = init_params()
    
    m_examples = y.size
    for epoch in range(epochs):
        # Shuffle training data
        permutation = np.random.permutation(m_examples)
        x_shuffled = x[:, permutation]
        y_shuffled = y[permutation]
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, m_examples, batch_size):
            x_batch = x_shuffled[:, i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x_batch)
            loss = compute_loss(a2, y_batch, w1, w2, lambda_reg)
            epoch_loss += loss
            batch_count += 1
            
            dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w2, w1, x_batch, y_batch, lambda_reg)
            
            # Update parameters with SGD
            w1 = sgd_update(w1, dw1, learning_rate)
            b1 = sgd_update(b1, db1, learning_rate)
            w2 = sgd_update(w2, dw2, learning_rate)
            b2 = sgd_update(b2, db2, learning_rate)
        
        # Only calculate full dataset metrics every 5 epochs to save time
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / batch_count
            z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x)
            acc = get_accuracy(a2, y)
            print(f"Epoch: {epoch+1}, Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    
    return w1, b1, w2, b2

# ----------------------------
# 2. Train the Neural Network
# ----------------------------

# Load MNIST training data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
m = train_images.shape[0]
x = train_images.reshape(m, -1).T  # (784, m)
x = (x / 255.0).astype(np.float32) # Normalize and cast to float32
y = train_labels                  # (m,)

# Hyperparameters for faster training
epochs = 10  # Reduced epochs
learning_rate = 0.01  # Increased learning rate
w1, b1, w2, b2 = gradient_descent(x, y, epochs, learning_rate, batch_size=512, lambda_reg=0.0005)

# ----------------------------
# 3A. Evaluate on MNIST Test Set
# ----------------------------

m_test = test_images.shape[0]
x_test = test_images.reshape(m_test, -1).T / 255.0  # (784, m_test)
x_test = x_test.astype(np.float32)
_, _, _, a2_test = forward_propagation(w1, b1, w2, b2, x_test)
test_acc = get_accuracy(a2_test, test_labels)
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
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill='white', width=8, capstyle=tk.ROUND, smooth=True)
            self.draw_image.line((self.last_x, self.last_y, event.x, event.y),
                                 fill='white', width=8)
        self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
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
        img_array = np.array(img) / 255.0
        img_flat = img_array.flatten().reshape(784, 1).astype(np.float32)
        _, _, _, a2 = forward_propagation(w1, b1, w2, b2, img_flat)
        pred = np.argmax(a2, axis=0)[0]
        self.label.config(text=f"Prediction: {pred} (Confidence: {a2[pred][0]:.4f})")

if __name__ == '__main__':
    root = tk.Tk()
    app = DrawDigitApp(root)
    root.mainloop()
