Neuro-Network
A simple neural network for predicting hand-drawn digits with Python.
<img src="https://github.com/user-attachments/assets/4085c676-077e-4615-bc49-5d80abbe6121" width="400" alt="Digit Recognition Demo">
Overview
This project implements a neural network using NumPy to recognize handwritten digits from the MNIST dataset, with a drawing interface for testing.

Architecture: 784 input neurons → 128 hidden neurons → 10 output neurons
Features: Custom NN implementation, interactive drawing UI, configurable speed/accuracy tradeoffs

Requirements

Python 3.6+
NumPy, Matplotlib, TensorFlow (for dataset), Pillow, Tkinter

Installation
bashCopygit clone https://github.com/yourusername/Neuro-Network.git
pip install numpy matplotlib tensorflow pillow
python main.py
Usage

Run the application
Draw a digit on the canvas
Click "Predict" to see the network's prediction
Use "Clear" to reset the canvas

Performance Tuning
Adjust these parameters to balance speed and accuracy:

Hidden layer size (n_h1)
Training examples (subset_size)
Epochs, batch size, learning rate
