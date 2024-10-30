# Multilayer Perceptron for Iris Classification

This project implements and compares three different neural network classifiers for Iris flower classification using a multilayer perceptron (MLP) architecture. The project includes a custom MLP implemented with NumPy, a scikit-learn MLP, and a TensorFlow model. The unique aspect of the custom MLP model is its dynamic configuration, allowing users to customize the number of layers and neurons in each layer.

# Key Features
- Custom MLP with NumPy
- Customizable network structure (layers and neurons per layer)
- Comparison of loss and accuracy across models

# Project Structure
The project files include:
**PrayagPiya_MPL.py**: Custom MLP implementation using NumPy.
**main.py**: Main file to execute.

# Strategy and Implementation
- **User Input**: The user specifies the number of layers, neurons per layer, and input/output neurons.
- Initialization: Based on user input, weights and biases are initialized randomly to match the network structure.
- Forward Propagation:
  * Linear and ReLU Activation: Hidden layers use the ReLU activation function, while the output layer uses Softmax for probability distribution.
  * Softmax is ideal for multi-class classification problems, making it a suitable choice for Iris classification.
- Backward Propagation:
  * Gradient Descent: Optimizes weights and biases by calculating and applying gradients based on the loss.
  * Categorical Cross-Entropy: Measures the strength of the prediction, using the predicted and actual outputs to compute the error.

# Loss and Accuracy Comparison
The models’ performance can be observed through loss and accuracy:
![Loss and Accuracy for the custom NumPy-based MLP over time.]Images/custome.png

![Loss and Accuracy using TensorFlow.]Images/tensorflow.png

These plots provide insights into the models' convergence and predictive capabilities, showing how each algorithm’s performance varies based on structure and optimizer.
