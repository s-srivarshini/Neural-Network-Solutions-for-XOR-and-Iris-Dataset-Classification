# Neural-Network-Solutions-for-XOR-and-Iris-Dataset-Classification

## Overview
This repository contains the implementation and application of neural network models to solve the XOR problem and classify the Iris dataset. It leverages PyTorch, a leading deep learning library, to build and train the models. The XOR problem demonstrates the capability of neural networks to model non-linear relationships, while the Iris dataset classification showcases a multi-layer perceptron (MLP) classifier's effectiveness in handling multi-class classification problems.

## Getting Started
### Installation
To run the notebook and replicate the findings, ensure you have the following Python packages installed:
```bash
pip install torch
pip install numpy
pip install scikit-learn
pip install matplotlib
```

## Implementation Details
**XOR Problem**
- Model: A simple neural network with one hidden layer.
- Activation Function: Sigmoid.
- Training: Uses backpropagation and gradient descent for weight updates.

**Iris Dataset Classification**
- Model: Multi-Layer Perceptron (MLP) classifier with one hidden layer.
- Activation Function: Sigmoid for both hidden and output layers.
- Training: Utilizes Stochastic Gradient Descent (SGD) for optimization.

## Results
**XOR Problem**: The neural network model was able to learn the XOR function successfully, demonstrating the power of neural networks in capturing non-linear relationships.

**Iris Dataset Classification**: The MLP classifier achieved high accuracy in classifying the Iris dataset. Performance metrics such as precision, recall, and F1 score were used to evaluate the model thoroughly.

## Key Observations
- The number of neurons in the hidden layer significantly influences the model's ability to learn complex patterns.
- Proper normalization of input data is crucial for the training process.
- The choice of hyperparameters, like the learning rate, affects the convergence and overall performance of the models.
