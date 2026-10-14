# PyTorch Tutorial

Welcome to the **PyTorch-Tutorial** repository! This project provides hands-on code examples and tutorials designed to help you understand core concepts in PyTorch, including custom neural network modules, tensor operations, and model implementation.

## Features

* **Custom Neural Network Modules:** Learn how to inherit from `torch.nn.Module` to build modular architectures.
* **Polynomial Regression Implementation:** A custom implementation of a polynomial regression model leveraging PyTorch parameters and matrix operations.

---

## Featured Component: Polynomial Regression (`PolyRegression`)

This repository includes a custom PyTorch class named `PolyRegression` that computes polynomial regression efficiently using matrix operations rather than slow loops.

### Implementation

```python
import torch
import torch.nn as nn

class PolyRegression(nn.Module):
    def __init__(self, degree):
        super().__init__()  # Call the parent class's __init__

        self.degree = degree
        self.weights = nn.Parameter(torch.randn(degree + 1)) # Create weights as parameters

    def forward(self, x):
        """
        Forward pass of the polynomial regression model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = x.float() # Ensure x is float for calculations
        batch_size = x.shape[0] # Get the batch size
        
        # Expand x to have polynomial features
        x_powers = torch.ones((batch_size, self.degree + 1)) # Initialize with x^0 = 1
        for i in range(1, self.degree + 1):
            x_powers[:, i] = x.pow(i).squeeze() # Calculate x^1, x^2, ..., x^degree

        # Perform the polynomial regression: f(x, w) = w0 + w1*x + w2*x^2 + ... + wd*x^d
        y_pred = torch.matmul(x_powers, self.weights.unsqueeze(1)) # Matrix multiplication

        return y_pred


# Example usage:
degree = 3  # Example polynomial degree
model = PolyRegression(degree)

# Example input
x = torch.tensor([[2.0], [3.0], [4.0]]) # input must be a tensor

# Forward pass
y_pred = model(x)
print(y_pred)

# Accessing weights:
print("Weights:", model.weights)
