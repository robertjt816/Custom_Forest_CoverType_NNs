""" This contains my activation functions """

import numpy as np

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)

def heaviside_step(x):
    """Heaviside Step activation function."""
    return np.where(x >= 0, 1, 0)

def heaviside_step_derivative(x):
    """Derivative of the Heaviside function."""
    return np.where(x > 0, 0, 0)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of the tanh function."""
    return 1 - np.tanh(x)**2

def prelu(x,alpha=0.1):
    """PReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def prelu_derivative(x, alpha=0.1):
    """Derivative of PReLU function."""
    return np.where(x > 0, 1, alpha)

def leaky_relu(x):
    """Leaky ReLU activation function."""
    return prelu(x,0.01)

def leaky_relu_derivative(x):
    """Derivative of Leaky ReLU function."""
    return np.where(x > 0, 1, 0.01)

def parametrised_sigmoid(x,alpha=1):
    """Parametrised sigmoid function."""
    return 1 / (1 + np.exp(-x/alpha))
    
def parametrised_sigmoid_derivative(x,alpha=1):
    """Derivative of Parametrised sigmoid function."""
    return (1/alpha) * ((parametrised_sigmoid(x,alpha))**2) * np.exp(-x/alpha)
    
def softsign(x):
    """Softsign function."""
    return x / (1+np.abs(x))

def silu(x):
    """
    Sigmoid Linear Unit (SiLU) = x * sigmoid(x)
    Also known as Swish.
    """
    return x * sigmoid(x)

def silu_derivative(x):
    """
    Derivative of SiLU.
    dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """
    s = sigmoid(x)
    return s * (1 + x * (1 - s))

def softplus(x):
    """
    Softplus activation function.
    y = ln(1 + exp(x))
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def softplus_derivative(x):
    """
    Derivative of Softplus = sigmoid(x)
    """
    return sigmoid(x)

def selu(x, lam=1.0507, alpha=1.67326):
    """
    Scaled Exponential Linear Unit (SELU).
    Default parameters ensure self-normalising networks:
    λ=1.0507, α=1.67326
    """
    return lam * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_derivative(x, lam=1.0507, alpha=1.67326):
    """
    Derivative of SELU.
    """
    return lam * np.where(x > 0, 1, alpha * np.exp(x))

def elish(x):
    """
    Exponential Linear Sigmoid Squashing (ELiSH) activation.
    Combines ELU and SiLU.
    """
    return np.where(x >= 0,
                    x / (1 + np.exp(-x)),              # x * sigmoid(x)
                    (np.exp(x) - 1) / (1 + np.exp(-x)))  # ELU * sigmoid(x)

def elish_derivative(x):
    """
    Derivative of ELiSH.
    """
    s = sigmoid(x)
    e = np.exp(x)
    pos = s + x * s * (1 - s)
    neg = e * s + ((e - 1) * s * (1 - s))
    return np.where(x >= 0, pos, neg)

def soboleva_tanh(x):
    """
    Soboleva Modified Hyperbolic Tangent.
    Modification of tanh with cubic and quintic terms.
    Reference: Soboleva et al., 2019.
    """
    return np.tanh(x) + 0.01 * x**3 + 0.001 * x**5

def soboleva_tanh_derivative(x):
    """
    Derivative of Soboleva Modified Hyperbolic Tangent.
    """
    return (1 - np.tanh(x)**2) + 0.03 * x**2 + 0.005 * x**4

def gelu(x):
    """
    Gaussian Error Linear Unit (GeLU).
    Approximation using tanh.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """
    Derivative of GeLU (approximation).
    """
    tanh_term = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
    sech2 = 1 - tanh_term**2
    term1 = 0.5 * (1 + tanh_term)
    term2 = 0.5 * x * sech2 * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
    return term1 + term2