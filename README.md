# Broad Learning System (BLS)

A Python implementation of the Broad Learning System, a novel neural network architecture that provides an efficient alternative to deep learning for various machine learning tasks. This implementation leverages scikit-learn for robust preprocessing and regularized learning.

## Overview

The Broad Learning System is a flat network architecture that uses random feature mapping and enhancement nodes to create a broad structure instead of a deep one. This approach offers several advantages:

-   **Fast Training**: Uses closed-form solutions via Ridge regression
-   **Incremental Learning**: Support for adding feature and enhancement groups dynamically
-   **Versatile**: Handles both classification and regression tasks
-   **Efficient**: No need for complex backpropagation algorithms

## Features

-   🚀 **Quick Setup**: Easy-to-use configuration system
-   🔧 **Flexible Architecture**: Configurable feature and enhancement groups
-   📊 **Built-in Preprocessing**: StandardScaler and OneHotEncoder integration
-   🎯 **Multiple Activations**: Support for identity, tanh, sigmoid, and ReLU activations
-   📈 **Incremental Learning**: Add groups and refit without full retraining
-   🔀 **Train/Test Split**: Built-in data splitting utilities

## Installation

### Requirements

-   Python 3.8+
-   NumPy
-   scikit-learn
-   pandas (optional)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/FamALouiz/Broad-Learning-System.git
cd Broad-Learning-System
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from BLS import BLSConfig, BroadLearningSystem

# Generate sample data
X = np.random.randn(1000, 32)
y = (X[:, 0] - 0.8 * X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = BroadLearningSystem.split(
    X, y, test_size=0.25, random_state=1, stratify=y
)

# Configure the model
config = BLSConfig(
    n_feature_groups=16,
    feature_group_size=12,
    n_enhancement_groups=8,
    enhancement_group_size=12,
    feature_activation="tanh",
    enhancement_activation="tanh",
    lambda_reg=1e-2,
    standardize=True,
    random_state=1
)

# Train and evaluate
model = BroadLearningSystem(config).fit(X_train, y_train)
train_acc = (model.predict(X_train) == y_train).mean()
test_acc = (model.predict(X_test) == y_test).mean()

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

## Configuration

The `BLSConfig` class provides comprehensive configuration options:

| Parameter                | Type     | Default | Description                               |
| ------------------------ | -------- | ------- | ----------------------------------------- |
| `n_feature_groups`       | int      | 10      | Number of feature mapping groups          |
| `feature_group_size`     | int      | 10      | Size of each feature group                |
| `n_enhancement_groups`   | int      | 10      | Number of enhancement groups              |
| `enhancement_group_size` | int      | 10      | Size of each enhancement group            |
| `feature_activation`     | str      | "tanh"  | Activation function for feature nodes     |
| `enhancement_activation` | str      | "tanh"  | Activation function for enhancement nodes |
| `lambda_reg`             | float    | 1e-2    | Ridge regression regularization parameter |
| `add_bias`               | bool     | True    | Whether to add bias terms                 |
| `standardize`            | bool     | True    | Whether to standardize input features     |
| `random_state`           | int/None | 42      | Random seed for reproducibility           |

### Supported Activation Functions

-   `"identity"`: Linear activation (f(x) = x)
-   `"tanh"`: Hyperbolic tangent
-   `"sigmoid"`: Sigmoid function
-   `"relu"`: Rectified Linear Unit

## Advanced Usage

### Incremental Learning

The BLS supports dynamic expansion of the network architecture:

```python
# Train initial model
model = BroadLearningSystem(config).fit(X_train, y_train)

# Expand the network
model.add_feature_groups(4)      # Add 4 more feature groups
model.add_enhancement_groups(4)  # Add 4 more enhancement groups

# Refit only the output weights (fast)
model.refit_output(X_train, y_train)

# Evaluate expanded model
expanded_acc = (model.predict(X_test) == y_test).mean()
print(f"Accuracy after expansion: {expanded_acc:.4f}")
```

### Classification with Probabilities

For classification tasks, you can get prediction probabilities:

```python
# Get class probabilities
probabilities = model.predict_proba(X_test)
print(f"Prediction probabilities shape: {probabilities.shape}")
```

### Regression Tasks

The BLS automatically detects regression tasks when the target is continuous:

```python
# Regression example
y_regression = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, X.shape[0])
model_reg = BroadLearningSystem(config).fit(X_train, y_regression[:len(X_train)])
predictions = model_reg.predict(X_test)
```

## Project Structure

```
Broad-Learning-System/
├── BLS.py              # Main BroadLearningSystem implementation
├── BLS_config.py       # Configuration dataclass
├── main.py             # Example usage script
├── requirements.txt    # Dependencies (currently empty)
├── LICENSE             # MIT License
└── README.md           # This file
```

## Algorithm Details

The Broad Learning System works in three main steps:

1. **Feature Mapping**: Input data is mapped through random feature groups with configurable activation functions
2. **Enhancement**: Feature outputs are further processed through enhancement groups
3. **Output Learning**: A Ridge regression solver computes optimal output weights using a closed-form solution

The network structure can be represented as:

```
Input → Feature Groups → Enhancement Groups → Output Layer
   ↓           ↓              ↓               ↓
   X    →     [F₁, F₂, ...]  →  [E₁, E₂, ...] → Y
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

-   Chen, C. L. P., & Liu, Z. (2017). Broad learning system: An effective and efficient incremental learning system without the need for deep architecture. IEEE transactions on neural networks and learning systems, 29(1), 10-24.

## Acknowledgments

This implementation uses scikit-learn for robust preprocessing and regularized learning, ensuring compatibility with the broader Python machine learning ecosystem.
