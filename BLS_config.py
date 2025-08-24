from dataclasses import dataclass


@dataclass
class BLSConfig:
    n_feature_groups: int = 10
    feature_group_size: int = 10
    n_enhancement_groups: int = 10
    enhancement_group_size: int = 10
    feature_activation: str = "tanh"  # identity, tanh, sigmoid, relu
    enhancement_activation: str = "tanh"  # identity, tanh, sigmoid, relu
    lambda_reg: float = 1e-2
    add_bias: bool = True
    standardize: bool = True
    random_state: int | None = 42
