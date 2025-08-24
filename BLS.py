import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.extmath import softmax

from BLS_config import BLSConfig

_ACTS = {
    "identity": lambda x: x,
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "relu": lambda x: np.maximum(0.0, x),
}


class BroadLearningSystem:
    """
    Broad Learning System that uses scikit-learn:
      - StandardScaler
      - OneHotEncoder
      - Ridge for closed-form output weights
      - train_test_split helper
    """

    def __init__(self, cfg: BLSConfig):
        """
        Initialize the Broad Learning System.

        Args:
            cfg (BLSConfig): Configuration object containing all hyperparameters
                for the BLS model including feature groups, enhancement groups,
                activation functions, and regularization parameters.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_state)
        self.scaler = StandardScaler(
            with_mean=True, with_std=True) if cfg.standardize else None
        self.enc = None
        self.is_classification = None
        self.classes_ = None

        self.Wf, self.bf = [], []
        self.We, self.be = [], []
        self.Wout = None
        self.ridge = None

        if cfg.feature_activation not in _ACTS or cfg.enhancement_activation not in _ACTS:
            raise ValueError("Unknown activation name.")
        self.act_f = _ACTS[cfg.feature_activation]
        self.act_e = _ACTS[cfg.enhancement_activation]

    @staticmethod
    def split(X, y, test_size=0.2, random_state=0, stratify=None):
        """
        Split arrays or matrices into random train and test subsets.

        This is a convenience wrapper around sklearn's train_test_split.

        Args:
            X (array-like): Features array of shape (n_samples, n_features).
            y (array-like): Target array of shape (n_samples,) or (n_samples, n_outputs).
            test_size (float, optional): Proportion of dataset to include in test split. 
                Defaults to 0.2.
            random_state (int, optional): Random seed for reproducible splits. 
                Defaults to 0.
            stratify (array-like, optional): If not None, data is split in a stratified 
                fashion using this as class labels. Defaults to None.

        Returns:
            tuple: X_train, X_test, y_train, y_test arrays.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    # ---------- internal ----------
    def _init_groups(self, in_dim):
        """
        Initialize feature and enhancement groups with random weights and biases.

        Creates random weight matrices and bias vectors for both feature mapping
        groups and enhancement groups. Feature groups map input to feature space,
        while enhancement groups map feature outputs to enhancement space.

        Args:
            in_dim (int): Input dimension for feature groups.
        """
        self.Wf, self.bf = [], []
        for _ in range(self.cfg.n_feature_groups):
            W = self.rng.normal(0.0, 1.0, size=(
                in_dim, self.cfg.feature_group_size))
            b = self.rng.normal(0.0, 0.1, size=(self.cfg.feature_group_size,))
            self.Wf.append(W)
            self.bf.append(b)

        total_feature_nodes = self.cfg.n_feature_groups * self.cfg.feature_group_size
        self.We, self.be = [], []
        for _ in range(self.cfg.n_enhancement_groups):
            W = self.rng.normal(0.0, 1.0, size=(
                total_feature_nodes, self.cfg.enhancement_group_size))
            b = self.rng.normal(0.0, 0.1, size=(
                self.cfg.enhancement_group_size,))
            self.We.append(W)
            self.be.append(b)

    def _map_features(self, X):
        """
        Map input data through feature groups using random weights and activation functions.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Feature mappings of shape (n_samples, n_feature_groups * feature_group_size).
        """
        feats = [self.act_f(X @ W + b) for W, b in zip(self.Wf, self.bf)]
        return np.concatenate(feats, axis=1) if feats else np.empty((X.shape[0], 0))

    def _map_enhancements(self, F):
        """
        Map feature outputs through enhancement groups using random weights and activation functions.

        Args:
            F (np.ndarray): Feature mappings of shape (n_samples, n_feature_nodes).

        Returns:
            np.ndarray: Enhancement mappings of shape (n_samples, n_enhancement_groups * enhancement_group_size).
        """
        enh = [self.act_e(F @ W + b) for W, b in zip(self.We, self.be)]
        return np.concatenate(enh, axis=1) if enh else np.empty((F.shape[0], 0))

    def _design(self, X):
        """
        Create the design matrix by concatenating feature and enhancement mappings.

        Applies feature mapping, enhancement mapping, and optionally adds bias term
        to create the final design matrix used for output weight learning.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Design matrix of shape (n_samples, total_nodes + bias).
        """
        F = self._map_features(X)
        E = self._map_enhancements(F)
        H = np.concatenate([F, E], axis=1) if E.size else F
        if self.cfg.add_bias:
            H = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
        return H

    def fit(self, X, y):
        """
        Train the Broad Learning System on the given data.

        Fits the model by standardizing input, initializing random groups,
        preparing target encoding (for classification), creating design matrix,
        and solving for optimal output weights using Ridge regression.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Training targets of shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            BroadLearningSystem: Returns self for method chaining.
        """
        X = np.asarray(X, dtype=float)

        if self.scaler:
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = X

        self._init_groups(Xs.shape[1])

        y = np.asarray(y)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            self.is_classification = True
            y = y.reshape(-1, 1)
            self.enc = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore")
            T = self.enc.fit_transform(y)
            self.classes_ = self.enc.categories_[0]
        else:
            self.is_classification = False
            T = y.astype(float)

        H = self._design(Xs)
        self.ridge = Ridge(alpha=self.cfg.lambda_reg,
                           fit_intercept=False, random_state=self.cfg.random_state)
        self.ridge.fit(H, T)
        self.Wout = self.ridge.coef_.T
        return self

    def _forward(self, X):
        """
        Perform forward pass through the trained BLS model.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Model outputs of shape (n_samples, n_outputs).
        """
        if self.scaler:
            X = self.scaler.transform(X)
        H = self._design(X)
        return H @ self.Wout

    def predict(self, X):
        """
        Make predictions on new data.

        For classification tasks, returns predicted class labels.
        For regression tasks, returns predicted continuous values.

        Args:
            X (array-like): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predictions of shape (n_samples,) for classification or 
                       (n_samples, n_outputs) for regression.
        """
        X = np.asarray(X, dtype=float)
        Y = self._forward(X)
        if self.is_classification:
            idx = np.argmax(Y, axis=1)
            return self.classes_[idx]
        return Y

    def predict_proba(self, X):
        """
        Compute class probabilities for input samples.

        Only available for classification tasks. Uses softmax to convert
        logits to probability distributions.

        Args:
            X (array-like): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).

        Raises:
            ValueError: If called on a regression model.
        """
        if not self.is_classification:
            raise ValueError("predict_proba only for classification.")
        X = np.asarray(X, dtype=float)
        logits = self._forward(X)
        return softmax(logits)

    def add_feature_groups(self, k):
        """
        Add additional feature groups to expand the network breadth.

        This method supports incremental learning by adding new feature mapping
        groups without retraining the entire model. The output weights will need
        to be refitted after expansion.

        Args:
            k (int): Number of feature groups to add.

        Raises:
            RuntimeError: If called before the model is fitted.
        """
        in_dim = self.scaler.mean_.shape[0] if self.scaler else None
        if in_dim is None:
            raise RuntimeError("Model must be fitted before adding groups.")
        for _ in range(k):
            W = self.rng.normal(0.0, 1.0, size=(
                in_dim, self.cfg.feature_group_size))
            b = self.rng.normal(0.0, 0.1, size=(self.cfg.feature_group_size,))
            self.Wf.append(W)
            self.bf.append(b)

    def add_enhancement_groups(self, k):
        """
        Add additional enhancement groups to expand the network breadth.

        This method supports incremental learning by adding new enhancement
        groups that operate on the current feature space. The output weights
        will need to be refitted after expansion.

        Args:
            k (int): Number of enhancement groups to add.
        """
        total_feature_nodes = len(self.Wf) * self.cfg.feature_group_size
        for _ in range(k):
            W = self.rng.normal(0.0, 1.0, size=(
                total_feature_nodes, self.cfg.enhancement_group_size))
            b = self.rng.normal(0.0, 0.1, size=(
                self.cfg.enhancement_group_size,))
            self.We.append(W)
            self.be.append(b)

    def refit_output(self, X, y):
        """
        Refit only the output weights using the current network architecture.

        This method is efficient for updating the model after adding new feature
        or enhancement groups, as it only recomputes the final Ridge regression
        without reinitializing the random groups.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Training targets of shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            BroadLearningSystem: Returns self for method chaining.
        """
        X = np.asarray(X, dtype=float)
        if self.scaler:
            X = self.scaler.transform(X)

        y = np.asarray(y)
        if self.is_classification:
            T = self.enc.transform(y.reshape(-1, 1))
        else:
            T = y.astype(float)

        H = self._design(X)
        self.ridge.fit(H, T)
        self.Wout = self.ridge.coef_.T
        return self

    def extract_features(self, X):
        """
        Extract the intermediate feature representation from the BLS network.

        Returns the design matrix (concatenated feature and enhancement mappings)
        without applying the final output weights. Useful for feature extraction
        and as input to other models like LSTMs.

        Args:
            X (array-like): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Feature representation of shape (n_samples, total_nodes + bias).
        """
        X = np.asarray(X, dtype=float)
        if self.scaler:
            X = self.scaler.transform(X)
        return self._design(X)
