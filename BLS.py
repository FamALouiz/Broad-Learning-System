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
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    # ---------- internal ----------
    def _init_groups(self, in_dim):
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
        feats = [self.act_f(X @ W + b) for W, b in zip(self.Wf, self.bf)]
        return np.concatenate(feats, axis=1) if feats else np.empty((X.shape[0], 0))

    def _map_enhancements(self, F):
        enh = [self.act_e(F @ W + b) for W, b in zip(self.We, self.be)]
        return np.concatenate(enh, axis=1) if enh else np.empty((F.shape[0], 0))

    def _design(self, X):
        F = self._map_features(X)
        E = self._map_enhancements(F)
        H = np.concatenate([F, E], axis=1) if E.size else F
        if self.cfg.add_bias:
            H = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)
        return H

    def fit(self, X, y):
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
        if self.scaler:
            X = self.scaler.transform(X)
        H = self._design(X)
        return H @ self.Wout

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Y = self._forward(X)
        if self.is_classification:
            idx = np.argmax(Y, axis=1)
            return self.classes_[idx]
        return Y

    def predict_proba(self, X):
        if not self.is_classification:
            raise ValueError("predict_proba only for classification.")
        X = np.asarray(X, dtype=float)
        logits = self._forward(X)
        return softmax(logits)

    def add_feature_groups(self, k):
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
        total_feature_nodes = len(self.Wf) * self.cfg.feature_group_size
        for _ in range(k):
            W = self.rng.normal(0.0, 1.0, size=(
                total_feature_nodes, self.cfg.enhancement_group_size))
            b = self.rng.normal(0.0, 0.1, size=(
                self.cfg.enhancement_group_size,))
            self.We.append(W)
            self.be.append(b)

    def refit_output(self, X, y):
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
