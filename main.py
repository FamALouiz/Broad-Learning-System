import numpy as np
from BLS import BLSConfig, BroadLearningSystem

X = np.random.randn(1000, 32)   # change n_features as you like
y = (X[:, 0] - 0.8 * X[:, 1] > 0).astype(int)

# split using sklearn
Xtr, Xte, ytr, yte = BroadLearningSystem.split(
    X, y, test_size=0.25, random_state=1, stratify=y)

cfg = BLSConfig(
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

model = BroadLearningSystem(cfg).fit(Xtr, ytr)
print("train acc:", (model.predict(Xtr) == ytr).mean())
print("test  acc:", (model.predict(Xte) == yte).mean())

# expand breadth later
model.add_feature_groups(4)
model.add_enhancement_groups(4)
model.refit_output(Xtr, ytr)
print("test acc after expansion:", (model.predict(Xte) == yte).mean())
