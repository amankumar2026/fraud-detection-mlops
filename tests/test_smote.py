import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression


def _imbalanced_data(n=500, minority=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.zeros(n, dtype=int)
    y[:minority] = 1
    X[:minority] += 2.0
    return X, y


def test_smote_only_resamples_during_fit():
    X, y = _imbalanced_data()
    pipe = ImbPipeline([("smote", SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=0)),
                        ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)

    X_val = np.random.default_rng(1).normal(size=(37, 5))
    proba = pipe.predict_proba(X_val)
    assert proba.shape == (37, 2)


def test_smote_increases_minority_in_training_only():
    X, y = _imbalanced_data()
    X_res, y_res = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=0).fit_resample(X, y)
    assert (y_res == 1).sum() > (y == 1).sum()
    assert (y_res == 0).sum() == (y == 0).sum()
    assert len(X) == 500  # original data untouched
