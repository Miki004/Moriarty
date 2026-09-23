

import os
import copy
import joblib
import numpy as np
import pandas as pd


class Config:
    def __init__(self, name, model_name, strategy, estimator,best_params=None, fit_params=None):
        self.name = name                  # e.g. "RF_cost_sensitive"
        self.model_name = model_name      # e.g. "RandomForestClassifier"
        self.strategy = strategy          # "baseline" | "oversampling" | "smote" | "cost_sensitive" | "threshold_tuned"
        self.estimator = estimator        # unfitted or fitted sklearn/imblearn estimator
        self.best_params = best_params or {}
        self.fit_params = fit_params or {}  # e.g. {"sample_weight": array}
        self.cv_scores = {}               # metric_name -> np.ndarray (per fold)
        self.test_scores = {}             # metric_name -> float

    def fit(self, X_train, y_train):
        self.estimator = copy.deepcopy(self.estimator)
        self.estimator.fit(X_train, y_train, **self.fit_params)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score_cv(self, X, y, cv, metric_fns):
        per_metric_scores = {name: [] for name in metric_fns}

        for train_idx, test_idx in cv.split(X, y):
            X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_te = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            y_te = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]

            model = copy.deepcopy(self.estimator)
            fold_fit_params = {}
            if "sample_weight" in self.fit_params:
                fold_fit_params["sample_weight"] = np.asarray(self.fit_params["sample_weight"])[train_idx]

            model.fit(X_tr, y_tr, **fold_fit_params)
            y_pred = model.predict(X_te)

            for name, fn in metric_fns.items():
                per_metric_scores[name].append(fn(y_te, y_pred))

        self.cv_scores = {name: np.array(scores) for name, scores in per_metric_scores.items()}
        return self.cv_scores

    def score_test(self, X_test, y_test, metric_fns):
        y_pred = self.predict(X_test)
        self.test_scores = {name: fn(y_test, y_pred) for name, fn in metric_fns.items()}
        return self.test_scores

    def save(self, directory="report/configs"):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, f"{directory}/{self.name}.joblib")

    @staticmethod
    def load(name, directory="report/configs"):
        return joblib.load(f"{directory}/{name}.joblib")

    def __repr__(self):
        return f"Config(name={self.name!r}, strategy={self.strategy!r}, params={self.best_params})"


def save_configs(configs, directory="report/configs"):
    for cfg in configs:
        cfg.save(directory)


def load_configs(names, directory="report/configs"):
    return [Config.load(name, directory) for name in names]


def configs_to_leaderboard(configs, metric_name):
    """Build a leaderboard DataFrame from a list of Config objects,
    reading whichever metric (already scored via score_test) is requested."""
    rows = []
    for cfg in configs:
        rows.append({
            "name": cfg.name,
            "model": cfg.model_name,
            "strategy": cfg.strategy,
            "best_params": cfg.best_params,
            f"test_{metric_name}": cfg.test_scores.get(metric_name),
        })
    return pd.DataFrame(rows).sort_values(f"test_{metric_name}", ascending=False)