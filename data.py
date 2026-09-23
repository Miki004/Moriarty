import sklearn
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from models import MODELS, PARAM_GRID
from configs import Config, save_configs
from metrics import METRIC_FNS


class Data:
    def __init__(self):
        dataset = fetch_ucirepo(id=601)
        self.X = dataset.data.features
        self.y = dataset.data.targets["Machine failure"]

    def _preprocess(self):
        mapping = {'M': 0, 'L': 1, 'H': 2}
        self.X = self.X.replace(mapping)

    def split_data(self, test_size=0.2, random_state=42):
        self._preprocess()
        return sklearn.model_selection.train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )


def baseline_search(X_train, y_train, X_test, y_test):
    """No imbalance handling: plain GridSearchCV optimized for F1."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    configs = []

    for model_name, model in MODELS.items():
        print(f"Baseline search for model: {model_name}")
        grid_search = GridSearchCV(
            model, PARAM_GRID[model_name], cv=cv, scoring="f1", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        cfg = Config(
            name=f"{model_name}_baseline",
            model_name=model_name,
            strategy="baseline",
            estimator=grid_search.best_estimator_,
            best_params=grid_search.best_params_,
        )
        cfg.score_test(X_test, y_test, METRIC_FNS)
        configs.append(cfg)

    save_configs(configs)
    return configs
