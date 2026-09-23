from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

from models import MODELS, PARAM_GRID
from configs import Config, save_configs
from metrics import METRIC_FNS


def prefix_param_grid(param_grid, prefix="model__"):
    """
    Prefix every grid key with the outer pipeline step name ('model').
    Works for both flat estimators ('n_estimators' -> 'model__n_estimators')
    and models that are themselves a Pipeline, like LogisticRegression
    ('clf__C' -> 'model__clf__C', reaching through both nesting levels).
    """
    return {
        model_name: {f"{prefix}{k}": v for k, v in grid.items()}
        for model_name, grid in param_grid.items()
    }


def _search_resampling(strategy_name, sampler_factory, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = prefix_param_grid(PARAM_GRID)
    configs = []

    for model_name, model in MODELS.items():
        if model_name == "DummyClassifier":
            continue  # resampling a trivial baseline adds nothing meaningful

        print(f"{strategy_name} search for model: {model_name}")
        pipeline = Pipeline([
            ("sampler", sampler_factory()),
            ("model", model),
        ])
        grid_search = GridSearchCV(
            pipeline, grid[model_name], cv=cv, scoring="f1", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        cfg = Config(
            name=f"{model_name}_{strategy_name}",
            model_name=model_name,
            strategy=strategy_name,
            estimator=grid_search.best_estimator_,
            best_params=grid_search.best_params_,
        )
        cfg.score_test(X_test, y_test, METRIC_FNS)
        configs.append(cfg)

    save_configs(configs)
    return configs


def oversampling_search(X_train, y_train, X_test, y_test):
    return _search_resampling(
        "oversampling", lambda: RandomOverSampler(random_state=42),
        X_train, y_train, X_test, y_test
    )


def smote_search(X_train, y_train, X_test, y_test):
    return _search_resampling(
        "smote", lambda: SMOTE(random_state=42),
        X_train, y_train, X_test, y_test
    )