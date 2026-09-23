import copy
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV
from sklearn import set_config
set_config(enable_metadata_routing=True)
from models import MODELS, PARAM_GRID
from configs import Config, save_configs, load_configs
from metrics import METRIC_FNS, gain_scorer, pos_label, neg_label

CLASS_WEIGHT_OPTIONS = [
    {neg_label: 1, pos_label: 3},
    {neg_label: 1, pos_label: 5},
]
SAMPLE_WEIGHT_RATIOS = [3, 5]

MODELS_WITH_CLASS_WEIGHT = {"LogisticRegression", "RandomForestClassifier"}
MODELS_WITH_SAMPLE_WEIGHT = {"GradientBoostingClassifier"}


def build_cost_sensitive_grid(param_grid):
    """Add class_weight to the grid for models that support it natively,
    matching existing nested-pipeline key prefixes where present."""
    new_grid = {}
    for model_name, grid in param_grid.items():
        grid = dict(grid)
        if model_name in MODELS_WITH_CLASS_WEIGHT:
            sample_key = next(iter(grid), None)
            if sample_key and "__" in sample_key:
                prefix = sample_key.rsplit("__", 1)[0]
                grid[f"{prefix}__class_weight"] = CLASS_WEIGHT_OPTIONS
            else:
                grid["class_weight"] = CLASS_WEIGHT_OPTIONS
        new_grid[model_name] = grid
    return new_grid


def compute_sample_weight(y, pos_weight):
    weights = np.ones(len(y))
    weights[np.asarray(y) == pos_label] = pos_weight
    return weights


def _search_standard(model_name, model, grid, X_train, y_train, X_test, y_test, cv):
    """Models where class_weight is a native, gridsearchable param."""
    grid_search = GridSearchCV(copy.deepcopy(model), grid, cv=cv, scoring=gain_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cfg = Config(
        name=f"{model_name}_cost_sensitive",
        model_name=model_name,
        strategy="cost_sensitive",
        estimator=grid_search.best_estimator_,
        best_params=grid_search.best_params_,
    )
    cfg.score_test(X_test, y_test, METRIC_FNS)
    return cfg


def _search_sample_weight(model_name, model, grid, X_train, y_train, X_test, y_test, cv):
    """Models without native class_weight: loop over sample_weight ratios,
    running a full GridSearchCV for each, keep the best."""
    best_cfg, best_score = None, -np.inf

    for ratio in SAMPLE_WEIGHT_RATIOS:
        print(f"  -> trying sample_weight ratio {ratio}:1")
        sample_weight = compute_sample_weight(y_train, ratio)
        grid_search = GridSearchCV(copy.deepcopy(model), grid, cv=cv, scoring=gain_scorer, n_jobs=-1)
        grid_search.fit(X_train, y_train, sample_weight=sample_weight)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_cfg = Config(
                name=f"{model_name}_cost_sensitive",
                model_name=model_name,
                strategy="cost_sensitive",
                estimator=grid_search.best_estimator_,
                best_params={**grid_search.best_params_, "sample_weight_ratio": ratio},
                fit_params={"sample_weight": compute_sample_weight(y_train, ratio)},
            )

    best_cfg.score_test(X_test, y_test, METRIC_FNS)
    return best_cfg


def cost_sensitive_search(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid = build_cost_sensitive_grid(PARAM_GRID)
    configs = []

    for model_name, model in MODELS.items():
        if model_name == "DummyClassifier":
            continue
        print(f"Cost-sensitive search for model: {model_name}")

        if model_name in MODELS_WITH_SAMPLE_WEIGHT:
            cfg = _search_sample_weight(model_name, model, grid[model_name], X_train, y_train, X_test, y_test, cv)
        else:
            cfg = _search_standard(model_name, model, grid[model_name], X_train, y_train, X_test, y_test, cv)

        configs.append(cfg)

    save_configs(configs)
    return configs


def threshold_tuning(X_train, y_train, X_test, y_test):
    """Post-hoc decision threshold tuning on top of the cost-sensitive
    models already found by cost_sensitive_search."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    base_names = [f"{m}_cost_sensitive" for m in MODELS if m != "DummyClassifier"]
    base_configs = load_configs(base_names)

    tuned_configs = []
    for cfg in base_configs:
        if cfg.fit_params and "sample_weight" in cfg.fit_params:
            if hasattr(cfg.estimator, "set_fit_request"):
                cfg.estimator.set_fit_request(sample_weight=True)
        print(f"Tuning threshold for {cfg.model_name}")
        tuned_estimator = TunedThresholdClassifierCV(
            copy.deepcopy(cfg.estimator), cv=cv, scoring=gain_scorer, n_jobs=-1
        )
        tuned_estimator.fit(X_train, y_train, **cfg.fit_params)

        tuned_cfg = Config(
            name=f"{cfg.model_name}_threshold_tuned",
            model_name=cfg.model_name,
            strategy="threshold_tuned",
            estimator=tuned_estimator,
            best_params={**cfg.best_params, "threshold": tuned_estimator.best_threshold_},
            fit_params=cfg.fit_params,
        )
        tuned_cfg.score_test(X_test, y_test, METRIC_FNS)
        tuned_configs.append(tuned_cfg)

        print(f"  soglia ottimale: {tuned_estimator.best_threshold_:.3f}")
        print(f"  gain: {cfg.test_scores.get('gain')} -> {tuned_cfg.test_scores.get('gain')}")

    save_configs(tuned_configs)
    return tuned_configs