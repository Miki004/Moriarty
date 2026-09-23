import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

MODELS = {
    'DummyClassifier': sklearn.dummy.DummyClassifier(strategy='constant', constant=0),
    'LogisticRegression': Pipeline([
        ("scaler", StandardScaler()),
        ("clf", sklearn.linear_model.LogisticRegression(max_iter=5000, solver='saga')),
    ]),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
}

PARAM_GRID = {
    'DummyClassifier': {},

    'LogisticRegression': {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    },

    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [None, 10, 20, 30],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', None],
        'min_samples_leaf': [2, 5],
    },

    'GradientBoostingClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
    },
}