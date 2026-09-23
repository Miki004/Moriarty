import argparse

from data import Data, baseline_search
from imbalanceLearning import oversampling_search, smote_search
from SensitiveLearning import cost_sensitive_search, threshold_tuning
from configs import load_configs
from plots import Plotter
from models import MODELS


def main():
    parser = argparse.ArgumentParser(description="Moriarty: A pipeline for maintenance detection")

    parser.add_argument("--tune", action="store_true", help="Run baseline hyperparameter search")
    parser.add_argument("--imb", action="store_true", help="Run the imbalance (oversampling + SMOTE) experiments")
    parser.add_argument("--cost", action="store_true", help="Run the cost-sensitive experiment")
    parser.add_argument("--threshold", action="store_true", help="Run post-hoc threshold tuning")
    parser.add_argument(
        "--plot", metavar="STRATEGY",
        help="Plot PR/ROC curves for a saved experiment's configs " "(e.g. baseline, oversampling, smote, cost_sensitive, threshold_tuned)",
    )

    args = parser.parse_args()

    data = Data()
    X_train, X_test, y_train, y_test = data.split_data()

    if args.tune:
        baseline_search(X_train, y_train, X_test, y_test)
    if args.imb:
        oversampling_search(X_train, y_train, X_test, y_test)
        smote_search(X_train, y_train, X_test, y_test)
    if args.cost:
        cost_sensitive_search(X_train, y_train, X_test, y_test)
    if args.threshold:
        threshold_tuning(X_train, y_train, X_test, y_test)
    if args.plot:
        names = [f"{m}_{args.plot}" for m in MODELS if m != "DummyClassifier"]
        configs = load_configs(names)
        Plotter(configs, X_test, y_test).plot_pr_roc(experiment_name=args.plot)


if __name__ == "__main__":
    main()