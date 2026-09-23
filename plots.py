import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


class Plotter:
    def __init__(self, configs, X_test, y_test, pos_label=1):
        self.configs = configs
        self.X_test = X_test
        self.y_test = y_test
        self.pos_label = pos_label

    def plot_pr_roc(self, experiment_name):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        for cfg in self.configs:
            PrecisionRecallDisplay.from_estimator(
                cfg.estimator, self.X_test, self.y_test,
                pos_label=self.pos_label, ax=axs[0], name=cfg.name,
            )
            RocCurveDisplay.from_estimator(
                cfg.estimator, self.X_test, self.y_test,
                pos_label=self.pos_label, ax=axs[1], name=cfg.name,
                plot_chance_level=(cfg is self.configs[-1]),
            )

        axs[0].set_title("Precision-Recall curve")
        axs[1].set_title("ROC curve")
        fig.suptitle(experiment_name)

        plt.tight_layout()
        plt.savefig(f"report/img/{experiment_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)