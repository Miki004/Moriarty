
import itertools
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn import set_config
set_config(enable_metadata_routing=True)
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from data import Data
from models import MODELS
from configs import load_configs
from metrics import f1, credit_gain_score


def run_friedman_wilcoxon(score_df, label, out_prefix):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(score_df.mean().sort_values(ascending=False))

    stat, p_value = friedmanchisquare(*[score_df[col] for col in score_df.columns])
    print(f"\nFriedman test: statistic={stat:.3f}, p-value={p_value:.5f}")

    if p_value < 0.05:
        print("=> Significant differences found. Running pairwise Wilcoxon tests...")
        pairs = list(itertools.combinations(score_df.columns, 2))
        raw_pvalues = [wilcoxon(score_df[c1], score_df[c2])[1] for c1, c2 in pairs]
        reject, corrected_pvalues, _, _ = multipletests(raw_pvalues, alpha=0.05, method="holm")

        pairwise_results = pd.DataFrame({
            "config_1": [p[0] for p in pairs],
            "config_2": [p[1] for p in pairs],
            "raw_p_value": raw_pvalues,
            "corrected_p_value": corrected_pvalues,
            "significant": reject,
        }).sort_values("corrected_p_value")

        pairwise_results.to_csv(f"report\\pairwise_wilcoxon_{out_prefix}.csv", index=False)
        print(pairwise_results.to_string(index=False))
    else:
        print("=> No significant difference detected (p >= 0.05). Skipping pairwise tests.")

    score_df.to_csv(f"report\\per_fold_scores_{out_prefix}.csv", index=False)


def score_configs(configs, X, y, cv, metric_fn, metric_name):
    scores = {}
    for cfg in configs:
        print(f"Scoring ({metric_name}) configuration: {cfg.name}")
        cfg.score_cv(X, y, cv, {metric_name: metric_fn})
        scores[cfg.name] = cfg.cv_scores[metric_name]
    return pd.DataFrame(scores)


if __name__ == "__main__":
    data = Data()
    X_train, X_test, y_train, y_test = data.split_data()
    cva = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    cvb = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model_names = [m for m in MODELS if m != "DummyClassifier"]


    test_a_names = (
        [f"{m}_baseline" for m in model_names]
        + [f"{m}_oversampling" for m in model_names]
        + [f"{m}_smote" for m in model_names]
    )
    test_a_configs = load_configs(test_a_names)

    score_df_f1 = score_configs(test_a_configs, X_train, y_train, cva, f1, "f1")
    run_friedman_wilcoxon(score_df_f1, "TEST A: F1 across baseline / oversampling / SMOTE", "test_A_f1")


    test_b_names = (
        [f"{m}_cost_sensitive" for m in model_names]
        + [f"{m}_threshold_tuned" for m in model_names]
    )
    test_b_configs = load_configs(test_b_names)

    score_df_gain = score_configs(test_b_configs, X_train, y_train, cvb, credit_gain_score, "gain")
    run_friedman_wilcoxon(score_df_gain, "TEST B: business gain across cost-sensitive / threshold-tuned", "test_B_gain")