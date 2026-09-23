# Moriarty: Imbalance-Handling Strategies for Predictive Maintenance

This repository contains the code, experiments, and analysis for evaluating machine learning classifiers on highly imbalanced predictive maintenance data. It serves as the companion codebase to the research paper detailed in **MLreport.pdf** ("A Synthesis of Classifiers Under Unbalanced Approaches for Rare Failures").

## Project Overview

Predictive maintenance systems aim to detect rare equipment failures from continuous sensor streams. Because failures are inherently rare, classifiers often struggle with extreme class imbalance. This project investigates the AI4I 2020 dataset (10,000 records, 3.4% failure rate) and evaluates four experimental conditions across Logistic Regression, Random Forest, and Gradient Boosting:

1. **Vanilla (Baseline):** No imbalance handling.
2. **Resampling:** Random oversampling and SMOTE.
3. **Cost-Sensitive Learning:** Tuning directly against an explicit cost matrix (penalizing missed failures more heavily than false alarms).
4. **Post-Hoc Threshold Tuning:** Calibrating decision thresholds after training.

### Key Findings
* **Ensembles Excel Unaided:** Random Forest and Gradient Boosting achieve strong performance (F1 ~0.79) with no imbalance handling at all. Resampling techniques (oversampling/SMOTE) actually degrade their performance.
* **Cost-Sensitive Triumphs:** Explicitly weighting classes based on business costs (e.g., missed failures cost 5x more than false alarms) roughly triples the expected business gain compared to a trivial baseline. 
* **Redundancy in Threshold Tuning:** Post-hoc threshold tuning on top of cost-sensitive models offers no statistically significant benefit, acting as a redundant intervention.

For an in-depth statistical analysis and full methodology, please refer to **MLreport.pdf**.

## Dataset

This project utilizes the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository (ID 601). The dataset contains tool wear, torque, rotational speed, and temperature features against a binary machine failure target spanning five underlying failure modes.

## Installation

Ensure you have Python 3.8+ installed along with the necessary dependencies:
Core dependencies typically include: scikit-learn, imbalanced-learn, pandas, numpy, matplotlib

## Usage

All experiments and plotting capabilities are orchestrated through the **cli.py** script. 

### Running Experiments

Use the following flags with `cli.py` to reproduce the training pipelines and hyperparameter searches:

* **Baseline Search (Vanilla):**
  ```bash
  python cli.py --tune
  ```
* **Imbalance Handling (Oversampling & SMOTE):**
  ```bash
  python cli.py --imb
  ```
* **Cost-Sensitive Learning:**
  ```bash
  python cli.py --cost
  ```
* **Post-hoc Threshold Tuning:**
  ```bash
  python cli.py --threshold
  ```

### Plotting Results

To generate Precision-Recall and ROC curves for a specific experiment strategy, use the `--plot` argument. Valid strategy names include `baseline`, `oversampling`, `smote`, `cost_sensitive`, and `threshold_tuned`.

```bash
python cli.py --plot baseline
python cli.py --plot smote
python cli.py --plot cost_sensitive
```

## Repository Structure

* `cli.py`: The main command-line interface for running experiments and generating plots.
* `MLreport.pdf`: The complete academic report detailing the methodology, statistical tests (Friedman and Wilcoxon), and conclusions.
* `data.py`: Handles loading, preprocessing, and splitting the AI4I dataset, as well as the baseline search.
* `imbalanceLearning.py`: Contains the pipelines for random oversampling and SMOTE.
* `SensitiveLearning.py`: Implements cost-sensitive weighting and threshold tuning optimizations.
* `plots.py`: Generates the comparative PR and ROC curves.
* `models.py`: Defines the model estimators (Logistic Regression, Random Forest, Gradient Boosting).
* `configs.py`: Handles saving and loading trained configurations via `joblib` for reproducibility.

## License & Citation
If you use this code or the findings in **MLreport.pdf**, please refer to the author details in the report for citation instructions.