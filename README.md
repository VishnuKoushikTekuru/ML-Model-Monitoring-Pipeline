# ML Model Monitoring Pipeline

An end-to-end machine learning monitoring pipeline built with MLflow and Evidently, demonstrating how to detect data drift, track model performance degradation over time, and trigger automated retraining alerts in a production-style workflow.

---

## What this project does

Once a machine learning model is deployed, its performance can quietly degrade as the real-world data it sees shifts away from the data it was trained on. This is called **data drift**, and it is one of the most common and least-visible causes of model failure in production.

This pipeline addresses that problem end-to-end:

1. Trains a baseline model and logs it with MLflow
2. Simulates three months of incoming data with increasing drift
3. Generates an Evidently data drift report comparing reference vs incoming data
4. Tracks model accuracy and AUC-ROC across batches using MLflow experiment tracking
5. Visualises performance degradation over time
6. Fires retrain alerts when performance drops below a defined threshold

---

## Results

**Baseline model (trained on clean data)**
```
Accuracy : 0.9649
AUC-ROC  : 0.9953
```

**Performance across simulated monthly batches**

| Batch   | Accuracy | AUC-ROC | Drift Level |
|---------|----------|---------|-------------|
| Month 1 | 0.9561   | 0.9805  | None        |
| Month 2 | 0.9035   | 0.9858  | Low (0.1)   |
| Month 3 | 0.6053   | 0.9912  | High (0.3)  |

**Key observation:** Accuracy collapses from 96% to 60% under high drift, while AUC-ROC remains artificially high because the model's ranking is preserved even as its absolute predictions shift. This is an important real-world insight — AUC-ROC alone is not sufficient for monitoring under covariate shift. Accuracy and calibration metrics matter too.

---

## Project structure

```
ml-model-monitoring/
├── MLflow.ipynb                  # Main notebook
├── reports/
│   └── data_drift_report.html    # Evidently drift report (auto-generated)
├── plots/
│   └── performance_monitoring.png # Accuracy and AUC-ROC over time
└── README.md
```

---

## Pipeline walkthrough

### Section 1 — Train and log baseline model

Trains a Random Forest classifier on the Wisconsin Breast Cancer dataset (569 samples, 30 features) and logs everything to MLflow: hyperparameters, performance metrics, and the serialised model artifact.

```python
mlflow.set_experiment("clinical-outcome-monitoring")
with mlflow.start_run(run_name="baseline_model"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc_roc", auc)
    mlflow.sklearn.log_model(rf, "model")
```

MLflow stores each run in a local `mlruns/` directory. Run `mlflow ui` to open the experiment tracking dashboard in your browser.

### Section 2 — Simulate data drift

Creates three batches of incoming data by adding Gaussian noise with increasing mean shift to the test set, simulating the kind of covariate shift that happens in production over time as data distributions change.

```
Batch 1 mean shift: 0.0026  (effectively no drift)
Batch 2 mean shift: 0.0960  (low drift)
Batch 3 mean shift: 0.3005  (high drift)
```

### Section 3 — Evidently drift report

Runs an automated statistical drift analysis comparing the training data (reference) against the most-drifted incoming batch. Evidently tests each feature individually using appropriate statistical tests (KS test for continuous features, chi-squared for categorical) and produces a full HTML report.

```python
report = Report([DataDriftPreset()])
result = report.run(reference_data=X_train, current_data=batches[2])
result.save_html("reports/data_drift_report.html")
```

Open `reports/data_drift_report.html` in a browser to see feature-by-feature drift analysis with distribution plots.

### Section 4 — Track performance across batches

Evaluates the original model on each incoming batch and logs results to MLflow under separate monitoring runs. This creates a time-series record of model performance that can be queried and compared in the MLflow UI.

### Section 5 — Visualise degradation

Plots accuracy and AUC-ROC across batches with a red threshold line at 0.90, making degradation immediately visible. Saved to `plots/performance_monitoring.png`.

### Section 6 — Retrain trigger logic

Checks each batch's AUC-ROC against a defined threshold and fires a human-readable alert if performance has dropped below acceptable levels.

```python
RETRAIN_THRESHOLD = 0.90
for _, row in perf_df.iterrows():
    if row['auc_roc'] < RETRAIN_THRESHOLD:
        print(f"ALERT {row['batch']}: AUC below threshold — retrain recommended")
```

In a real production system, this trigger would call a retraining job, open a ticket, or send a Slack alert. Here it serves as a clear demonstration of the decision logic.

---

## Dataset

**Wisconsin Breast Cancer Dataset** (sklearn built-in)
- 569 samples, 30 numeric features
- Binary classification: malignant vs benign
- Used here as a clinical outcome proxy — directly relevant to health data science applications

---

## Tools and libraries

| Tool | Role |
|---|---|
| MLflow | Experiment tracking, parameter and metric logging, model registry |
| Evidently | Statistical data drift detection, automated HTML reporting |
| Scikit-learn | Model training, evaluation metrics |
| Pandas / NumPy | Data manipulation and drift simulation |
| Matplotlib | Performance visualisation |

---

## How to run

```bash
pip install mlflow scikit-learn pandas numpy matplotlib evidently
python MLflow.ipynb   # or run cells in Jupyter
```

To open the MLflow experiment dashboard after running:
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

**Note on Evidently version:** This project uses Evidently v0.7+. The import path changed from earlier versions — use `from evidently import Report` and `from evidently.presets import DataDriftPreset`.

---

## Key concepts demonstrated

**Data drift** — when incoming data shifts away from the distribution the model was trained on, predictions become unreliable even if the model itself has not changed.

**Covariate shift** — the specific type of drift shown here, where feature distributions change while the underlying relationship between features and labels remains constant.

**AUC-ROC vs Accuracy under drift** — this project shows that AUC-ROC can remain stable while accuracy collapses under high drift, because AUC-ROC measures ranking ability rather than absolute calibration. Both metrics are needed for complete monitoring.

**Experiment tracking** — MLflow records every run with its parameters, metrics, and model artifacts, making it possible to compare runs, roll back to earlier model versions, and maintain a full audit trail.

---

## Author

**Vishnu Koushik Tekuru**
MSc Health Data Science, University of Liverpool
[github.com/VishnuKoushikTekuru](https://github.com/VishnuKoushikTekuru)
[linkedin.com/in/vishnu-koushik-t-46aa1b12a](https://linkedin.com/in/vishnu-koushik-t-46aa1b12a)
