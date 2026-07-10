import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

training_pred_path = r"training_predictions.csv"
testing_pred_path = r"testing_predictions.csv"
external_pred_path = r"E:\Jupter_workplace\plots\external_validation_randomforest\external_RandomForest_predictions.csv"

output_dir = r"E:\Jupter_workplace\plots\risk_group_validation"
os.makedirs(output_dir, exist_ok=True)

target_col = "y_true"

training_prob_col = "RandomForest_prob"
testing_prob_col = "RandomForest_prob"
external_prob_col = "RandomForest_pred_prob"

def load_prediction_data(path, cohort_name, target_col, prob_col):
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"{cohort_name} : {target_col}")

    if prob_col not in df.columns:
        raise ValueError(f"{cohort_name}: {prob_col}")

    tmp = df[[target_col, prob_col]].copy()
    tmp = tmp.dropna(subset=[target_col, prob_col])

    tmp[target_col] = tmp[target_col].astype(int)
    tmp[prob_col] = tmp[prob_col].astype(float)
    tmp["cohort"] = cohort_name

    return tmp

def assign_risk_group(prob, low_cutoff, intermediate_cutoff):
    if prob <= low_cutoff:
        return "Low risk"
    elif prob <= intermediate_cutoff:
        return "Intermediate risk"
    else:
        return "High risk"

def evaluate_binary_classification(y_true, y_pred_binary):
    y_true = np.asarray(y_true).astype(int)
    y_pred_binary = np.asarray(y_pred_binary).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred_binary,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "precision_ppv": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
        "npv": float(npv),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

train_df = load_prediction_data(
    training_pred_path,
    cohort_name="Training",
    target_col=target_col,
    prob_col=training_prob_col,
)

test_df = load_prediction_data(
    testing_pred_path,
    cohort_name="Testing",
    target_col=target_col,
    prob_col=testing_prob_col,
)

external_df = load_prediction_data(
    external_pred_path,
    cohort_name="External validation",
    target_col=target_col,
    prob_col=external_prob_col,
)

train_tmp = train_df.copy()

train_tmp["risk_decile"] = pd.qcut(
    train_tmp[training_prob_col].rank(method="first"),
    q=10,
    labels=np.arange(1, 11),
)

train_tmp["risk_decile"] = train_tmp["risk_decile"].astype(int)

low_cutoff = train_tmp.loc[
    train_tmp["risk_decile"] <= 5,
    training_prob_col,
].max()

intermediate_cutoff = train_tmp.loc[
    train_tmp["risk_decile"] <= 8,
    training_prob_col,
].max()

print("\nTraining-derived probability cutoffs:")
print(f"Low risk: probability <= {low_cutoff:.6f}")
print(f"Intermediate risk: {low_cutoff:.6f} < probability <= {intermediate_cutoff:.6f}")
print(f"High risk: probability > {intermediate_cutoff:.6f}")

def add_risk_group(df, prob_col):
    tmp = df.copy()

    tmp["risk_group"] = tmp[prob_col].apply(
        lambda x: assign_risk_group(
            x,
            low_cutoff=low_cutoff,
            intermediate_cutoff=intermediate_cutoff,
        )
    )

    risk_score_map = {
        "Low risk": 1,
        "Intermediate risk": 2,
        "High risk": 3,
    }

    tmp["risk_score"] = tmp["risk_group"].map(risk_score_map).astype(int)

    tmp["high_risk_binary"] = (tmp["risk_group"] == "High risk").astype(int)
    tmp["intermediate_high_binary"] = (
        tmp["risk_group"].isin(["Intermediate risk", "High risk"])
    ).astype(int)

    return tmp

train_grouped = add_risk_group(train_df, training_prob_col)
test_grouped = add_risk_group(test_df, testing_prob_col)
external_grouped = add_risk_group(external_df, external_prob_col)

all_grouped = pd.concat(
    [train_grouped, test_grouped, external_grouped],
    axis=0,
    ignore_index=True,
)

grouped_path = os.path.join(
    output_dir,
    "RandomForest_training_cutoff_all_cohorts_risk_groups.csv",
)
all_grouped.to_csv(grouped_path, index=False, encoding="utf-8-sig")

print(f"\nRisk-group assigned data saved to: {grouped_path}")

cohort_data = {
    "Training": train_grouped,
    "Testing": test_grouped,
    "External validation": external_grouped,
}

colors = {
    "Training": "#7B3294",
    "Testing": "#008837",
    "External validation": "#E66101",
}

plt.figure(figsize=(6.4, 6.0), dpi=300)

roc_summary = []

for cohort_name, df in cohort_data.items():
    y_true = df[target_col].astype(int)
    risk_score = df["risk_score"].astype(float)

    fpr, tpr, _ = roc_curve(y_true, risk_score)
    auc_value = roc_auc_score(y_true, risk_score)

    roc_summary.append({
        "cohort": cohort_name,
        "n": len(df),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "risk_stratification_auc": auc_value,
    })

    plt.plot(
        fpr * 100,
        tpr * 100,
        lw=2.2,
        color=colors[cohort_name],
        label=f"{cohort_name} (AUC = {auc_value:.3f})",
    )

plt.plot(
    [0, 100],
    [0, 100],
    color="gray",
    linestyle="--",
    lw=1.2,
    label="Chance",
)

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves for Risk Stratification", fontsize=14)

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.legend(loc="lower right", fontsize=8.5, frameon=True)
plt.grid(alpha=0.25)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

roc_png_path = os.path.join(
    output_dir,
    "RandomForest_risk_stratification_ROC_by_cohort.png",
)
roc_pdf_path = os.path.join(
    output_dir,
    "RandomForest_risk_stratification_ROC_by_cohort.pdf",
)

plt.savefig(roc_png_path, dpi=300, bbox_inches="tight")
plt.savefig(roc_pdf_path, bbox_inches="tight")
plt.show()

roc_summary_df = pd.DataFrame(roc_summary)

roc_summary_path = os.path.join(
    output_dir,
    "RandomForest_risk_stratification_ROC_AUC_summary.csv",
)
roc_summary_df.to_csv(roc_summary_path, index=False, encoding="utf-8-sig")

print(f"\nROC figure saved to: {roc_png_path}")
print(f"ROC AUC summary saved to: {roc_summary_path}")

risk_order = ["Low risk", "Intermediate risk", "High risk"]
cohort_order = ["Training", "Testing", "External validation"]

risk_group_summary = (
    all_grouped
    .groupby(["cohort", "risk_group"])
    .agg(
        n=(target_col, "size"),
        events=(target_col, "sum"),
        observed_event_rate=(target_col, "mean"),
        mean_risk_score=("risk_score", "mean"),
    )
    .reset_index()
)

risk_group_summary["observed_event_rate_percent"] = (
    risk_group_summary["observed_event_rate"] * 100
)

risk_group_summary["cohort"] = pd.Categorical(
    risk_group_summary["cohort"],
    categories=cohort_order,
    ordered=True,
)

risk_group_summary["risk_group"] = pd.Categorical(
    risk_group_summary["risk_group"],
    categories=risk_order,
    ordered=True,
)

risk_group_summary = (
    risk_group_summary
    .sort_values(["cohort", "risk_group"])
    .reset_index(drop=True)
)

risk_group_summary_path = os.path.join(
    output_dir,
    "RandomForest_risk_group_event_rate_by_cohort.csv",
)
risk_group_summary.to_csv(
    risk_group_summary_path,
    index=False,
    encoding="utf-8-sig",
)

print("\nRisk group event rate summary:")
print(risk_group_summary)

print(f"\nRisk group event rate table saved to: {risk_group_summary_path}")

performance_rows = []

for cohort_name, df in cohort_data.items():
    y_true = df[target_col].astype(int)

    metrics_high = evaluate_binary_classification(
        y_true=y_true,
        y_pred_binary=df["high_risk_binary"],
    )

    metrics_high["cohort"] = cohort_name
    metrics_high["risk_rule"] = "High risk vs Low/Intermediate"
    metrics_high["positive_definition"] = "High risk"
    metrics_high["auc_by_risk_score"] = roc_auc_score(y_true, df["risk_score"])

    performance_rows.append(metrics_high)
    metrics_intermediate_high = evaluate_binary_classification(
        y_true=y_true,
        y_pred_binary=df["intermediate_high_binary"],
    )

    metrics_intermediate_high["cohort"] = cohort_name
    metrics_intermediate_high["risk_rule"] = "Intermediate/High risk vs Low"
    metrics_intermediate_high["positive_definition"] = "Intermediate or High risk"
    metrics_intermediate_high["auc_by_risk_score"] = roc_auc_score(y_true, df["risk_score"])

    performance_rows.append(metrics_intermediate_high)

performance_df = pd.DataFrame(performance_rows)

ordered_cols = [
    "cohort",
    "risk_rule",
    "positive_definition",
    "n",
    "events",
    "event_rate",
    "auc_by_risk_score",
    "accuracy",
    "precision_ppv",
    "recall_sensitivity",
    "specificity",
    "npv",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
]

performance_df = performance_df[ordered_cols]

performance_path = os.path.join(
    output_dir,
    "RandomForest_risk_stratification_binary_performance_by_cohort.csv",
)
performance_df.to_csv(performance_path, index=False, encoding="utf-8-sig")

print("\nRisk stratification binary performance:")
print(performance_df)

print(f"\nBinary performance table saved to: {performance_path}")

cutoff_df = pd.DataFrame([
    {
        "model": "RandomForest",
        "source_cohort": "Training",
        "low_risk_definition": "Training decile 1-5",
        "intermediate_risk_definition": "Training decile 6-8",
        "high_risk_definition": "Training decile 9-10",
        "low_cutoff_probability": low_cutoff,
        "intermediate_cutoff_probability": intermediate_cutoff,
    }
])

cutoff_path = os.path.join(
    output_dir,
    "RandomForest_training_derived_risk_group_cutoffs.csv",
)
cutoff_df.to_csv(cutoff_path, index=False, encoding="utf-8-sig")

print(f"\nCutoff table saved to: {cutoff_path}")
print("\nRisk stratification ROC and performance evaluation completed.")