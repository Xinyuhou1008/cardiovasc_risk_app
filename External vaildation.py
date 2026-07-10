import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.optimize import brentq

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
)

external_path = r"E:\Jupter_workplace\data\external_cohort_imputed.csv"

model_name = "RandomForest"
model_path = r"RandomForest_best_model.pkl"
calibrated_model_path = r"RandomForest_calibrated_model.pkl"
meta_path = r"RandomForest_meta.pkl"

output_dir = r"E:\Jupter_workplace\plots\external_validation_randomforest"
os.makedirs(output_dir, exist_ok=True)

id_col = "id"
target_col = "cdeath"

SEED = 42
N_BOOT = 1000
N_BINS = 5

CURVE_COLORS = {
    "Original model": "#B9141B",
    "Internal testing calibration": "#A9A8A6",
    "External intercept-only update": "#4E93D8",
}

CURVE_LINESTYLES = {
    "Original model": "--",
    "Internal testing calibration": "-",
    "External intercept-only update": "-",
}

CURVE_MARKERS = {
    "Original model": "o",
    "Internal testing calibration": "s",
    "External intercept-only update": "^",
}

plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 34,
    "axes.labelsize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 16,
    "axes.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def read_csv_auto_encoding(path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "ansi"]

    last_error = None
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e

    raise ValueError(f"{last_error}")

def safe_prob(prob, eps=1e-6):
    prob = np.asarray(prob, dtype=float)
    return np.clip(prob, eps, 1.0 - eps)

def logit(prob, eps=1e-6):
    prob = safe_prob(prob, eps=eps)
    return np.log(prob / (1.0 - prob))

def predict_event_probability(model, X):
    if hasattr(model, "classes_") and 1 in list(model.classes_):
        event_index = list(model.classes_).index(1)
        return model.predict_proba(X)[:, event_index]

    return model.predict_proba(X)[:, 1]

def evaluate_at_threshold(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }

def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_boot=1000, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    scores = []

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample_idx]
        yp = y_prob[sample_idx]

        if len(np.unique(yt)) < 2:
            continue

        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            continue

    if len(scores) == 0:
        return np.nan, np.nan

    return (
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )

def bootstrap_brier_ci(y_true, y_prob, n_boot=1000, seed=42):
    return bootstrap_metric_ci(
        y_true=y_true,
        y_prob=y_prob,
        metric_fn=brier_score_loss,
        n_boot=n_boot,
        seed=seed,
    )

def calculate_calibration_intercept(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    lp = logit(y_prob)

    def score(alpha):
        return np.sum(y_true - expit(alpha + lp))

    try:
        alpha = brentq(score, -30.0, 30.0)
    except ValueError:
        observed = np.mean(y_true)
        predicted = np.mean(safe_prob(y_prob))
        alpha = logit(observed) - logit(predicted)

    return float(alpha)

def calculate_calibration_slope(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    lp = logit(y_prob).reshape(-1, 1)

    clf = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=10000,
    )
    clf.fit(lp, y_true)

    slope = float(clf.coef_[0][0])

    return slope

def bootstrap_calibration_intercept_slope_ci(y_true, y_prob, n_boot=1000, seed=42):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))

    intercepts = []
    slopes = []

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample_idx]
        yp = y_prob[sample_idx]

        if len(np.unique(yt)) < 2:
            continue

        try:
            intercept = calculate_calibration_intercept(yt, yp)
            slope = calculate_calibration_slope(yt, yp)

            intercepts.append(intercept)
            slopes.append(slope)
        except Exception:
            continue

    if len(intercepts) == 0:
        return np.nan, np.nan, np.nan, np.nan

    return (
        float(np.percentile(intercepts, 2.5)),
        float(np.percentile(intercepts, 97.5)),
        float(np.percentile(slopes, 2.5)),
        float(np.percentile(slopes, 97.5)),
    )

def external_intercept_only_update(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    lp = logit(y_prob)

    def score(alpha):
        return np.sum(y_true - expit(alpha + lp))

    try:
        alpha = brentq(score, -30.0, 30.0)
    except ValueError:
        observed = np.mean(y_true)
        predicted = np.mean(safe_prob(y_prob))
        alpha = logit(observed) - logit(predicted)

    updated_prob = expit(alpha + lp)

    return updated_prob, float(alpha)

def calibration_bins_from_reference(y_prob, n_bins=5):
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.unique(
        np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    )

    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    edges[0] = 0.0
    edges[-1] = 1.0

    return edges

def calibration_curve_fixed(y_true, y_prob, edges):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_bins = len(edges) - 1
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    mean_pred = np.full(n_bins, np.nan, dtype=float)
    frac_pos = np.full(n_bins, np.nan, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_ids == i
        bin_count[i] = int(np.sum(mask))

        if np.any(mask):
            mean_pred[i] = float(np.mean(y_prob[mask]))
            frac_pos[i] = float(np.mean(y_true[mask]))

    return mean_pred, frac_pos, bin_count

def calculate_calibration_curve_points(y_true, y_prob, n_bins=5):
    edges = calibration_bins_from_reference(y_prob, n_bins=n_bins)
    return calibration_curve_fixed(y_true, y_prob, edges)

def calculate_net_benefit(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n = len(y_true)
    net_benefits = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        net_benefits.append(nb)

    return np.asarray(net_benefits)

def calculate_net_benefit_all(y_true, thresholds):
    y_true = np.asarray(y_true).astype(int)
    prevalence = np.mean(y_true)

    return prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)

def smooth_curve(y, window=7):
    if window is None or window <= 1:
        return y

    return (
        pd.Series(y)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

print("Loading external cohort...")
external_df = read_csv_auto_encoding(external_path)

print("\nExternal data shape:", external_df.shape)
print("External columns:")
print(external_df.columns.tolist())

print("\nLoading RandomForest model and metadata...")

model = joblib.load(model_path)
meta = joblib.load(meta_path)

if not os.path.exists(calibrated_model_path):
    raise FileNotFoundError(
        f"{calibrated_model_path}"
    )

calibrated_model = joblib.load(calibrated_model_path)

selected_features = meta["features"]
threshold_oof = meta["threshold"]

print(f"\nSelected features ({len(selected_features)}):")
print(selected_features)
print(f"\nOOF-optimal threshold: {threshold_oof:.4f}")

if hasattr(model, "classes_"):
    print("\nRaw model classes:", model.classes_)

if hasattr(calibrated_model, "classes_"):
    print("Calibrated model classes:", calibrated_model.classes_)

if id_col not in external_df.columns:
    raise ValueError(f"{id_col}")

has_label = target_col in external_df.columns

if not has_label:
    raise ValueError(
        f"{target_col}。"
    )

missing_features = [col for col in selected_features if col not in external_df.columns]
if missing_features:
    raise ValueError(
        "\n"
        f"{missing_features}"
    )

X_ext = external_df[selected_features].copy()
y_ext = external_df[target_col].copy()

if y_ext.dtype == "object" or y_ext.dtype.name == "category":
    label_encoder = LabelEncoder()
    y_ext = pd.Series(
        label_encoder.fit_transform(y_ext),
        index=external_df.index,
    )

y_ext = y_ext.astype(int)

print("\nExternal label distribution:")
print(y_ext.value_counts(dropna=False).sort_index())
print(f"External event rate: {np.mean(y_ext):.4f}")
print("\nX_ext shape:", X_ext.shape)

print("\nPredicting on external cohort...")

prob_ext_raw = predict_event_probability(model, X_ext)
prob_ext_internal_calibrated = predict_event_probability(calibrated_model, X_ext)

prob_ext_intercept_update, external_update_alpha = external_intercept_only_update(
    y_true=y_ext.values,
    y_prob=prob_ext_raw,
)

print("\nExternal intercept-only update alpha:")
print(f"{external_update_alpha:.6f}")

probability_sets = {
    "Original model": prob_ext_raw,
    "Internal testing calibration": prob_ext_internal_calibrated,
    "External intercept-only update": prob_ext_intercept_update,
}

for label, prob in probability_sets.items():
    print(f"\nPredicted probability summary: {label}")
    print(
        pd.Series(prob).describe(
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        )
    )

pred_df = external_df[[id_col]].copy()
pred_df["y_true"] = y_ext.values
pred_df["RandomForest_prob_raw_no_calibration"] = prob_ext_raw
pred_df["RandomForest_prob_internal_testing_calibration"] = prob_ext_internal_calibrated
pred_df["RandomForest_prob_external_intercept_only_update"] = prob_ext_intercept_update

pred_df[f"RandomForest_pred_oof_threshold_{threshold_oof:.4f}_raw"] = (
    prob_ext_raw >= threshold_oof
).astype(int)
pred_df["RandomForest_pred_threshold_0.5_raw"] = (
    prob_ext_raw >= 0.5
).astype(int)

prediction_path = os.path.join(
    output_dir,
    "external_RandomForest_predictions_with_calibration_versions.csv",
)
pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

print(f"\nPredictions saved to: {prediction_path}")

performance_rows = []

for label, prob in probability_sets.items():
    metrics_oof = evaluate_at_threshold(
        y_true=y_ext,
        y_prob=prob,
        threshold=threshold_oof,
    )

    metrics_05 = evaluate_at_threshold(
        y_true=y_ext,
        y_prob=prob,
        threshold=0.5,
    )

    print(f"\n== External Validation Cohort Performance: {label}, OOF threshold = {threshold_oof:.4f} ==")
    for key, value in metrics_oof.items():
        if key != "y_pred":
            print(
                f"{key:15}: {value:.4f}"
                if isinstance(value, float)
                else f"{key:15}: {value}"
            )

    performance_rows.append({
        "cohort": "external_validation",
        "model": "RandomForest",
        "calibration_version": label,
        "threshold_used": "OOF_best_f1",
        "threshold_value": threshold_oof,
        "n_samples": len(y_ext),
        "event_rate": np.mean(y_ext),
        **{k: v for k, v in metrics_oof.items() if k != "y_pred"},
    })

    performance_rows.append({
        "cohort": "external_validation",
        "model": "RandomForest",
        "calibration_version": label,
        "threshold_used": "fixed_0.5",
        "threshold_value": 0.5,
        "n_samples": len(y_ext),
        "event_rate": np.mean(y_ext),
        **{k: v for k, v in metrics_05.items() if k != "y_pred"},
    })

performance_df = pd.DataFrame(performance_rows)

performance_path = os.path.join(
    output_dir,
    "external_RandomForest_performance_all_calibration_versions.csv",
)
performance_df.to_csv(performance_path, index=False, encoding="utf-8-sig")

print(f"\nPerformance saved to: {performance_path}")

fpr, tpr, _ = roc_curve(y_ext, prob_ext_raw)
auc_ext = roc_auc_score(y_ext, prob_ext_raw)

plt.figure(figsize=(7.2, 6.8), dpi=300)

plt.plot(
    fpr,
    tpr,
    label=f"RandomForest AUC={auc_ext:.3f}",
    lw=3.0,
    color="#B9141B",
)

plt.plot(
    [0, 1],
    [0, 1],
    color="black",
    linestyle="--",
    lw=1.5,
    label="Chance",
)

plt.xlabel("False Positive Rate", fontsize=24)
plt.ylabel("True Positive Rate", fontsize=24)
plt.title("ROC Curve in External Validation Cohort", fontsize=28, pad=18)
plt.tick_params(axis="both", labelsize=20, width=1.8, length=7)
plt.legend(loc="lower right", fontsize=16, frameon=True)
plt.grid(alpha=0.25)

for spine in plt.gca().spines.values():
    spine.set_linewidth(1.8)

plt.tight_layout()

roc_png_path = os.path.join(output_dir, "external_RandomForest_roc.png")
roc_pdf_path = os.path.join(output_dir, "external_RandomForest_roc.pdf")

plt.savefig(roc_png_path, dpi=300, bbox_inches="tight")
plt.savefig(roc_pdf_path, bbox_inches="tight")
plt.show()

print(f"\nROC figure saved to: {roc_png_path}")
print(f"ROC figure saved to: {roc_pdf_path}")

calibration_metric_rows = []
calibration_curve_rows = {}

for label, prob in probability_sets.items():
    intercept = calculate_calibration_intercept(
        y_true=y_ext.values,
        y_prob=prob,
    )

    slope = calculate_calibration_slope(
        y_true=y_ext.values,
        y_prob=prob,
    )

    (
        intercept_ci_low,
        intercept_ci_high,
        slope_ci_low,
        slope_ci_high,
    ) = bootstrap_calibration_intercept_slope_ci(
        y_true=y_ext.values,
        y_prob=prob,
        n_boot=N_BOOT,
        seed=SEED,
    )

    brier = brier_score_loss(y_ext.values, prob)
    brier_ci_low, brier_ci_high = bootstrap_brier_ci(
        y_true=y_ext.values,
        y_prob=prob,
        n_boot=N_BOOT,
        seed=SEED,
    )

    auc_ci_low, auc_ci_high = bootstrap_metric_ci(
        y_true=y_ext.values,
        y_prob=prob,
        metric_fn=roc_auc_score,
        n_boot=N_BOOT,
        seed=SEED,
    )

    mean_pred, observed_rate, bin_count = calculate_calibration_curve_points(
        y_true=y_ext.values,
        y_prob=prob,
        n_bins=N_BINS,
    )

    calibration_curve_rows[label] = {
        "mean_pred": mean_pred,
        "observed_rate": observed_rate,
        "bin_count": bin_count,
        "intercept": intercept,
        "intercept_ci_low": intercept_ci_low,
        "intercept_ci_high": intercept_ci_high,
        "slope": slope,
        "slope_ci_low": slope_ci_low,
        "slope_ci_high": slope_ci_high,
        "brier": brier,
        "brier_ci_low": brier_ci_low,
        "brier_ci_high": brier_ci_high,
    }

    calibration_metric_rows.append({
        "cohort": "external_validation",
        "model": "RandomForest",
        "calibration_version": label,
        "external_intercept_only_alpha": external_update_alpha if label == "External intercept-only update" else np.nan,
        "calibration_intercept_citl": intercept,
        "calibration_intercept_citl_ci_low": intercept_ci_low,
        "calibration_intercept_citl_ci_high": intercept_ci_high,
        "calibration_slope": slope,
        "calibration_slope_ci_low": slope_ci_low,
        "calibration_slope_ci_high": slope_ci_high,
        "brier": brier,
        "brier_ci_low": brier_ci_low,
        "brier_ci_high": brier_ci_high,
        "auc": roc_auc_score(y_ext.values, prob),
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "event_rate": np.mean(y_ext.values),
        "mean_predicted_probability": np.mean(prob),
    })

calibration_metrics_df = pd.DataFrame(calibration_metric_rows)

calibration_metrics_path = os.path.join(
    output_dir,
    "external_RandomForest_calibration_metrics_three_versions.csv",
)
calibration_metrics_df.to_csv(
    calibration_metrics_path,
    index=False,
    encoding="utf-8-sig",
)

print("\nCalibration metrics:")
print(calibration_metrics_df)
print(f"\nCalibration metrics saved to: {calibration_metrics_path}")

curve_point_records = []

for label, r in calibration_curve_rows.items():
    for i in range(len(r["mean_pred"])):
        curve_point_records.append({
            "cohort": "external_validation",
            "model": "RandomForest",
            "calibration_version": label,
            "bin_index": i,
            "mean_predicted_probability": r["mean_pred"][i],
            "observed_event_rate": r["observed_rate"][i],
            "bin_count": r["bin_count"][i],
        })

calibration_points_df = pd.DataFrame(curve_point_records)

calibration_points_path = os.path.join(
    output_dir,
    "external_RandomForest_calibration_curve_points_three_versions.csv",
)
calibration_points_df.to_csv(
    calibration_points_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"Calibration curve points saved to: {calibration_points_path}")

fig, ax = plt.subplots(figsize=(14, 11), dpi=300)

for label in [
    "Original model",
    "Internal testing calibration",
    "External intercept-only update",
]:
    r = calibration_curve_rows[label]

    x = r["mean_pred"]
    y = r["observed_rate"]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        continue

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    label_text = (
        f"{label}\n"
        f"Int={r['intercept']:.2f}, "
        f"Slope={r['slope']:.2f}, "
        f"Brier={r['brier']:.3f}, "
        f"95% CI [{r['brier_ci_low']:.3f}, {r['brier_ci_high']:.3f}]"
    )

    ax.plot(
        x,
        y,
        linestyle=CURVE_LINESTYLES[label],
        marker=CURVE_MARKERS[label],
        markersize=11,
        markeredgewidth=1.5,
        lw=4.0,
        color=CURVE_COLORS[label],
        label=label_text,
    )

event_rate = float(np.mean(y_ext.values))

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    lw=2.5,
    label="Perfect calibration",
)

ax.axhline(
    event_rate,
    linestyle=":",
    color="black",
    lw=2.5,
    label=f"Event rate={event_rate:.3f}",
)

ax.set_title(
    "Calibration Curves in External Validation Cohort",
    fontsize=34,
    pad=22,
)
ax.set_xlabel("Mean Predicted Probability", fontsize=30, labelpad=16)
ax.set_ylabel("Observed Event Rate", fontsize=30, labelpad=16)

ax.set_xlim(0, 0.15)
ax.set_ylim(0, 0.15)

ax.set_xticks([0.00, 0.03, 0.06, 0.09, 0.12, 0.15])
ax.set_yticks([0.00, 0.03, 0.06, 0.09, 0.12, 0.15])

ax.tick_params(
    axis="both",
    which="major",
    labelsize=26,
    width=2.0,
    length=8,
)

ax.grid(alpha=0.25, linewidth=1.2)

for spine in ax.spines.values():
    spine.set_linewidth(2.0)

ax.legend(
    fontsize=22,
    loc="upper left",
    frameon=True,
    framealpha=0.95,
    edgecolor="lightgray",
    borderpad=1.0,
    labelspacing=0.95,
    handlelength=2.3,
    handletextpad=1.0,
)

plt.tight_layout()

calibration_png_path = os.path.join(
    output_dir,
    "external_RandomForest_calibration_three_versions.png",
)
calibration_pdf_path = os.path.join(
    output_dir,
    "external_RandomForest_calibration_three_versions.pdf",
)

plt.savefig(calibration_png_path, dpi=300, bbox_inches="tight")
plt.savefig(calibration_pdf_path, bbox_inches="tight")
plt.show()

print(f"\nCalibration figure saved to: {calibration_png_path}")
print(f"Calibration figure saved to: {calibration_pdf_path}")

threshold_min = 0.005
threshold_max = 0.10
threshold_step = 0.001
smooth_window = 7

thresholds = np.arange(
    threshold_min,
    threshold_max + threshold_step,
    threshold_step,
)

treat_none_nb = np.zeros_like(thresholds)
treat_all_nb = calculate_net_benefit_all(y_ext.values, thresholds)

dca_rows = []

for label, prob in probability_sets.items():
    nb = calculate_net_benefit(
        y_true=y_ext.values,
        y_prob=prob,
        thresholds=thresholds,
    )
    nb_smooth = smooth_curve(nb, window=smooth_window)

    for i, threshold in enumerate(thresholds):
        dca_rows.append({
            "cohort": "external_validation",
            "model": "RandomForest",
            "calibration_version": label,
            "threshold": threshold,
            "net_benefit": nb[i],
            "net_benefit_smoothed": nb_smooth[i],
            "net_benefit_treat_all": treat_all_nb[i],
            "net_benefit_treat_none": treat_none_nb[i],
            "better_than_none": nb[i] > 0,
            "better_than_all": nb[i] > treat_all_nb[i],
            "better_than_both": (nb[i] > 0) and (nb[i] > treat_all_nb[i]),
        })

dca_df = pd.DataFrame(dca_rows)

dca_csv_path = os.path.join(
    output_dir,
    "external_RandomForest_DCA_three_versions_0.005_0.10.csv",
)
dca_df.to_csv(dca_csv_path, index=False, encoding="utf-8-sig")

print(f"\nDCA values saved to: {dca_csv_path}")

fig, ax = plt.subplots(figsize=(14, 11), dpi=300)

ax.plot(
    thresholds,
    treat_none_nb,
    color="black",
    lw=2.6,
    linestyle="-",
    label="Treat none",
)

ax.plot(
    thresholds,
    treat_all_nb,
    color="gray",
    lw=2.8,
    linestyle="--",
    label="Treat all",
)

for label in [
    "Original model",
    "Internal testing calibration",
    "External intercept-only update",
]:
    df_m = dca_df[dca_df["calibration_version"] == label].copy()
    df_m = df_m.sort_values("threshold")

    ax.plot(
        df_m["threshold"],
        df_m["net_benefit_smoothed"],
        color=CURVE_COLORS[label],
        lw=4.0,
        linestyle=CURVE_LINESTYLES[label],
        label=label,
    )

ax.set_title(
    "Decision Curve Analysis in External Validation Cohort",
    fontsize=34,
    pad=22,
)
ax.set_xlabel("Threshold Probability", fontsize=30, labelpad=16)
ax.set_ylabel("Net Benefit", fontsize=30, labelpad=16)

ax.set_xlim(threshold_min, threshold_max)
ax.set_ylim(-0.01, 0.02)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=26,
    width=2.0,
    length=8,
)

ax.grid(alpha=0.25, linewidth=1.2)

for spine in ax.spines.values():
    spine.set_linewidth(2.0)

ax.legend(
    fontsize=26,
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    edgecolor="lightgray",
    borderpad=1.0,
    labelspacing=0.75,
    handlelength=2.4,
    handletextpad=1.0,
)

plt.tight_layout()

dca_png_path = os.path.join(
    output_dir,
    "external_RandomForest_DCA_three_versions_0.005_0.10.png",
)
dca_pdf_path = os.path.join(
    output_dir,
    "external_RandomForest_DCA_three_versions_0.005_0.10.pdf",
)

plt.savefig(dca_png_path, dpi=300, bbox_inches="tight")
plt.savefig(dca_pdf_path, bbox_inches="tight")
plt.show()

print(f"DCA figure saved to: {dca_png_path}")
print(f"DCA figure saved to: {dca_pdf_path}")

print("\nExternal validation completed.")
print("All outputs saved in:", output_dir)