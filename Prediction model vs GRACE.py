import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

training_pred_path = "training_predictions.csv"
testing_pred_path = "testing_predictions.csv"
external_pred_path = "external_predictions.csv"

grace_testing_path = "GRACE_testing_cohort_with_GRACE.csv"
grace_external_path = "GRACE_external_cohort_with_GRACE.csv"

output_dir = "RF_GRACE_analysis"
os.makedirs(output_dir, exist_ok=True)

y_col_rf = "y_true"
y_col_grace = "cdeath"
USE_GRACE_OUTCOME_AS_Y = False

rf_prob_col_train = "RandomForest_prob"
rf_prob_col_test = "RandomForest_prob"
rf_prob_col_external = "RandomForest_prob"

grace_score_col = "GRACE_score"
grace_risk_col = "GRACE_risk_group"

N_BOOT = 1000
RANDOM_SEED = 42
DO_EXTERNAL_INTERCEPT_UPDATE = True

def safe_clip_prob(p):
    p = np.asarray(p, dtype=float)
    return np.clip(p, 1e-6, 1 - 1e-6)

def logit(p):
    p = safe_clip_prob(p)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))

def make_lr():
    try:
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    except Exception:
        return LogisticRegression(penalty="none", solver="lbfgs", max_iter=5000)

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    idx = np.arange(len(y_true))
    values = []

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample_idx]
        ys = y_score[sample_idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            values.append(metric_fn(yt, ys))
        except Exception:
            continue

    if len(values) == 0:
        return np.nan, np.nan

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def prepare_grace_risk_numeric(df):
    risk_map = {
        "Low risk": 1,
        "Intermediate risk": 2,
        "High risk": 3,
        "low risk": 1,
        "intermediate risk": 2,
        "high risk": 3,
        "Low": 1,
        "Intermediate": 2,
        "High": 3,
        "low": 1,
        "intermediate": 2,
        "high": 3,
        "LOW": 1,
        "INTERMEDIATE": 2,
        "HIGH": 3,
    }

    df = df.copy()
    df["GRACE_risk_numeric"] = df[grace_risk_col].map(risk_map)

    if df["GRACE_risk_numeric"].isna().any():
        missing = df.loc[df["GRACE_risk_numeric"].isna(), grace_risk_col].unique()
        raise ValueError(f"Unrecognized GRACE risk groups: {missing}")

    return df

def assign_rf_group(prob, low_cutoff, intermediate_cutoff):
    if prob <= low_cutoff:
        return "Low risk"
    elif prob <= intermediate_cutoff:
        return "Intermediate risk"
    else:
        return "High risk"

def get_rf_cutoffs_from_training():
    train_df = pd.read_csv(training_pred_path)
    train_df = train_df[[y_col_rf, rf_prob_col_train]].dropna().copy()
    train_df[y_col_rf] = train_df[y_col_rf].astype(int)
    train_df[rf_prob_col_train] = train_df[rf_prob_col_train].astype(float)

    train_df["risk_decile"] = pd.qcut(
        train_df[rf_prob_col_train].rank(method="first"),
        q=10,
        labels=np.arange(1, 11),
    ).astype(int)

    low_cutoff = train_df.loc[train_df["risk_decile"] <= 5, rf_prob_col_train].max()
    intermediate_cutoff = train_df.loc[train_df["risk_decile"] <= 8, rf_prob_col_train].max()

    return float(low_cutoff), float(intermediate_cutoff)

def merge_rf_grace(rf_df, grace_df, rf_prob_col, cohort_name):
    rf_df = rf_df.copy()
    grace_df = grace_df.copy()

    print("\n" + "=" * 80)
    print(f"{cohort_name}: raw data check")

    if y_col_rf in rf_df.columns:
        print(
            f"RF raw: N={len(rf_df)}, events={int(rf_df[y_col_rf].sum())}, "
            f"event rate={rf_df[y_col_rf].mean():.4f}"
        )

    if y_col_grace in grace_df.columns:
        print(
            f"GRACE raw: N={len(grace_df)}, events={int(grace_df[y_col_grace].sum())}, "
            f"event rate={grace_df[y_col_grace].mean():.4f}"
        )

    if "id" in rf_df.columns and "id" in grace_df.columns:
        df = pd.merge(
            rf_df[["id", y_col_rf, rf_prob_col]],
            grace_df[["id", y_col_grace, grace_score_col, grace_risk_col, "GRACE_risk_numeric"]],
            on="id",
            how="inner",
        )
        print(f"Merged by id: N={len(df)}")
    else:
        print("WARNING: no common id column; merging by row order.")
        df = pd.concat(
            [
                rf_df[[y_col_rf, rf_prob_col]].reset_index(drop=True),
                grace_df[[y_col_grace, grace_score_col, grace_risk_col, "GRACE_risk_numeric"]].reset_index(drop=True),
            ],
            axis=1,
        )

    if y_col_rf in df.columns and y_col_grace in df.columns:
        mismatch = (df[y_col_rf].astype(int) != df[y_col_grace].astype(int)).sum()
        print(f"Outcome mismatch between RF and GRACE files: {mismatch}")

    required = [y_col_rf, y_col_grace, rf_prob_col, grace_score_col, grace_risk_col, "GRACE_risk_numeric"]
    before = len(df)
    df = df.dropna(subset=required).copy()
    print(f"After dropna: N={len(df)}, dropped={before - len(df)}")

    if USE_GRACE_OUTCOME_AS_Y:
        df["y_true"] = df[y_col_grace].astype(int)
    else:
        df["y_true"] = df[y_col_rf].astype(int)

    df["RandomForest_probability_original"] = df[rf_prob_col].astype(float)
    df["GRACE_score"] = df[grace_score_col].astype(float)
    df["GRACE_risk_score"] = df["GRACE_risk_numeric"].astype(float)

    print(
        f"Final {cohort_name}: N={len(df)}, events={int(df['y_true'].sum())}, "
        f"event rate={df['y_true'].mean():.4f}"
    )
    print("=" * 80)

    return df

def fit_probability_recalibrator(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = safe_clip_prob(prob)

    if len(np.unique(y_true)) < 2:
        return {"type": "constant", "value": float(np.mean(y_true))}

    lr = make_lr()
    lr.fit(logit(prob).reshape(-1, 1), y_true)

    return {
        "type": "logit_probability",
        "model": lr,
        "intercept": float(lr.intercept_[0]),
        "slope": float(lr.coef_[0][0]),
    }

def apply_probability_recalibrator(calibrator, prob):
    prob = safe_clip_prob(prob)
    if calibrator["type"] == "constant":
        return np.repeat(calibrator["value"], len(prob))
    return calibrator["model"].predict_proba(logit(prob).reshape(-1, 1))[:, 1]

def fit_grace_score_recalibrator(y_true, grace_score):
    y_true = np.asarray(y_true).astype(int)
    x = np.asarray(grace_score).astype(float).reshape(-1, 1)

    if len(np.unique(y_true)) < 2:
        return {"type": "constant", "value": float(np.mean(y_true))}

    lr = make_lr()
    lr.fit(x, y_true)

    return {
        "type": "grace_score_recalibration",
        "model": lr,
        "intercept": float(lr.intercept_[0]),
        "slope": float(lr.coef_[0][0]),
    }

def apply_grace_score_recalibrator(calibrator, grace_score):
    x = np.asarray(grace_score).astype(float).reshape(-1, 1)
    if calibrator["type"] == "constant":
        return np.repeat(calibrator["value"], len(x))
    return calibrator["model"].predict_proba(x)[:, 1]

def add_risk_groups(df, low_cutoff, intermediate_cutoff):
    df = df.copy()
    df["RandomForest_risk_group"] = df["RandomForest_probability_original"].apply(
        lambda x: assign_rf_group(x, low_cutoff, intermediate_cutoff)
    )
    df["RandomForest_risk_score"] = df["RandomForest_risk_group"].map(
        {"Low risk": 1, "Intermediate risk": 2, "High risk": 3}
    ).astype(float)
    df["GRACE_risk_score"] = df["GRACE_risk_numeric"].astype(float)
    return df

def fit_group_probability_map(df, group_col, prob_col):
    tmp = df[[group_col, prob_col, "y_true"]].dropna().copy()
    group_map = tmp.groupby(group_col)[prob_col].mean().to_dict()
    fallback = float(df["y_true"].mean())
    return group_map, fallback

def apply_group_probability_map(df, group_col, group_map, fallback):
    return df[group_col].map(group_map).fillna(fallback).astype(float)

def calibrate_testing_apply_external(testing_df, external_df, low_cutoff, intermediate_cutoff):
    testing_df = add_risk_groups(testing_df, low_cutoff, intermediate_cutoff)
    external_df = add_risk_groups(external_df, low_cutoff, intermediate_cutoff)

    y_cal = testing_df["y_true"].astype(int).values

    rf_cal = fit_probability_recalibrator(
        y_cal,
        testing_df["RandomForest_probability_original"].values,
    )
    testing_df["RandomForest_probability"] = apply_probability_recalibrator(
        rf_cal,
        testing_df["RandomForest_probability_original"].values,
    )
    external_df["RandomForest_probability"] = apply_probability_recalibrator(
        rf_cal,
        external_df["RandomForest_probability_original"].values,
    )

    grace_cal = fit_grace_score_recalibrator(
        y_cal,
        testing_df["GRACE_score"].values,
    )
    testing_df["GRACE_probability"] = apply_grace_score_recalibrator(
        grace_cal,
        testing_df["GRACE_score"].values,
    )
    external_df["GRACE_probability"] = apply_grace_score_recalibrator(
        grace_cal,
        external_df["GRACE_score"].values,
    )

    rf_group_map, rf_fallback = fit_group_probability_map(
        testing_df,
        "RandomForest_risk_group",
        "RandomForest_probability",
    )
    grace_group_map, grace_fallback = fit_group_probability_map(
        testing_df,
        grace_risk_col,
        "GRACE_probability",
    )

    testing_df["RandomForest_risk_probability"] = apply_group_probability_map(
        testing_df,
        "RandomForest_risk_group",
        rf_group_map,
        rf_fallback,
    )
    external_df["RandomForest_risk_probability"] = apply_group_probability_map(
        external_df,
        "RandomForest_risk_group",
        rf_group_map,
        rf_fallback,
    )

    testing_df["GRACE_risk_probability"] = apply_group_probability_map(
        testing_df,
        grace_risk_col,
        grace_group_map,
        grace_fallback,
    )
    external_df["GRACE_risk_probability"] = apply_group_probability_map(
        external_df,
        grace_risk_col,
        grace_group_map,
        grace_fallback,
    )

    cal_info = pd.DataFrame([
        {"item": "RF calibration intercept", "value": rf_cal.get("intercept")},
        {"item": "RF calibration slope", "value": rf_cal.get("slope")},
        {"item": "GRACE score recalibration intercept", "value": grace_cal.get("intercept")},
        {"item": "GRACE score recalibration slope", "value": grace_cal.get("slope")},
        {"item": "RF group probability map", "value": str(rf_group_map)},
        {"item": "GRACE group probability map", "value": str(grace_group_map)},
    ])
    cal_info.to_csv(
        os.path.join(output_dir, "testing_based_calibration_info.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return testing_df, external_df

def find_intercept_update(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = safe_clip_prob(prob)
    lp = logit(prob)
    target = float(np.mean(y_true))

    if target <= 0 or target >= 1:
        return 0.0

    try:
        from scipy.optimize import brentq

        def f(a):
            return np.mean(sigmoid(lp + a)) - target

        return float(brentq(f, -50, 50))
    except Exception:
        return float(logit(np.array([target]))[0] - logit(np.array([np.mean(prob)]))[0])

def apply_intercept_update(prob, a):
    return sigmoid(logit(prob) + a)

def add_external_intercept_update(external_df):
    external_df = external_df.copy()
    y = external_df["y_true"].astype(int).values

    rf_a = find_intercept_update(y, external_df["RandomForest_probability"].values)
    grace_a = find_intercept_update(y, external_df["GRACE_probability"].values)

    external_df["RandomForest_probability_ext_intercept"] = apply_intercept_update(
        external_df["RandomForest_probability"].values,
        rf_a,
    )
    external_df["GRACE_probability_ext_intercept"] = apply_intercept_update(
        external_df["GRACE_probability"].values,
        grace_a,
    )

    info = pd.DataFrame([
        {
            "model": "RandomForest",
            "external_intercept_update": rf_a,
            "mean_probability_before": external_df["RandomForest_probability"].mean(),
            "mean_probability_after": external_df["RandomForest_probability_ext_intercept"].mean(),
            "observed_event_rate": external_df["y_true"].mean(),
        },
        {
            "model": "GRACE",
            "external_intercept_update": grace_a,
            "mean_probability_before": external_df["GRACE_probability"].mean(),
            "mean_probability_after": external_df["GRACE_probability_ext_intercept"].mean(),
            "observed_event_rate": external_df["y_true"].mean(),
        },
    ])

    info.to_csv(
        os.path.join(output_dir, "external_intercept_update_info.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return external_df

def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n

def calc_pvalue(aucs, sigma):
    from scipy.stats import norm

    if np.asarray(sigma).ndim == 0:
        return np.nan

    l = np.array([[1, -1]])
    var = np.dot(np.dot(l, sigma), l.T)[0, 0]
    if var <= 0 or np.isnan(var):
        return np.nan

    z = np.abs(np.diff(aucs)) / np.sqrt(var)
    return float(2 * (1 - norm.cdf(z)))

def delong_roc_test(y_true, pred1, pred2):
    y_true = np.asarray(y_true).astype(int)
    pred1 = np.asarray(pred1).astype(float)
    pred2 = np.asarray(pred2).astype(float)

    valid = np.isfinite(y_true) & np.isfinite(pred1) & np.isfinite(pred2)
    y_true = y_true[valid]
    pred1 = pred1[valid]
    pred2 = pred2[valid]

    n_pos = int(np.sum(y_true))
    if n_pos == 0 or n_pos == len(y_true):
        return np.nan

    order = np.argsort(-y_true)
    preds = np.vstack((pred1, pred2))[:, order]
    aucs, cov = fast_delong(preds, n_pos)
    return calc_pvalue(aucs, cov)

def run_delong(df, cohort_name):

    y = df["y_true"].astype(int).values

    comparisons = [
        {
            "analysis": "Continuous risk predictors",
            "rf_col": "RandomForest_probability",
            "grace_col": "GRACE_score",
            "RF_predictor": "RF continuous probability",
            "GRACE_predictor": "Continuous GRACE score",
        },
        {
            "analysis": "Risk stratification",
            "rf_col": "RandomForest_risk_score",
            "grace_col": "GRACE_risk_score",
            "RF_predictor": "RF risk stratification",
            "GRACE_predictor": "Original GRACE risk stratificationy",
        },
        {
            "analysis": "Risk-stratum probability",
            "rf_col": "RandomForest_risk_probability",
            "grace_col": "GRACE_risk_probability",
            "RF_predictor": "RF risk-stratum probability",
            "GRACE_predictor": "GRACE risk-stratum probability",
        },
    ]

    rows = []
    for c in comparisons:
        rf_score = df[c["rf_col"]].astype(float).values
        grace_score = df[c["grace_col"]].astype(float).values

        auc_rf = roc_auc_score(y, rf_score)
        auc_grace = roc_auc_score(y, grace_score)
        p = delong_roc_test(y, rf_score, grace_score)

        rows.append({
            "cohort": cohort_name,
            "analysis": c["analysis"],
            "RF_predictor": c["RF_predictor"],
            "GRACE_predictor": c["GRACE_predictor"],
            "RF_column": c["rf_col"],
            "GRACE_column": c["grace_col"],
            "AUC_RF": auc_rf,
            "AUC_GRACE": auc_grace,
            "AUC_difference_RF_minus_GRACE": auc_rf - auc_grace,
            "DeLong_p_value": p,
            "DeLong_p_value_formatted": format_p(p),
        })

    return pd.DataFrame(rows)

def save_delong_results(testing_df, external_df):
    delong_df = pd.concat(
        [
            run_delong(testing_df, "Testing calibration cohort"),
            run_delong(external_df, "External validation cohort"),
        ],
        axis=0,
        ignore_index=True,
    )

    delong_df.to_csv(
        os.path.join(output_dir, "DeLong_RF_vs_GRACE_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDeLong results:")
    print(delong_df)

    return delong_df

def calibration_intercept_slope(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = safe_clip_prob(prob)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    lr = make_lr()
    lr.fit(logit(prob).reshape(-1, 1), y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])

def save_performance_summary(testing_df, external_df):
    rows = []
    datasets = [
        ("Testing calibration cohort", testing_df),
        ("External validation cohort", external_df),
    ]

    models = [
        ("RF original probability", "RandomForest_probability_original"),
        ("RF testing-calibrated probability", "RandomForest_probability"),
        ("GRACE continuous score", "GRACE_score"),
        ("GRACE testing-recalibrated probability", "GRACE_probability"),
        ("RF risk stratificationy score", "RandomForest_risk_score"),
        ("Original GRACE risk stratification score", "GRACE_risk_score"),
        ("RF risk-stratum probability", "RandomForest_risk_probability"),
        ("GRACE risk-stratum probability", "GRACE_risk_probability"),
        ("RF external intercept-updated probability", "RandomForest_probability_ext_intercept"),
        ("GRACE external intercept-updated probability", "GRACE_probability_ext_intercept"),
    ]

    for cohort_name, df in datasets:
        y = df["y_true"].astype(int).values
        for model_name, col in models:
            if col not in df.columns:
                continue

            score = df[col].astype(float).values
            row = {
                "cohort": cohort_name,
                "model": model_name,
                "score_column": col,
                "n": len(df),
                "events": int(df["y_true"].sum()),
                "event_rate": float(df["y_true"].mean()),
                "mean_score_or_probability": float(np.mean(score)),
                "auc": float(roc_auc_score(y, score)),
            }

            if np.nanmin(score) >= 0 and np.nanmax(score) <= 1:
                ci, cs = calibration_intercept_slope(y, score)
                row["brier_score"] = float(brier_score_loss(y, score))
                row["calibration_intercept"] = ci
                row["calibration_slope"] = cs
            else:
                row["brier_score"] = np.nan
                row["calibration_intercept"] = np.nan
                row["calibration_slope"] = np.nan

            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(
        os.path.join(output_dir, "performance_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return out

def plot_roc(testing_df, external_df, score_type="continuous"):
    if score_type == "continuous":
        curves = [
            (testing_df, "Testing RandomForest probability", "RandomForest_probability", "#DF4E4E", "-"),
            (testing_df, "Testing GRACE score", "GRACE_score", "#A3D676", "--"),
            (external_df, "External RF probability", "RandomForest_probability", "#F5A623", "-"),
            (external_df, "External GRACE score", "GRACE_score", "#6DA3DD", "--"),
        ]
        title = "RandomForest vs GRACE Continuous Risk Predictors"
        prefix = "ROC_continuous_RF_vs_GRACE_score"
    else:
        curves = [
            (testing_df, "Testing RandomForest risk stratification", "RandomForest_risk_score", "#DF4E4E", "-"),
            (testing_df, "Testing GRACE risk stratification", "GRACE_risk_score", "#A3D676", "--"),
            (external_df, "External RandomForest risk stratification", "RandomForest_risk_score", "#F5A623", "-"),
            (external_df, "External GRACE risk stratification", "GRACE_risk_score", "#6DA3DD", "--"),
        ]
        title = "RandomForest vs GRACE Risk Stratification"
        prefix = "ROC_risk_category_RF_vs_GRACE"

    fig, ax = plt.subplots(figsize=(8.8, 7.6), dpi=300)
    rows = []

    for df, label, col, color, linestyle in curves:
        y = df["y_true"].astype(int).values
        score = df[col].astype(float).values
        fpr, tpr, _ = roc_curve(y, score)
        auc_value = auc(fpr, tpr)
        ci_low, ci_high = bootstrap_ci(y, score, roc_auc_score)

        rows.append({
            "model": label,
            "score_column": col,
            "auc": auc_value,
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
        })

        ax.plot(
            fpr * 100,
            tpr * 100,
            color=color,
            linestyle=linestyle,
            linewidth=2.8,
            label=f"{label}\nAUC={auc_value:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
        )

    ax.plot([0, 100], [0, 100], "--", color="gray", linewidth=1.8, label="Chance")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("False Positive Rate (%)", fontsize=16)
    ax.set_ylabel("True Positive Rate (%)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(title, fontsize=20)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=16, frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{prefix}.pdf"), bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, f"{prefix}_AUC_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

def plot_delong_forest(delong_df):
    plot_df = delong_df.copy()
    order_cohort = {"External validation cohort": 0, "Testing calibration cohort": 1}
    order_analysis = {"Continuous risk predictors": 0, "Risk stratification": 1, "Risk-stratum probability": 2}

    plot_df["cohort_order"] = plot_df["cohort"].map(order_cohort)
    plot_df["analysis_order"] = plot_df["analysis"].map(order_analysis)
    plot_df = plot_df.sort_values(["cohort_order", "analysis_order"]).reset_index(drop=True)

    plot_df["label"] = plot_df["cohort"] + "\n" + plot_df["analysis"]
    y_pos = np.arange(len(plot_df))[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    colors = plot_df["cohort"].map({
        "Testing calibration cohort": "#F5AA14",
        "External validation cohort": "#5C884C",
    })

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.scatter(
        plot_df["AUC_difference_RF_minus_GRACE"],
        y_pos,
        s=110,
        color=colors,
        edgecolor="black",
        zorder=3,
    )

    for i, row in plot_df.iterrows():
        diff = row["AUC_difference_RF_minus_GRACE"]
        ax.text(
            diff + 0.006 if diff >= 0 else diff - 0.006,
            y_pos[i],
            f"ΔAUC={diff:+.2f}, P={format_p(row['DeLong_p_value'])}",
            va="center",
            ha="left" if diff >= 0 else "right",
            fontsize=15,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=14)
    ax.set_title("RandomForest vs GRACE", fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", alpha=0.25)
    
    ax.set_xlim(-0.05, 0.30)
    ax.set_xticks(np.arange(-0.05, 0.31, 0.05))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "DeLong_AUC_difference_forest_plot.png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "DeLong_AUC_difference_forest_plot.pdf"), bbox_inches="tight")
    plt.close(fig)

low_cutoff, intermediate_cutoff = get_rf_cutoffs_from_training()

rf_testing = pd.read_csv(testing_pred_path)
rf_external = pd.read_csv(external_pred_path)

grace_testing = pd.read_csv(grace_testing_path)
grace_external = pd.read_csv(grace_external_path)

grace_testing = prepare_grace_risk_numeric(grace_testing)
grace_external = prepare_grace_risk_numeric(grace_external)

testing_df = merge_rf_grace(
    rf_testing,
    grace_testing,
    rf_prob_col_test,
    "Testing calibration cohort",
)

external_df = merge_rf_grace(
    rf_external,
    grace_external,
    rf_prob_col_external,
    "External validation cohort",
)

testing_df, external_df = calibrate_testing_apply_external(
    testing_df,
    external_df,
    low_cutoff,
    intermediate_cutoff,
)

if DO_EXTERNAL_INTERCEPT_UPDATE:
    external_df = add_external_intercept_update(external_df)

testing_df.to_csv(
    os.path.join(output_dir, "testing_calibration_final_dataset.csv"),
    index=False,
    encoding="utf-8-sig",
)
external_df.to_csv(
    os.path.join(output_dir, "external_validation_final_dataset.csv"),
    index=False,
    encoding="utf-8-sig",
)

event_rate_summary = pd.DataFrame([
    {
        "cohort": "Testing calibration cohort",
        "n": len(testing_df),
        "events": int(testing_df["y_true"].sum()),
        "event_rate": testing_df["y_true"].mean(),
        "mean_RF_probability": testing_df["RandomForest_probability"].mean(),
        "mean_GRACE_score": testing_df["GRACE_score"].mean(),
        "mean_GRACE_probability": testing_df["GRACE_probability"].mean(),
    },
    {
        "cohort": "External validation cohort",
        "n": len(external_df),
        "events": int(external_df["y_true"].sum()),
        "event_rate": external_df["y_true"].mean(),
        "mean_RF_probability": external_df["RandomForest_probability"].mean(),
        "mean_GRACE_score": external_df["GRACE_score"].mean(),
        "mean_GRACE_probability": external_df["GRACE_probability"].mean(),
    },
])

event_rate_summary.to_csv(
    os.path.join(output_dir, "event_rate_summary.csv"),
    index=False,
    encoding="utf-8-sig",
)

save_performance_summary(testing_df, external_df)

delong_df = save_delong_results(testing_df, external_df)
plot_delong_forest(delong_df)

plot_roc(testing_df, external_df, score_type="continuous")
plot_roc(testing_df, external_df, score_type="risk_category")

print("\nAnalysis completed.")
print(f"Results saved to: {output_dir}")
print("\nKey outputs:")
print("  event_rate_summary.csv")
print("  performance_summary.csv")
print("  testing_based_calibration_info.csv")
print("  external_intercept_update_info.csv")
print("  DeLong_RF_vs_GRACE_summary.csv")
print("  DeLong_AUC_difference_forest_plot.png/pdf")
print("  ROC_continuous_RF_vs_GRACE_score.png/pdf")
print("  ROC_risk_category_RF_vs_GRACE.png/pdf")