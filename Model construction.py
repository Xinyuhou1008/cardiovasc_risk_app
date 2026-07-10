import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import config_context

SEED = 42
np.random.seed(SEED)

train_data_path = r"E:\Jupter_workplace\data\training_cohort_imputed_classed.csv"
test_data_path = r"E:\Jupter_workplace\data\testing_cohort_imputed_classed.csv"

target_col = "cdeath"
id_col = "id"

selected_feature_files = {
    "LogisticRegression": "LogisticRegression_selected_features.csv",
    "RandomForest": "RandomForest_selected_features.csv",
    "CatBoost": "CatBoost_selected_features.csv",
    "DecisionTree": "DecisionTree_selected_features.csv",
    "LGBM": "LGBM_selected_features.csv",
    "XGBoost": "XGBoost_selected_features.csv",
}

model_order = [
    "LGBM",
    "XGBoost",
    "LogisticRegression",
    "RandomForest",
    "CatBoost",
    "DecisionTree",
]

model_colors = {
    "RandomForest": "#DF4E4E",
    "LogisticRegression": "#8380BB",
    "XGBoost": "#FDA973",
    "CatBoost": "#3FA85B",
    "DecisionTree": "#47ABCF",
    "LGBM": "#DB3196",
}

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def load_selected_features(path):
    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {path}")

    if "selected_folds" in df.columns:
        feats = df.loc[df["selected_folds"] == 5, "feature"].dropna().astype(str).tolist()
        if len(feats) == 0:
            feats = df.loc[df["selected_folds"] >= 4, "feature"].dropna().astype(str).tolist()
    else:
        feats = df["feature"].dropna().astype(str).tolist()

    return feats

def infer_variable_types(raw_df, selected_features):
    raw_cols = list(raw_df.columns)

    nominal_categorical_cols = [c for c in raw_cols[1:47] if c in selected_features]
    ordered_cols = [c for c in raw_cols[47:64] if c in selected_features]
    numeric_cols = [c for c in selected_features if c not in nominal_categorical_cols + ordered_cols]

    return nominal_categorical_cols, ordered_cols, numeric_cols

def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_idx = int(np.nanargmax(f1_scores))

    return float(thresholds[best_idx])

def evaluate_at_threshold(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "ap": average_precision_score(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "y_pred": y_pred,
    }

def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_boot=1000, seed=SEED):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

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

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def bootstrap_threshold_ci(y_true, y_prob, threshold, n_boot=1000, seed=SEED):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))

    records = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "sensitivity": [],
        "specificity": [],
        "ppv": [],
        "npv": [],
    }

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample_idx]
        yp = y_prob[sample_idx]

        if len(np.unique(yt)) < 2:
            continue

        m = evaluate_at_threshold(yt, yp, threshold)

        records["accuracy"].append(m["accuracy"])
        records["precision"].append(m["precision"])
        records["recall"].append(m["recall"])
        records["f1"].append(m["f1"])
        records["sensitivity"].append(m["sensitivity"])
        records["specificity"].append(m["specificity"])
        records["ppv"].append(m["ppv"])
        records["npv"].append(m["npv"])

    ci = {}

    for k, vals in records.items():
        if len(vals) == 0:
            ci[k] = (np.nan, np.nan)
        else:
            ci[k] = (
                float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5)),
            )

    return ci

def calibration_bins_from_reference(y_prob, n_bins=10):
    y_prob = np.asarray(y_prob)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))

    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    return edges

def calibration_curve_fixed(y_true, y_prob, edges):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_bins = len(edges) - 1
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    mean_pred = np.full(n_bins, np.nan, dtype=float)
    frac_pos = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask = bin_ids == i

        if np.any(mask):
            mean_pred[i] = float(np.mean(y_prob[mask]))
            frac_pos[i] = float(np.mean(y_true[mask]))

    return mean_pred, frac_pos

def calibration_curve_ci(y_true, y_prob, n_boot=300, n_bins=10, seed=SEED):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    edges = calibration_bins_from_reference(y_prob, n_bins=n_bins)

    boot_frac_pos = []

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample_idx]
        yp = y_prob[sample_idx]

        if len(np.unique(yt)) < 2:
            continue

        _, frac_pos = calibration_curve_fixed(yt, yp, edges)
        boot_frac_pos.append(frac_pos)

    mean_pred, frac_pos = calibration_curve_fixed(y_true, y_prob, edges)

    if len(boot_frac_pos) == 0:
        return mean_pred, frac_pos, None, None

    boot_frac_pos = np.asarray(boot_frac_pos)
    lower = np.nanpercentile(boot_frac_pos, 2.5, axis=0)
    upper = np.nanpercentile(boot_frac_pos, 97.5, axis=0)

    return mean_pred, frac_pos, lower, upper

def bootstrap_brier_ci(y_true, y_prob, n_boot=1000, seed=SEED):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

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
            scores.append(brier_score_loss(yt, yp))
        except Exception:
            continue

    if len(scores) == 0:
        return np.nan, np.nan

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def make_generic_pipeline(model_name, nominal_cols, ordered_cols, numeric_cols):
    transformers = []

    if len(ordered_cols) > 0:
        transformers.append(("ord", "passthrough", ordered_cols))

    if len(nominal_cols) > 0:
        transformers.append(("cat", make_ohe(), nominal_cols))

    if len(numeric_cols) > 0:
        if model_name == "LogisticRegression":
            transformers.append(("num", StandardScaler(), numeric_cols))
        else:
            transformers.append(("num", "passthrough", numeric_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=5000, random_state=SEED)

    elif model_name == "RandomForest":
        clf = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    elif model_name == "CatBoost":
        clf = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            scale_pos_weight=None,
        )

    elif model_name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=SEED)

    elif model_name == "LGBM":
        clf = LGBMClassifier(
            objective="binary",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )

    elif model_name == "XGBoost":
        clf = XGBClassifier(
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

def get_generic_param_space(model_name, pos_weight_ratio=None):
    if model_name == "LogisticRegression":
        return {
            "classifier__C": [0.001, 0.01, 0.1, 1, 3, 10, 30, 100],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
            "classifier__class_weight": [None, "balanced"],
        }

    if model_name == "RandomForest":
        return {
            "classifier__n_estimators": randint(500, 2500),
            "classifier__max_depth": [None, 3, 4, 5, 6, 8, 10, 12, 15, 20],
            "classifier__min_samples_split": randint(2, 60),
            "classifier__min_samples_leaf": randint(1, 40),
            "classifier__max_features": ["sqrt", "log2", 0.2, 0.3, 0.4, 0.5, 0.7],
            "classifier__bootstrap": [True],
            "classifier__class_weight": [
                None,
                "balanced",
                "balanced_subsample",
            ],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__max_samples": [None, 0.5, 0.6, 0.7, 0.8, 0.9],
            "classifier__ccp_alpha": loguniform(1e-6, 1e-2),
        }

    if model_name == "CatBoost":
        return {
            "classifier__iterations": [100, 200, 300, 500],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__depth": [3, 4, 5, 6],
            "classifier__l2_leaf_reg": [float(x) for x in np.logspace(-3, 1, 50)],
            "classifier__subsample": [0.6, 0.7, 0.8, 1.0],
            "classifier__border_count": [32, 64, 128, 254],
            "classifier__scale_pos_weight": [1.0, max(1.0, pos_weight_ratio or 1.0)],
        }

    if model_name == "DecisionTree":
        return {
            "classifier__max_depth": [None, 3, 5, 7, 10, 15],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__class_weight": [None, "balanced"],
        }

    if model_name == "LGBM":
        ratio = float(pos_weight_ratio) if pos_weight_ratio is not None else 1.0
    
        return {
            "classifier__n_estimators": randint(400, 4000),
            "classifier__learning_rate": loguniform(0.003, 0.06),
            "classifier__num_leaves": randint(8, 128),
            "classifier__max_depth": [-1, 3, 4, 5, 6, 8, 10, 12],
            "classifier__min_child_samples": randint(5, 120),
            "classifier__min_split_gain": loguniform(1e-5, 0.2),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__subsample_freq": [0, 1, 2, 3, 5],
            "classifier__colsample_bytree": uniform(0.55, 0.45),
            "classifier__reg_alpha": loguniform(1e-6, 10.0),
            "classifier__reg_lambda": loguniform(1e-4, 50.0),
            "classifier__min_sum_hessian_in_leaf": loguniform(1e-3, 10.0),
            "classifier__max_bin": [63, 127, 255, 511],
            "classifier__boosting_type": ["gbdt"],
            "classifier__extra_trees": [False, True],
            "classifier__scale_pos_weight": [
                1.0,
                max(1.0, ratio * 0.5),
                max(1.0, ratio * 0.75),
                max(1.0, ratio),
                max(1.0, ratio * 1.25),
                max(1.0, ratio * 1.5),
                max(1.0, ratio * 2.0),
            ],
        }

    if model_name == "XGBoost":
        ratio = pos_weight_ratio if pos_weight_ratio is not None else 1.0

        return {
            "classifier__n_estimators": [300, 500, 800, 1200],
            "classifier__max_depth": [3, 4, 5, 6, 8],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__gamma": [0, 0.1, 0.2, 0.5],
            "classifier__reg_alpha": [0, 0.1, 1.0],
            "classifier__reg_lambda": [1.0, 2.0, 5.0],
            "classifier__scale_pos_weight": [
                1.0,
                max(1.0, ratio * 0.5),
                max(1.0, ratio),
                max(1.0, ratio * 1.5),
            ],
        }

    raise ValueError(f"Unknown model name: {model_name}")

def train_generic_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    nominal_cols,
    ordered_cols,
    numeric_cols,
):
    pipeline = make_generic_pipeline(
        model_name,
        nominal_cols,
        ordered_cols,
        numeric_cols,
    )

    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    pos_weight_ratio = neg / max(pos, 1)

    param_space = get_generic_param_space(
        model_name,
        pos_weight_ratio=pos_weight_ratio if model_name in ["CatBoost", "LGBM", "XGBoost"] else None,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    n_iter = 120 if model_name == "RandomForest" else 25

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=SEED,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print(f"{model_name} best params: {search.best_params_}")
    print(f"{model_name} best CV AP: {search.best_score_:.4f}")

    oof_prob = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    threshold = best_f1_threshold(y_train, oof_prob)

    best_model.fit(X_train, y_train)

    train_prob = oof_prob
    test_prob = best_model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_at_threshold(y_train, train_prob, threshold)
    test_metrics = evaluate_at_threshold(y_test, test_prob, threshold)

    return best_model, threshold, train_metrics, test_metrics, train_prob, test_prob, search

def make_calibrated_classifier(estimator, method="sigmoid", cv=None):
    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method=method,
            cv=cv,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method=method,
            cv=cv,
        )

train_raw = pd.read_csv(train_data_path)
test_raw = pd.read_csv(test_data_path)

if target_col not in train_raw.columns:
    raise ValueError(f"Target column '{target_col}' not found in training cohort.")

if target_col not in test_raw.columns:
    raise ValueError(f"Target column '{target_col}' not found in testing cohort.")

train_df = train_raw.copy()
test_df = test_raw.copy()

if id_col in train_df.columns:
    train_df = train_df.drop(columns=[id_col])

if id_col in test_df.columns:
    test_df = test_df.drop(columns=[id_col])

y_train_full = train_df[target_col]
y_test_full = test_df[target_col]

if y_train_full.dtype == "object" or y_train_full.dtype.name == "category":
    le = LabelEncoder()
    y_train_full = pd.Series(le.fit_transform(y_train_full), index=train_df.index)
    y_test_full = pd.Series(le.transform(y_test_full), index=test_df.index)
else:
    y_train_full = y_train_full.astype(int)
    y_test_full = y_test_full.astype(int)

X_train_all = train_df.drop(columns=[target_col])
X_test_all = test_df.drop(columns=[target_col])

common_cols = [c for c in X_train_all.columns if c in X_test_all.columns]

X_train_all = X_train_all[common_cols].copy()
X_test_all = X_test_all[common_cols].copy()

results = []

roc_records_train = {}
roc_records_test = {}
pr_records_test = {}

preds_train = {}
preds_test = {}
preds_test_calibrated = {}

trained_models = {}
calibrated_models = {}
model_meta = {}

train_auc_ci_records = {}
test_auc_ci_records = {}
test_ap_ci_records = {}
test_brier_ci_records = {}

for model_name in model_order:
    feature_path = selected_feature_files[model_name]
    selected_features = load_selected_features(feature_path)
    selected_features = [f for f in selected_features if f in common_cols]

    if len(selected_features) == 0:
        print(f"\n{model_name}: no valid selected features found, skipping.")
        continue

    nominal_cols, ordered_cols, numeric_cols = infer_variable_types(
        train_raw,
        selected_features,
    )

    X_train = X_train_all[selected_features].copy()
    X_test = X_test_all[selected_features].copy()

    print(f"\n===== {model_name} =====")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Nominal categorical ({len(nominal_cols)}): {nominal_cols}")
    print(f"Ordered ({len(ordered_cols)}): {ordered_cols}")
    print(f"Numeric ({len(numeric_cols)}): {numeric_cols}")

    best_model, threshold, train_metrics, test_metrics, train_prob, test_prob, search_obj = train_generic_model(
        model_name,
        X_train,
        y_train_full,
        X_test,
        y_test_full,
        nominal_cols,
        ordered_cols,
        numeric_cols,
    )

    calibration_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    calibrated_model = make_calibrated_classifier(
        best_model,
        method="sigmoid",
        cv=calibration_cv,
    )
    calibrated_model.fit(X_train, y_train_full)
    test_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    train_auc_ci = bootstrap_metric_ci(
        y_train_full.values,
        train_prob,
        roc_auc_score,
        n_boot=1000,
    )

    test_auc_ci = bootstrap_metric_ci(
        y_test_full.values,
        test_prob,
        roc_auc_score,
        n_boot=1000,
    )

    test_ap_ci = bootstrap_metric_ci(
        y_test_full.values,
        test_prob,
        average_precision_score,
        n_boot=1000,
    )

    test_brier_ci = bootstrap_brier_ci(
        y_test_full.values,
        test_prob,
        n_boot=1000,
    )

    thr_ci = bootstrap_threshold_ci(
        y_test_full.values,
        test_prob,
        threshold,
        n_boot=1000,
    )

    preds_train[model_name] = train_prob
    preds_test[model_name] = test_prob
    preds_test_calibrated[model_name] = test_prob_calibrated

    trained_models[model_name] = best_model
    calibrated_models[model_name] = calibrated_model

    model_meta[model_name] = {
        "threshold": threshold,
        "features": selected_features,
        "nominal_cols": nominal_cols,
        "ordered_cols": ordered_cols,
        "numeric_cols": numeric_cols,
        "best_params": search_obj.best_params_,
        "best_cv_ap": search_obj.best_score_,
    }

    train_auc_ci_records[model_name] = train_auc_ci
    test_auc_ci_records[model_name] = test_auc_ci
    test_ap_ci_records[model_name] = test_ap_ci
    test_brier_ci_records[model_name] = test_brier_ci

    fpr_train, tpr_train, _ = roc_curve(y_train_full, train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test_full, test_prob)
    precision_test, recall_test, _ = precision_recall_curve(y_test_full, test_prob)

    train_auc = roc_auc_score(y_train_full, train_prob)
    test_auc = roc_auc_score(y_test_full, test_prob)
    test_ap = average_precision_score(y_test_full, test_prob)

    roc_records_train[model_name] = (fpr_train, tpr_train, train_auc)
    roc_records_test[model_name] = (fpr_test, tpr_test, test_auc)
    pr_records_test[model_name] = (precision_test, recall_test, test_ap)

    joblib.dump(best_model, f"{model_name}_best_model.pkl")
    joblib.dump(calibrated_model, f"{model_name}_calibrated_model.pkl")
    joblib.dump(model_meta[model_name], f"{model_name}_meta.pkl")

    row = {
        "model": model_name,
        "n_features": len(selected_features),
        "threshold": threshold,

        "train_oof_ap": train_metrics["ap"],
        "train_oof_auc": train_metrics["auc"],
        "train_oof_auc_ci_low": train_auc_ci[0],
        "train_oof_auc_ci_high": train_auc_ci[1],
        "train_oof_brier": brier_score_loss(y_train_full, train_prob),

        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_ap": test_metrics["ap"],
        "test_auc": test_metrics["auc"],
        "test_brier": brier_score_loss(y_test_full, test_prob),

        "test_sensitivity": test_metrics["sensitivity"],
        "test_specificity": test_metrics["specificity"],
        "test_ppv": test_metrics["ppv"],
        "test_npv": test_metrics["npv"],

        "test_auc_ci_low": test_auc_ci[0],
        "test_auc_ci_high": test_auc_ci[1],
        "test_ap_ci_low": test_ap_ci[0],
        "test_ap_ci_high": test_ap_ci[1],
        "test_brier_ci_low": test_brier_ci[0],
        "test_brier_ci_high": test_brier_ci[1],

        "test_accuracy_ci_low": thr_ci["accuracy"][0],
        "test_accuracy_ci_high": thr_ci["accuracy"][1],
        "test_precision_ci_low": thr_ci["precision"][0],
        "test_precision_ci_high": thr_ci["precision"][1],
        "test_recall_ci_low": thr_ci["recall"][0],
        "test_recall_ci_high": thr_ci["recall"][1],
        "test_f1_ci_low": thr_ci["f1"][0],
        "test_f1_ci_high": thr_ci["f1"][1],
        "test_sensitivity_ci_low": thr_ci["sensitivity"][0],
        "test_sensitivity_ci_high": thr_ci["sensitivity"][1],
        "test_specificity_ci_low": thr_ci["specificity"][0],
        "test_specificity_ci_high": thr_ci["specificity"][1],
        "test_ppv_ci_low": thr_ci["ppv"][0],
        "test_ppv_ci_high": thr_ci["ppv"][1],
        "test_npv_ci_low": thr_ci["npv"][0],
        "test_npv_ci_high": thr_ci["npv"][1],
    }

    results.append(row)

    print(f"{model_name} train OOF AP: {train_metrics['ap']:.3f}")
    print(f"{model_name} train OOF AUC: {train_metrics['auc']:.3f}")
    print(f"{model_name} train OOF AUC 95% CI: [{train_auc_ci[0]:.3f}, {train_auc_ci[1]:.3f}]")
    print(f"{model_name} test AP: {test_metrics['ap']:.3f}")
    print(f"{model_name} test AP 95% CI: [{test_ap_ci[0]:.3f}, {test_ap_ci[1]:.3f}]")
    print(f"{model_name} test AUC: {test_metrics['auc']:.3f}")
    print(f"{model_name} test AUC 95% CI: [{test_auc_ci[0]:.3f}, {test_auc_ci[1]:.3f}]")
    print(f"{model_name} test Brier: {row['test_brier']:.4f}")
    print(f"{model_name} test Brier 95% CI: [{test_brier_ci[0]:.4f}, {test_brier_ci[1]:.4f}]")
    print(f"{model_name} test Acc: {test_metrics['accuracy']:.3f}")
    print(f"{model_name} test Precision: {test_metrics['precision']:.3f}")
    print(f"{model_name} test Recall/Sens: {test_metrics['recall']:.3f}")
    print(f"{model_name} test Spec: {test_metrics['specificity']:.3f}")
    print(f"{model_name} test F1: {test_metrics['f1']:.3f}")
    print(f"{model_name} test PPV: {test_metrics['ppv']:.3f}")
    print(f"{model_name} test NPV: {test_metrics['npv']:.3f}")
    print(f"{model_name} tuned threshold: {threshold:.4f}")

results_df = pd.DataFrame(results).sort_values("test_ap", ascending=False)
results_df.to_csv(
    "external_validation_model_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== External validation summary ===")
print(results_df)

pred_df_train = pd.DataFrame({"y_true": y_train_full.values})
pred_df_test = pd.DataFrame({"y_true": y_test_full.values})

for model_name in preds_train:
    pred_df_train[f"{model_name}_prob"] = preds_train[model_name]
    pred_df_test[f"{model_name}_prob"] = preds_test[model_name]

pred_df_train.to_csv(
    "training_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

pred_df_test.to_csv(
    "testing_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

import json

pred_df_test_calibrated = pd.DataFrame({"y_true": y_test_full.values})

for model_name in preds_test_calibrated:
    pred_df_test_calibrated[f"{model_name}_prob_calibrated"] = preds_test_calibrated[model_name]

pred_df_test_calibrated.to_csv(
    "testing_predictions_calibrated.csv",
    index=False,
    encoding="utf-8-sig",
)

plotting_config_rows = []

for i, model_name in enumerate(model_order):
    if model_name in preds_test:
        plotting_config_rows.append({
            "plot_order": i + 1,
            "model": model_name,
            "color": model_colors.get(model_name, ""),
            "raw_probability_column": f"{model_name}_prob",
            "calibrated_probability_column": f"{model_name}_prob_calibrated",
        })

plotting_config_df = pd.DataFrame(plotting_config_rows)
plotting_config_df.to_csv(
    "plotting_model_config.csv",
    index=False,
    encoding="utf-8-sig",
)

with open("plotting_model_colors.json", "w", encoding="utf-8") as f:
    json.dump(model_colors, f, ensure_ascii=False, indent=4)

def save_roc_curve_points(y_true, prob_dict, output_path):
    rows = []

    for model_name in model_order:
        if model_name not in prob_dict:
            continue

        y_prob = np.asarray(prob_dict[model_name])
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)

        for i in range(len(fpr)):
            rows.append({
                "model": model_name,
                "point_index": i,
                "fpr": fpr[i],
                "tpr": tpr[i],
                "threshold": thresholds[i],
                "auc": auc_value,
                "color": model_colors.get(model_name, ""),
            })

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

def save_pr_curve_points(y_true, prob_dict, output_path):
    rows = []

    for model_name in model_order:
        if model_name not in prob_dict:
            continue

        y_prob = np.asarray(prob_dict[model_name])
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)

        # precision_recall_curve returns len(precision)=len(thresholds)+1
        for i in range(len(precision)):
            threshold_value = thresholds[i] if i < len(thresholds) else np.nan

            rows.append({
                "model": model_name,
                "point_index": i,
                "recall": recall[i],
                "precision": precision[i],
                "threshold": threshold_value,
                "ap": ap_value,
                "baseline_prevalence": float(np.mean(y_true)),
                "color": model_colors.get(model_name, ""),
            })

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

def save_calibration_curve_points(
    y_true,
    prob_dict,
    output_path,
    n_boot=300,
    n_bins=5,
    seed=SEED,
):
    rows = []

    for model_name in model_order:
        if model_name not in prob_dict:
            continue

        y_prob = np.asarray(prob_dict[model_name])

        mean_pred, frac_pos, lower_true, upper_true = calibration_curve_ci(
            y_true,
            y_prob,
            n_boot=n_boot,
            n_bins=n_bins,
            seed=seed,
        )

        for i in range(len(mean_pred)):
            rows.append({
                "model": model_name,
                "bin_index": i,
                "mean_predicted_probability": mean_pred[i],
                "observed_event_rate": frac_pos[i],
                "observed_event_rate_ci_low": lower_true[i] if lower_true is not None else np.nan,
                "observed_event_rate_ci_high": upper_true[i] if upper_true is not None else np.nan,
                "n_bins": n_bins,
                "n_boot": n_boot,
                "probability_type": "calibrated",
                "color": model_colors.get(model_name, ""),
            })

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

def calculate_net_benefit_for_export(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    net_benefits = []

    for threshold in thresholds:
        y_pred = y_prob >= threshold
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))

        net_benefits.append({
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "net_benefit": nb,
        })

    return net_benefits

def save_dca_curve_points(
    y_true,
    prob_dict,
    output_path,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.linspace(0.005, 0.20, 200)

    y_true = np.asarray(y_true).astype(int)
    prevalence = float(np.mean(y_true))
    rows = []

    # Treat None
    for i, threshold in enumerate(thresholds):
        rows.append({
            "model": "Treat None",
            "point_index": i,
            "threshold": threshold,
            "net_benefit": 0.0,
            "tp": np.nan,
            "fp": np.nan,
            "prevalence": prevalence,
            "probability_type": "reference",
            "color": "black",
        })

    # Treat All
    for i, threshold in enumerate(thresholds):
        nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))

        rows.append({
            "model": "Treat All",
            "point_index": i,
            "threshold": threshold,
            "net_benefit": nb_all,
            "tp": np.nan,
            "fp": np.nan,
            "prevalence": prevalence,
            "probability_type": "reference",
            "color": "gray",
        })

    # Models
    for model_name in model_order:
        if model_name not in prob_dict:
            continue

        y_prob = np.asarray(prob_dict[model_name])
        nb_records = calculate_net_benefit_for_export(
            y_true,
            y_prob,
            thresholds,
        )

        for i, rec in enumerate(nb_records):
            rows.append({
                "model": model_name,
                "point_index": i,
                "threshold": rec["threshold"],
                "net_benefit": rec["net_benefit"],
                "tp": rec["tp"],
                "fp": rec["fp"],
                "prevalence": prevalence,
                "probability_type": "calibrated",
                "color": model_colors.get(model_name, ""),
            })

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

save_roc_curve_points(
    y_true=y_train_full.values,
    prob_dict=preds_train,
    output_path="roc_curve_points_training.csv",
)

save_roc_curve_points(
    y_true=y_test_full.values,
    prob_dict=preds_test,
    output_path="roc_curve_points_testing.csv",
)

save_pr_curve_points(
    y_true=y_test_full.values,
    prob_dict=preds_test,
    output_path="pr_curve_points_testing.csv",
)

save_calibration_curve_points(
    y_true=y_test_full.values,
    prob_dict=preds_test_calibrated,
    output_path="calibration_curve_points_testing_calibrated.csv",
    n_boot=300,
    n_bins=5,
    seed=SEED,
)

dca_thresholds_for_export = np.linspace(0.005, 0.20, 200)

save_dca_curve_points(
    y_true=y_test_full.values,
    prob_dict=preds_test_calibrated,
    output_path="dca_curve_points_testing_calibrated.csv",
    thresholds=dca_thresholds_for_export,
)

plotting_data_bundle = {
    "model_order": model_order,
    "model_colors": model_colors,

    "y_train": y_train_full.values,
    "y_test": y_test_full.values,

    # raw probabilities
    "preds_train_raw_oof": preds_train,
    "preds_test_raw": preds_test,

    # calibrated probabilities
    "preds_test_calibrated": preds_test_calibrated,

    # summary and CI records
    "results_df": results_df,
    "train_auc_ci_records": train_auc_ci_records,
    "test_auc_ci_records": test_auc_ci_records,
    "test_ap_ci_records": test_ap_ci_records,
    "test_brier_ci_records": test_brier_ci_records,

    # curve records already calculated in training loop
    "roc_records_train": roc_records_train,
    "roc_records_test": roc_records_test,
    "pr_records_test": pr_records_test,
}

joblib.dump(
    plotting_data_bundle,
    "plotting_data_bundle.pkl",
)

print("\nAdditional plotting files saved:")
print("- testing_predictions_calibrated.csv")
print("- plotting_model_config.csv")
print("- plotting_model_colors.json")
print("- roc_curve_points_training.csv")
print("- roc_curve_points_testing.csv")
print("- pr_curve_points_testing.csv")
print("- calibration_curve_points_testing_calibrated.csv")
print("- dca_curve_points_testing_calibrated.csv")
print("- plotting_data_bundle.pkl")

fig, axes = plt.subplots(1, 2, figsize=(17, 7), dpi=300)

ax = axes[0]

for model_name in preds_train:
    fpr, tpr, auc_val = roc_records_train[model_name]
    auc_ci = train_auc_ci_records[model_name]

    ax.plot(
        fpr,
        tpr,
        lw=2.2,
        color=model_colors.get(model_name, None),
        label=f"{model_name} (AUC={auc_val:.2f}, 95% CI [{auc_ci[0]:.2f}, {auc_ci[1]:.2f}])",
    )

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
ax.set_title("Training Cohort ROC", fontsize=15)
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)

ax = axes[1]

for model_name in preds_test:
    fpr, tpr, auc_val = roc_records_test[model_name]
    auc_ci = test_auc_ci_records[model_name]

    ax.plot(
        fpr,
        tpr,
        lw=2.2,
        color=model_colors.get(model_name, None),
        label=f"{model_name} (AUC={auc_val:.2f}, 95% CI [{auc_ci[0]:.2f}, {auc_ci[1]:.2f}])",
    )

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
ax.set_title("Internal Testing Cohort ROC", fontsize=15)
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("roc_training_testing.png", bbox_inches="tight", dpi=300)
plt.savefig("roc_training_testing.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

plot_df = results_df.sort_values("test_auc", ascending=True)

xerr = np.vstack([
    plot_df["test_auc"] - plot_df["test_auc_ci_low"],
    plot_df["test_auc_ci_high"] - plot_df["test_auc"],
])

ax.barh(
    plot_df["model"],
    plot_df["test_auc"],
    xerr=xerr,
    color=[model_colors.get(m, "#4C72B0") for m in plot_df["model"]],
    error_kw=dict(ecolor="black", lw=1, capsize=3),
)

for i, v in enumerate(plot_df["test_auc"]):
    ax.text(
        min(v + 0.005, 0.98),
        i,
        f"{v:.2f}",
        va="center",
        fontsize=10,
    )

ax.set_xlim(0, 1)
ax.set_xlabel("AUC", fontsize=13)
ax.set_title("Testing Cohort AUC Comparison with 95% CI", fontsize=15)
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig("testing_auc_comparison.png", bbox_inches="tight", dpi=300)
plt.savefig("testing_auc_comparison.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

test_pr_baseline = float(np.mean(y_test_full.values))

for model_name in preds_test:
    precision, recall, ap_val = pr_records_test[model_name]
    ap_ci = test_ap_ci_records[model_name]

    ax.plot(
        recall,
        precision,
        lw=2.0,
        color=model_colors.get(model_name, None),
        label=f"{model_name} (AP={ap_val:.2f}, 95% CI [{ap_ci[0]:.2f}, {ap_ci[1]:.2f}])",
    )

ax.axhline(
    test_pr_baseline,
    linestyle="--",
    color="gray",
    lw=1.5,
    label=f"Baseline prevalence AP={test_pr_baseline:.2f}",
)

ax.set_title("Internal Testing Cohort PR Curves", fontsize=15)
ax.set_xlabel("Recall", fontsize=13)
ax.set_ylabel("Precision", fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("testing_pr_curves.png", bbox_inches="tight", dpi=300)
plt.savefig("testing_pr_curves.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

test_event_rate = float(np.mean(y_test_full.values))

all_calib_x = []
all_calib_y = []

for model_name in preds_test_calibrated:
    mean_pred, frac_pos, lower_true, upper_true = calibration_curve_ci(
        y_test_full.values,
        preds_test_calibrated[model_name],
        n_boot=300,
        n_bins=5,
    )

    mask = np.isfinite(mean_pred) & np.isfinite(frac_pos)

    if not np.any(mask):
        continue

    x = mean_pred[mask]
    y = frac_pos[mask]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    all_calib_x.extend(x.tolist())
    all_calib_y.extend(y.tolist())

    ax.plot(
        x,
        y,
        marker="o",
        markersize=6,
        lw=1.8,
        color=model_colors.get(model_name, None),
        label=model_name,
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    lw=1.2,
    label="Perfect calibration",
)

ax.axhline(
    test_event_rate,
    linestyle=":",
    color="black",
    lw=1.2,
    label=f"Event rate={test_event_rate:.2f}",
)

ax.set_xlabel("Mean Predicted Probability", fontsize=13)
ax.set_ylabel("Observed Event Rate", fontsize=13)
ax.set_title("Calibration Curves on Internal Testing Cohort", fontsize=15)

if len(all_calib_x) > 0 and len(all_calib_y) > 0:
    ax.set_xlim(
        max(0.0, min(all_calib_x) - 0.02),
        min(1.0, max(all_calib_x) + 0.02),
    )
    ax.set_ylim(
        max(0.0, min(all_calib_y) - 0.02),
        min(1.0, max(all_calib_y) + 0.02),
    )
else:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("calibration_testing_calibrated.png", bbox_inches="tight", dpi=300)
plt.savefig("calibration_testing_calibrated.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

plot_df = results_df.sort_values("test_ap", ascending=True)

xerr = np.vstack([
    plot_df["test_ap"] - plot_df["test_ap_ci_low"],
    plot_df["test_ap_ci_high"] - plot_df["test_ap"],
])

ax.barh(
    plot_df["model"],
    plot_df["test_ap"],
    xerr=xerr,
    color=[model_colors.get(m, "#4C72B0") for m in plot_df["model"]],
    error_kw=dict(ecolor="black", lw=1, capsize=3),
)

for i, v in enumerate(plot_df["test_ap"]):
    ax.text(
        min(v + 0.005, 0.98),
        i,
        f"{v:.2f}",
        va="center",
        fontsize=10,
    )

ax.axvline(
    test_pr_baseline,
    linestyle="--",
    color="gray",
    lw=1.3,
    label=f"Baseline prevalence AP={test_pr_baseline:.2f}",
)

ax.set_xlim(0, 1)
ax.set_xlabel("Average Precision", fontsize=13)
ax.set_title("Testing Cohort AP Comparison with 95% CI", fontsize=15)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig("testing_ap_comparison.png", bbox_inches="tight", dpi=300)
plt.savefig("testing_ap_comparison.pdf", bbox_inches="tight")
plt.show()

best_params_rows = []

for model_name, meta in model_meta.items():
    row = {
        "model": model_name,
        "threshold": meta["threshold"],
        "n_features": len(meta["features"]),
        "best_cv_ap": meta["best_cv_ap"],
    }

    for param_name, param_value in meta["best_params"].items():
        row[param_name] = param_value

    best_params_rows.append(row)

best_params_df = pd.DataFrame(best_params_rows)
best_params_df.to_csv(
    "best_model_parameters.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== Best model parameters ===")
print(best_params_df)

print("\nAll done.")
print("Saved files:")
print("- external_validation_model_summary.csv")
print("- training_predictions.csv")
print("- testing_predictions.csv")
print("- testing_predictions_calibrated.csv")
print("- best_model_parameters.csv")
print("- plotting_model_config.csv")
print("- plotting_model_colors.json")
print("- roc_curve_points_training.csv")
print("- roc_curve_points_testing.csv")
print("- pr_curve_points_testing.csv")
print("- calibration_curve_points_testing_calibrated.csv")
print("- dca_curve_points_testing_calibrated.csv")
print("- plotting_data_bundle.pkl")
print("- *_best_model.pkl")
print("- *_calibrated_model.pkl")
print("- *_meta.pkl")
print("- roc_training_testing.png / .pdf")
print("- testing_auc_comparison.png / .pdf")
print("- testing_pr_curves.png / .pdf")
print("- calibration_testing_calibrated.png / .pdf")
print("- testing_ap_comparison.png / .pdf")
print("- DCA_testing_cohort.png / .pdf")