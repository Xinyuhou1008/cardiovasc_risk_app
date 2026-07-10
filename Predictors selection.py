import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from textwrap import fill

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)

file_path = r"E:\Jupter_workplace\data\training_cohort_imputed_classed.csv"
data = pd.read_csv(file_path)

target_col = "cdeath"
y = data[target_col].astype(int)

exclude_cols = [target_col]
if "id" in data.columns:
    exclude_cols.append("id")

X = data.drop(columns=exclude_cols)

nominal_categorical_cols = [c for c in data.columns[1:47] if c in X.columns]
ordered_cols = [c for c in data.columns[47:64] if c in X.columns]
numeric_cols = [c for c in X.columns if c not in nominal_categorical_cols + ordered_cols]

print(f"Nominal categorical cols ({len(nominal_categorical_cols)}): {nominal_categorical_cols}")
print(f"Ordered cols ({len(ordered_cols)}): {ordered_cols}")
print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")

ordinal_categories = []
for c in ordered_cols:
    uniq = data[c].dropna().unique().tolist()
    try:
        ordinal_categories.append(sorted(uniq))
    except TypeError:
        ordinal_categories.append(sorted(uniq, key=lambda x: str(x)))

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

ord_enc = OrdinalEncoder(
    categories=ordinal_categories if len(ordered_cols) > 0 else "auto",
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", ord_enc, ordered_cols),
        ("cat", ohe, nominal_categorical_cols),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)

def make_estimators(y_train):
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    return {
        "LogisticRegression": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            scale_pos_weight=scale_pos_weight,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=SEED,
            class_weight="balanced",
        ),
        "LGBM": LGBMClassifier(
            n_estimators=500,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
    }

def map_to_original_features(transformed_names, nominal_categorical_cols, ordered_cols, numeric_cols):

    original_features = []

    for name in transformed_names:
        if name.startswith("ord__"):
            raw = name[len("ord__"):]
            if raw in ordered_cols:
                original_features.append(raw)

        elif name.startswith("cat__"):
            raw = name[len("cat__"):]
            matched = None
            for col in nominal_categorical_cols:
                if raw == col or raw.startswith(col + "_"):
                    matched = col
                    break
            if matched is not None:
                original_features.append(matched)

        elif name.startswith("num__"):
            raw = name[len("num__"):]
            if raw in numeric_cols:
                original_features.append(raw)

        else:
            if name in nominal_categorical_cols or name in ordered_cols or name in numeric_cols:
                original_features.append(name)

    return sorted(set(original_features))

def get_model_importance(fitted_estimator):

    if hasattr(fitted_estimator, "feature_importances_"):
        return np.asarray(fitted_estimator.feature_importances_, dtype=float)

    if hasattr(fitted_estimator, "coef_"):
        coef = np.asarray(fitted_estimator.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]
        return np.abs(coef)

    raise ValueError(f"{type(fitted_estimator).__name__} does not expose feature importance.")

def aggregate_importance_to_original(feature_names, importances, nominal_categorical_cols, ordered_cols, numeric_cols):

    agg = {}

    for name, imp in zip(feature_names, importances):
        if name.startswith("ord__"):
            key = name[len("ord__"):]

        elif name.startswith("cat__"):
            raw = name[len("cat__"):]
            matched = None
            for col in nominal_categorical_cols:
                if raw == col or raw.startswith(col + "_"):
                    matched = col
                    break
            key = matched if matched is not None else raw

        elif name.startswith("num__"):
            key = name[len("num__"):]

        else:
            key = name

        agg[key] = agg.get(key, 0.0) + float(imp)

    return agg

def plot_importance_panel(ax, importance_dict, title, cmap_name, top_n=None):
    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        items = items[:top_n]

    if len(items) == 0:
        ax.set_axis_off()
        return

    features = [x[0] for x in items][::-1]
    values = [x[1] for x in items][::-1]

    try:
        cmap = plt.colormaps.get_cmap(cmap_name)
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)

    colors = cmap(np.linspace(0.35, 0.95, len(features)))

    ax.barh(range(len(features)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([fill(f, width=32) for f in features], fontsize=12)
    ax.set_xlabel("Normalized Importance", fontsize=14)
    ax.set_title(title, fontsize=16, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.invert_yaxis()

    max_val = max(values) if len(values) > 0 else 0
    offset = max_val * 0.01 if max_val > 0 else 0.001

    for i, v in enumerate(values):
        ax.text(v + offset, i, f"{v:.3f}", va="center", fontsize=8)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

selected_features = {}
selected_features_detail = {}
feature_importances_all_folds = {
    "LogisticRegression": defaultdict(list),
    "RandomForest": defaultdict(list),
    "CatBoost": defaultdict(list),
    "DecisionTree": defaultdict(list),
    "LGBM": defaultdict(list),
    "XGBoost": defaultdict(list),
}

model_names = [
    "LogisticRegression",
    "RandomForest",
    "CatBoost",
    "DecisionTree",
    "LGBM",
    "XGBoost",
]

for model_name in model_names:
    fold_counter = Counter()

    for train_idx, val_idx in kfold.split(X, y):
        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()

        X_train_trans = preprocessor.fit_transform(X_train_fold, y_train_fold)
        feature_names = np.array(preprocessor.get_feature_names_out())

        clf = make_estimators(y_train_fold)[model_name]

        selector = SelectFromModel(
            estimator=clf,
            threshold="mean",
        )
        selector.fit(X_train_trans, y_train_fold)

        support_mask = selector.get_support()
        selected_transformed_names = feature_names[support_mask]
        selected_original_names = map_to_original_features(
            selected_transformed_names,
            nominal_categorical_cols,
            ordered_cols,
            numeric_cols,
        )
        fold_counter.update(selected_original_names)

        fitted_estimator = selector.estimator_
        raw_importance = get_model_importance(fitted_estimator)
        fold_importance = aggregate_importance_to_original(
            feature_names,
            raw_importance,
            nominal_categorical_cols,
            ordered_cols,
            numeric_cols,
        )

        for feat, imp in fold_importance.items():
            feature_importances_all_folds[model_name][feat].append(imp)

    final_selected = [feat for feat, cnt in fold_counter.items() if cnt == 5]
    selected_features[model_name] = final_selected

    selected_features_detail[model_name] = pd.DataFrame(
        [{"feature": feat, "selected_folds": cnt} for feat, cnt in fold_counter.items()]
    ).sort_values("selected_folds", ascending=False)

    print(f"\n===== {model_name} =====")
    print("最终筛选变量：")
    print(final_selected)

for model_name, df_res in selected_features_detail.items():
    out_path = f"{model_name}_selected_features.csv"
    df_res.to_csv(out_path, index=False, encoding="utf-8-sig")

average_feature_importances = {}

for model_name, feat_dict in feature_importances_all_folds.items():
    avg_imp = {feat: np.mean(vals) for feat, vals in feat_dict.items()}

    total = sum(avg_imp.values())
    if total > 0:
        avg_imp = {feat: val / total for feat, val in avg_imp.items()}

    average_feature_importances[model_name] = avg_imp

plot_feature_importances = {}
for model_name in average_feature_importances:
    final_feats = selected_features[model_name]
    plot_feature_importances[model_name] = {
        feat: average_feature_importances[model_name][feat]
        for feat in final_feats
        if feat in average_feature_importances[model_name]
    }

model_order = [
    ("RandomForest", "Blues"),
    ("XGBoost", "OrRd"),
    ("LogisticRegression", "Purples"),
    ("CatBoost", "Greens"),
    ("LGBM", "RdPu"),
    ("DecisionTree", "GnBu"),
]

fig, axes = plt.subplots(3, 2, figsize=(22, 20), dpi=300)
axes = axes.flatten()

for ax, (model_name, cmap_name) in zip(axes, model_order):
    plot_importance_panel(
        ax=ax,
        importance_dict=plot_feature_importances[model_name],
        title=f"Selected Variable Importance for {model_name}",
        cmap_name=cmap_name,
        top_n=None,
    )

plt.tight_layout()
fig.savefig("selected_variable_importance_horizontal.png", bbox_inches="tight", dpi=300)
fig.savefig("selected_variable_importance_horizontal.pdf", bbox_inches="tight")
plt.show()