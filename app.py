import os
import warnings

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Cardiovascular Risk Prediction Model",
    layout="wide",
)

LOW_RISK_THRESHOLD = 0.0164       # 1.64%
HIGH_RISK_THRESHOLD = 0.0545      # 5.45%

@st.cache_resource
def load_model_and_meta():
    model = joblib.load("RandomForest_best_model.pkl")
    meta = joblib.load("RandomForest_meta.pkl")
    return model, meta

@st.cache_resource
def load_shap_background():
    background_path = "RandomForest_shap_background.pkl"
    if os.path.exists(background_path):
        background = joblib.load(background_path)
        return background
    return None

try:
    model, meta = load_model_and_meta()
    selected_features = meta.get("features", [])
    model_threshold = float(meta.get("threshold", 0.5))
except Exception as e:
    st.error(
        "Model files were not loaded successfully. Please check whether "
        "RandomForest_best_model.pkl and RandomForest_meta.pkl are in the same folder as app.py."
    )
    st.exception(e)
    st.stop()

shap_background = load_shap_background()

if not selected_features:
    selected_features = [
        "Age",
        "Alcohol_consumption",
        "BMI",
        "BNP",
        "D_dimer",
        "EF",
        "GLU",
        "Gensini_score",
        "HF_HPI",
        "cTnI",
        "LDL_C",
        "LVEDD",
        "PM2.5",
        "Temperature",
        "TG",
        "Weighted_value_of_calcification",
        "Weighted_value_of_diffuse_lesion",
    ]

def get_model_required_raw_features(model, fallback_features):
    """
    If the model is a sklearn Pipeline with a ColumnTransformer preprocessor,
    use the raw columns expected by the preprocessor.

    This is important because meta['features'] may not always include all raw
    columns required by the fitted preprocessor, which can cause errors such as:
    columns are missing: {'HF_HPI', 'Alcohol_consumption'}.
    """
    if not hasattr(model, "named_steps"):
        return fallback_features

    if "preprocessor" not in model.named_steps:
        return fallback_features

    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(preprocessor, "transformers_"):
        return fallback_features

    required_features = []

    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue

        if columns is None:
            continue

        if isinstance(columns, str):
            columns = [columns]

        for col in list(columns):
            if col not in required_features:
                required_features.append(col)

    if len(required_features) == 0:
        return fallback_features

    return required_features

# Use the actual raw input columns required by the fitted model pipeline.
selected_features = get_model_required_raw_features(model, selected_features)

st.title("Cardiovascular Risk Prediction Model")

st.markdown(
    "A discharge-oriented risk stratification tool for estimating "
    "**1-year post-discharge cardiac mortality risk** in survivors of acute myocardial infarction."
)

FEATURE_LABELS = {
    "Age": "Age, years",
    "Alcohol_consumption": "Alcohol consumption",
    "BMI": "Body mass index, kg/m²",
    "BNP": "B-type natriuretic peptide, BNP",
    "D_dimer": "D-dimer",
    "EF": "Ejection fraction, %",
    "GLU": "Glucose, GLU",
    "Gensini_score": "Gensini score",
    "HF_HPI": "Heart failure during index hospitalization",
    "cTnI": "Cardiac troponin I, cTnI",
    "LDL_C": "Low-density lipoprotein cholesterol, LDL-C",
    "LVEDD": "Left ventricular end-diastolic diameter, mm",
    "PM2.5": "PM2.5, μg/m³",
    "Temperature": "Environmental temperature, °C",
    "TG": "Triglyceride, TG",
    "Weighted_value_of_calcification": "Weighted value of calcification",
    "Weighted_value_of_diffuse_lesion": "Weighted value of diffuse lesion",
}

FEATURE_GROUPS = {
    "General information and lifestyle": [
        "Age",
        "BMI",
        "Alcohol_consumption",
    ],
    "Environmental exposure": [
        "Temperature",
        "PM2.5",
    ],
    "Clinical presentation during hospitalization": [
        "HF_HPI",
    ],
    "Angiographic characteristics": [
        "Gensini_score",
        "Weighted_value_of_calcification",
        "Weighted_value_of_diffuse_lesion",
    ],
    "Echocardiography": [
        "LVEDD",
        "EF",
    ],
    "Laboratory tests": [
        "BNP",
        "D_dimer",
        "GLU",
        "cTnI",
        "LDL_C",
        "TG",
    ],
}

def get_feature_group(feature):
    for group_name, features in FEATURE_GROUPS.items():
        if feature in features:
            return group_name
    return "Other"

CONTINUOUS_INPUTS = {
    "Age": {
        "min": 18,
        "max": 120,
        "value": 65,
        "step": 1,
        "definition": "Age of the patient in years.",
    },
    "BMI": {
        "min": 10.0,
        "max": 60.0,
        "value": 24.0,
        "step": 0.1,
        "definition": "Body mass index, kg/m².",
    },
    "Temperature": {
        "min": -40.0,
        "max": 50.0,
        "value": 15.0,
        "step": 0.1,
        "definition": "Environmental temperature, °C.",
    },
    "PM2.5": {
        "min": 0.0,
        "max": 1000.0,
        "value": 45.0,
        "step": 1.0,
        "definition": "Ambient PM2.5 exposure, μg/m³.",
    },
    "Gensini_score": {
        "min": 0.0,
        "max": 500.0,
        "value": 30.0,
        "step": 1.0,
        "definition": "Gensini score reflecting the severity of coronary artery stenosis.",
    },
    "Weighted_value_of_calcification": {
        "min": 0.0,
        "max": 30.0,
        "value": 1.0,
        "step": 0.1,
        "definition": "Weighted value of coronary calcification.",
    },
    "Weighted_value_of_diffuse_lesion": {
        "min": 0.0,
        "max": 30.0,
        "value": 2.0,
        "step": 0.1,
        "definition": "Weighted value of diffuse coronary lesion.",
    },
    "LVEDD": {
        "min": 20.0,
        "max": 100.0,
        "value": 50.0,
        "step": 1.0,
        "definition": "Left ventricular end-diastolic diameter, mm.",
    },
    "EF": {
        "min": 10.0,
        "max": 100.0,
        "value": 55.0,
        "step": 1.0,
        "definition": "Left ventricular ejection fraction, %.",
    },
}

CATEGORICAL_OPTIONS = {
    "Alcohol_consumption": {
        0: "Never",
        1: "Current drinker",
        2: "Former drinker",
    },
    "HF_HPI": {
        0: "No heart failure during hospitalization",
        1: "Heart failure during hospitalization",
    },
}

# Keys are model input codes. Values are only display labels.
BINNED_OPTIONS = {
    "BNP": {
        1: "Normal or below ULN",
        2: ">ULN to 1000 pg/mL",
        3: "1001-5000 pg/mL",
        4: ">5000 pg/mL",
    },
    "D_dimer": {
        0: "Normal or below ULN",
        1: ">ULN and <500 ng/mL",
        2: ">=500 ng/mL",
    },
    "GLU": {
        0: "Normal or below ULN",
        1: ">ULN and <7 mmol/L",
        2: "7-16.6 mmol/L",
        3: ">=16.7 mmol/L",
    },
    "cTnI": {
        1: "<=10 x ULN",
        2: "10-100 x ULN",
        3: "100-1000 x ULN",
        4: "1000-10000 x ULN",
        5: ">10000 x ULN",
    },
    "LDL_C": {
        1: "<1.40 mmol/L",
        2: "1.40-1.79 mmol/L",
        3: "1.80-2.59 mmol/L",
        4: "2.60-3.39 mmol/L",
        5: "3.40-4.09 mmol/L",
        6: "4.10-4.89 mmol/L",
        7: ">=4.90 mmol/L",
    },
    "TG": {
        1: "<1.70 mmol/L",
        2: "1.70-2.29 mmol/L",
        3: ">=2.30 mmol/L",
    },
}

# More detailed definitions for the variable definition table
BINNED_DEFINITIONS = {
    "BNP": {
        1: "Below or within the upper limit of normal range",
        2: "Above the upper limit of normal range and <=1000 pg/mL",
        3: "1001-5000 pg/mL",
        4: ">5000 pg/mL",
    },
    "D_dimer": {
        0: "Below or within the upper limit of normal range",
        1: "Above the upper limit of normal range and <500 ng/mL",
        2: ">=500 ng/mL",
    },
    "GLU": {
        0: "Below or within the upper limit of normal range",
        1: "Above the upper limit of normal range and <7 mmol/L",
        2: ">=7 and <16.7 mmol/L",
        3: ">=16.7 mmol/L",
    },
    "cTnI": {
        1: "Within 10 times the upper limit of normal range",
        2: "Within 100 times the upper limit of normal range",
        3: "Within 1000 times the upper limit of normal range",
        4: "Within 10000 times the upper limit of normal range",
        5: "More than 10000 times the upper limit of normal range",
    },
    "LDL_C": BINNED_OPTIONS["LDL_C"],
    "TG": BINNED_OPTIONS["TG"],
}

def build_variable_definition_table():
    rows = []

    for feat, setting in CONTINUOUS_INPUTS.items():
        if feat not in selected_features:
            continue

        rows.append({
            "Category": get_feature_group(feat),
            "Variable": feat,
            "Display name": FEATURE_LABELS.get(feat, feat),
            "Input type": "Direct numerical input",
            "Code": "-",
            "Definition": setting.get("definition", "Direct numerical input."),
        })

    for feat, options in CATEGORICAL_OPTIONS.items():
        if feat not in selected_features:
            continue

        for code, definition in options.items():
            rows.append({
                "Category": get_feature_group(feat),
                "Variable": feat,
                "Display name": FEATURE_LABELS.get(feat, feat),
                "Input type": "Categorical selection",
                "Code": code,
                "Definition": definition,
            })

    for feat, options in BINNED_OPTIONS.items():
        if feat not in selected_features:
            continue

        for code in options.keys():
            rows.append({
                "Category": get_feature_group(feat),
                "Variable": feat,
                "Display name": FEATURE_LABELS.get(feat, feat),
                "Input type": "Classified laboratory variable",
                "Code": code,
                "Definition": BINNED_DEFINITIONS.get(feat, options).get(code, options[code]),
            })

    definition_df = pd.DataFrame(rows)

    if not definition_df.empty:
        category_order = list(FEATURE_GROUPS.keys())
        definition_df["Category"] = pd.Categorical(
            definition_df["Category"],
            categories=category_order,
            ordered=True,
        )
        definition_df["Code_sort"] = definition_df["Code"].astype(str)
        definition_df = definition_df.sort_values(
            by=["Category", "Variable", "Code_sort"]
        ).drop(columns=["Code_sort"]).reset_index(drop=True)

    return definition_df

with st.expander("Variable definitions and classification criteria", expanded=False):
    definition_df = build_variable_definition_table()
    st.dataframe(definition_df, use_container_width=True, height=520)

def collect_user_inputs(selected_features):
    input_data = {}

    st.header("Enter Clinical Parameters")

    for group_name, group_features in FEATURE_GROUPS.items():
        current_features = [f for f in group_features if f in selected_features]

        if len(current_features) == 0:
            continue

        st.subheader(group_name)
        cols = st.columns(3)

        for i, feat in enumerate(current_features):
            col = cols[i % 3]
            label = FEATURE_LABELS.get(feat, feat)

            if feat in CONTINUOUS_INPUTS:
                setting = CONTINUOUS_INPUTS[feat]
                default_value = setting["value"]

                if isinstance(default_value, int):
                    value = col.number_input(
                        label,
                        min_value=int(setting["min"]),
                        max_value=int(setting["max"]),
                        value=int(setting["value"]),
                        step=int(setting["step"]),
                        key=feat,
                    )
                else:
                    value = col.number_input(
                        label,
                        min_value=float(setting["min"]),
                        max_value=float(setting["max"]),
                        value=float(setting["value"]),
                        step=float(setting["step"]),
                        key=feat,
                    )

                input_data[feat] = value

            elif feat in CATEGORICAL_OPTIONS:
                options = CATEGORICAL_OPTIONS[feat]

                value = col.selectbox(
                    label,
                    options=list(options.keys()),
                    format_func=lambda x, options=options: options[x],
                    key=feat,
                )
                input_data[feat] = value

            elif feat in BINNED_OPTIONS:
                options = BINNED_OPTIONS[feat]

                value = col.selectbox(
                    label,
                    options=list(options.keys()),
                    format_func=lambda x, options=options: options[x],
                    key=feat,
                )
                input_data[feat] = value

            else:
                st.warning(f"No input definition found for variable: {feat}")

    return input_data

input_data = collect_user_inputs(selected_features)

def get_risk_group(prob):
    if prob > HIGH_RISK_THRESHOLD:
        return {
            "name": "High risk group",
            "short_name": "High risk",
            "color": "#f8d7da",
            "border": "#dc3545",
            "description": f"Predicted probability > {HIGH_RISK_THRESHOLD * 100:.2f}%",
        }
    elif prob > LOW_RISK_THRESHOLD:
        return {
            "name": "Intermediate risk group",
            "short_name": "Intermediate risk",
            "color": "#fff3cd",
            "border": "#ffc107",
            "description": f"{LOW_RISK_THRESHOLD * 100:.2f}% < predicted probability <= {HIGH_RISK_THRESHOLD * 100:.2f}%",
        }
    else:
        return {
            "name": "Low risk group",
            "short_name": "Low risk",
            "color": "#d1e7dd",
            "border": "#198754",
            "description": f"Predicted probability <= {LOW_RISK_THRESHOLD * 100:.2f}%",
        }

def get_modifiable_suggestions(input_data):
    suggestions = []

    if "Alcohol_consumption" in input_data:
        alcohol = input_data["Alcohol_consumption"]
        if alcohol == 1:
            suggestions.append({
                "Domain": "Alcohol consumption",
                "Current status": "Current drinker",
                "Follow-up focus": "Assess alcohol intake and provide lifestyle counseling."
            })
        elif alcohol == 2:
            suggestions.append({
                "Domain": "Alcohol consumption",
                "Current status": "Former drinker",
                "Follow-up focus": "Reinforce abstinence and monitor relapse risk when relevant."
            })

    if "BMI" in input_data:
        bmi = float(input_data["BMI"])
        if bmi < 18.5:
            suggestions.append({
                "Domain": "BMI",
                "Current status": f"{bmi:.1f} kg/m², underweight range",
                "Follow-up focus": "Evaluate nutritional status and frailty risk."
            })
        elif bmi >= 24:
            suggestions.append({
                "Domain": "BMI",
                "Current status": f"{bmi:.1f} kg/m², elevated BMI",
                "Follow-up focus": "Support weight management, diet optimization, and physical activity planning."
            })

    if "GLU" in input_data:
        glu = input_data["GLU"]
        if glu == 1:
            suggestions.append({
                "Domain": "Glucose",
                "Current status": BINNED_OPTIONS["GLU"].get(glu, glu),
                "Follow-up focus": "Review glucose status and consider metabolic follow-up."
            })
        elif glu in [2, 3]:
            suggestions.append({
                "Domain": "Glucose",
                "Current status": BINNED_OPTIONS["GLU"].get(glu, glu),
                "Follow-up focus": "Strengthen glycemic monitoring and optimize glucose management."
            })

    if "LDL_C" in input_data:
        ldl = input_data["LDL_C"]
        if ldl >= 3:
            suggestions.append({
                "Domain": "LDL-C",
                "Current status": BINNED_OPTIONS["LDL_C"].get(ldl, ldl),
                "Follow-up focus": "Optimize lipid-lowering therapy and assess LDL-C target achievement."
            })

    if "TG" in input_data:
        tg = input_data["TG"]
        if tg >= 2:
            suggestions.append({
                "Domain": "Triglycerides",
                "Current status": BINNED_OPTIONS["TG"].get(tg, tg),
                "Follow-up focus": "Review diet, alcohol intake, glycemic control, and lipid management."
            })

    return pd.DataFrame(suggestions)

def get_final_estimator(model):
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model

def transform_for_final_estimator(model, X):
    """
    For Pipeline model: apply preprocessing steps only, then return transformed matrix.
    For non-Pipeline model: return X directly.
    """
    if hasattr(model, "steps") and len(model.steps) > 1:
        transformer = model[:-1]
        return transformer.transform(X)
    return X

def get_transformed_feature_names(model, selected_features):
    """
    Get feature names after preprocessing. This is needed because the final
    RandomForest estimator sees transformed features, not raw input features.
    """
    if not hasattr(model, "named_steps"):
        return selected_features

    if "preprocessor" not in model.named_steps:
        return selected_features

    preprocessor = model.named_steps["preprocessor"]

    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        pass

    transformed_names = []

    if hasattr(preprocessor, "transformers_"):
        for name, transformer, columns in preprocessor.transformers_:
            if transformer == "drop":
                continue

            if isinstance(columns, str):
                columns = [columns]

            columns = list(columns)

            if transformer == "passthrough":
                transformed_names.extend(columns)
            elif hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(columns)
                    transformed_names.extend([str(x) for x in names])
                except Exception:
                    transformed_names.extend(columns)
            else:
                transformed_names.extend(columns)

    if len(transformed_names) == 0:
        return selected_features

    return transformed_names

def map_transformed_feature_to_original(transformed_feature, selected_features):
    """
    Map transformed feature names back to original clinical variables.
    For example:
    Alcohol_consumption_1 -> Alcohol_consumption
    cat__HF_HPI_1 -> HF_HPI
    """
    clean_name = str(transformed_feature)

    if "__" in clean_name:
        clean_name = clean_name.split("__", 1)[1]

    for feature in sorted(selected_features, key=len, reverse=True):
        if clean_name == feature:
            return feature
        if clean_name.startswith(feature + "_"):
            return feature
        if clean_name.startswith(feature + "="):
            return feature

    return clean_name

def aggregate_explanation_to_original_features(df, value_col, selected_features):
    """
    Aggregate one-hot/transformed features back to original variables.
    """
    df = df.copy()
    df["Original feature"] = df["Feature"].apply(
        lambda x: map_transformed_feature_to_original(x, selected_features)
    )
    df["Display name"] = df["Original feature"].map(
        lambda x: FEATURE_LABELS.get(x, x)
    )

    if value_col == "SHAP value":
        grouped = (
            df.groupby(["Original feature", "Display name"], as_index=False)
            .agg({"SHAP value": "sum"})
        )
        grouped["Absolute value"] = grouped["SHAP value"].abs()
        grouped = grouped.sort_values("Absolute value", ascending=False)
        return grouped

    grouped = (
        df.groupby(["Original feature", "Display name"], as_index=False)
        .agg({value_col: "sum"})
    )
    grouped = grouped.sort_values(value_col, ascending=False)
    return grouped

def show_feature_importance_fallback(model, selected_features):
    final_estimator = get_final_estimator(model)

    if not hasattr(final_estimator, "feature_importances_"):
        st.warning("Model-derived feature importance is not available for the current model.")
        return

    importances = final_estimator.feature_importances_
    transformed_feature_names = get_transformed_feature_names(model, selected_features)

    if len(importances) != len(transformed_feature_names):
        st.warning(
            "Feature importance could not be displayed because the transformed feature number "
            "does not match the model feature importance length."
        )
        return

    importance_df = pd.DataFrame({
        "Feature": transformed_feature_names,
        "Importance": importances,
    })

    importance_df = aggregate_explanation_to_original_features(
        importance_df,
        "Importance",
        selected_features,
    )

    top_df = importance_df.head(10).copy()

    st.markdown("#### Top Model Features")
    st.dataframe(
        top_df[["Display name", "Importance"]],
        use_container_width=True,
        height=330,
    )

    plot_df = top_df.sort_values("Importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["Display name"], plot_df["Importance"], color="#4C78A8")
    ax.set_xlabel("Feature importance")
    ax.set_title("Top Model Features")
    st.pyplot(fig)

def show_model_explanation(model, X_input, selected_features, shap_background=None):
    st.subheader("Model Explanation")

    st.markdown(
        "This section shows the main variables contributing to the model prediction. "
        "Positive SHAP values indicate variables increasing the predicted risk; "
        "negative SHAP values indicate variables decreasing the predicted risk."
    )

    final_estimator = get_final_estimator(model)

    if not SHAP_AVAILABLE:
        st.info("The SHAP package is not available. Displaying model-derived feature importance instead.")
        show_feature_importance_fallback(model, selected_features)
        return

    if shap_background is None:
        st.info("SHAP background data were not found. Displaying model-derived feature importance instead.")
        show_feature_importance_fallback(model, selected_features)
        return

    try:
        X_input = X_input[selected_features].copy()

        if not isinstance(shap_background, pd.DataFrame):
            shap_background = pd.DataFrame(shap_background)

        shap_background = shap_background.copy()

        missing_cols = [c for c in selected_features if c not in shap_background.columns]
        if missing_cols:
            st.warning(
                "SHAP background data do not contain all required model input columns. "
                "Displaying model-derived feature importance instead."
            )
            with st.expander("Show missing SHAP background columns", expanded=False):
                st.write(missing_cols)
            show_feature_importance_fallback(model, selected_features)
            return

        shap_background = shap_background[selected_features]

        X_bg_transformed = transform_for_final_estimator(model, shap_background)
        X_input_transformed = transform_for_final_estimator(model, X_input)

        transformed_feature_names = get_transformed_feature_names(model, selected_features)

        if X_input_transformed.shape[1] != len(transformed_feature_names):
            transformed_feature_names = [
                f"Feature_{i + 1}" for i in range(X_input_transformed.shape[1])
            ]

        explainer = shap.TreeExplainer(final_estimator, data=X_bg_transformed)
        shap_values = explainer.shap_values(X_input_transformed)

        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values

        if len(shap_values_class1.shape) == 3:
            shap_values_class1 = shap_values_class1[:, :, 1]

        shap_df = pd.DataFrame({
            "Feature": transformed_feature_names,
            "SHAP value": shap_values_class1[0],
        })

        shap_df = aggregate_explanation_to_original_features(
            shap_df,
            "SHAP value",
            selected_features,
        )

        shap_df["Direction"] = shap_df["SHAP value"].apply(
            lambda x: "Increases predicted risk" if x > 0 else "Decreases predicted risk"
        )

        top_df = shap_df.head(10).copy()

        st.markdown("#### Top Individual Risk Contributors")
        st.dataframe(
            top_df[["Display name", "SHAP value", "Direction"]],
            use_container_width=True,
            height=360,
        )

        plot_df = top_df.sort_values("SHAP value")

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#D9534F" if v > 0 else "#5BC0DE" for v in plot_df["SHAP value"]]
        ax.barh(plot_df["Display name"], plot_df["SHAP value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title("Top Contributors to Individual Prediction")
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation could not be generated. Displaying model-derived feature importance instead.")
        show_feature_importance_fallback(model, selected_features)

        with st.expander("Show SHAP error details", expanded=False):
            st.exception(e)

def show_prediction_results(prob, input_data):
    risk_info = get_risk_group(prob)
    risk_percent = prob * 100

    st.markdown("---")

    st.markdown(
        """
        <div style="background-color:#e8f6ef; padding:22px; border-radius:10px; margin-bottom:25px;">
            <h1 style="margin:0; color:#2f3542;">Prediction Results</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Risk Probability")
        st.markdown("Predicted 1-year post-discharge cardiac mortality risk")
        st.markdown(
            f"<h1 style='font-size:46px;'>{risk_percent:.2f}%</h1>",
            unsafe_allow_html=True,
        )

        st.progress(min(float(prob), 1.0))
        st.caption(f"Model-derived probability: {risk_percent:.2f}%")

    with col2:
        st.subheader("Clinical Interpretation")

        st.markdown(
            f"""
            <div style="
                background-color:{risk_info['color']};
                border-left:8px solid {risk_info['border']};
                padding:18px;
                border-radius:10px;
                margin-bottom:18px;
            ">
                <h3 style="margin:0; color:#2f3542;">{risk_info['short_name']}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**{risk_info['description']}**")

        if risk_info["name"] == "High risk group":
            st.markdown(
                """
                - Classified as the **high-risk group**.
                - Recommend further clinical review and management optimization.
                - Consider intensified follow-up and closer monitoring after discharge.
                """
            )

        elif risk_info["name"] == "Intermediate risk group":
            st.markdown(
                """
                - Classified as the **intermediate-risk group**.
                - Review clinical risk factors and modifiable predictors.
                - Individualize follow-up intensity and secondary prevention.
                """
            )

        else:
            st.markdown(
                """
                - Classified as the **low-risk group**.
                - Standard follow-up may be appropriate if clinically consistent.
                - Routine secondary prevention remains necessary.
                """
            )

    st.subheader("Clinical Parameters Summary")

    summary_rows = []
    for feat in selected_features:
        if feat not in input_data:
            continue

        value = input_data[feat]
        display_name = FEATURE_LABELS.get(feat, feat)

        if feat in CATEGORICAL_OPTIONS:
            display_value = CATEGORICAL_OPTIONS[feat].get(value, value)
        elif feat in BINNED_OPTIONS:
            display_value = BINNED_OPTIONS[feat].get(value, value)
        else:
            display_value = value

        summary_rows.append({
            "Category": get_feature_group(feat),
            "Parameter": display_name,
            "Value": display_value,
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, height=420)

    st.subheader("Modifiable Risk Factor Suggestions")

    modifiable_df = get_modifiable_suggestions(input_data)

    if modifiable_df.empty:
        st.info(
            "No major modifiable risk factor alert was triggered based on the current inputs. "
            "Standard secondary prevention and routine follow-up remain necessary."
        )
    else:
        st.markdown(
            "The following modifiable domains may be considered during post-discharge follow-up and secondary prevention planning."
        )
        st.dataframe(modifiable_df, use_container_width=True, height=260)

st.markdown("---")

if st.button("Calculate Cardiovascular Risk"):
    try:
        X_input = pd.DataFrame([input_data])

        missing_input_cols = [c for c in selected_features if c not in X_input.columns]
        if missing_input_cols:
            st.error("Prediction failed because required input columns are missing.")
            with st.expander("Show missing input columns", expanded=True):
                st.write(missing_input_cols)
            st.stop()

        X_input = X_input[selected_features]

        with st.expander("Show model input data", expanded=False):
            st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)

        prob = model.predict_proba(X_input)[:, 1][0]

        show_prediction_results(prob, input_data)

        st.markdown("---")
        show_model_explanation(model, X_input, selected_features, shap_background)

    except Exception as e:
        st.error(
            "Prediction failed. Please check whether the input variables, variable names, "
            "and coding rules are consistent with the training data."
        )
        st.exception(e)

st.sidebar.markdown("## Clinical Context")

st.sidebar.markdown(
    """
### Risk Categories

- **High risk**: probability > **5.45%**
- **Intermediate risk**: **1.64% < probability <= 5.45%**
- **Low risk**: probability <= **1.64%**
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### Intended Use

For **AMI survivors** who:

- survived hospitalization;
- underwent coronary angiography;
- are assessed at discharge or shortly before discharge.

The model estimates **1-year post-discharge cardiac mortality risk**.
"""
)

st.sidebar.markdown(
    """
### Clinical Application

Use the result to support:

- discharge risk stratification;
- follow-up intensity planning;
- secondary prevention optimization;
- closer monitoring in selected high-risk patients.
"""
)

st.sidebar.markdown(
    """
### Note

High-risk results should prompt **clinical review and management optimization**, not definitive prognostic judgment.
"""
)

st.sidebar.markdown("---")
st.sidebar.caption("Clinical Decision Support Tool v2.1")
st.sidebar.caption("For medical professional use only")