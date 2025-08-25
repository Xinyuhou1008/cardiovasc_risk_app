import sys
import streamlit as st

# 检查关键依赖
required_packages = ['shap', 'pandas', 'lightgbm', 'sklearn', 'matplotlib']
missing = [pkg for pkg in required_packages if pkg not in sys.modules]

if missing:
    st.error(f"关键包导入失败: {', '.join(missing)}")
    st.stop()

import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

# 定义风险阈值常量
HIGH_RISK_THRESHOLD = 0.3252  # 32.52%
LOW_RISK_THRESHOLD = 0.1472   # 14.72%

# Load the saved model pipeline
with open('LightGBM_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

# Set the title for the Streamlit app
st.title("Cardiovascular Risk Prediction Model")

# User input features
st.header("Enter Clinical Parameters:")

# Input fields - continuous variables
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value=18, max_value=120, value=60)
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
    temperature = st.number_input('Environment Temperature (°C)', min_value=-40.0, max_value=50.0, value=15.0)
    gensini_score = st.number_input('Gensini Score', min_value=0.0, max_value=500.0, value=30.0)
    weighted_value_of_diffuse_lesion = st.number_input('Weighted Value of Diffuse Lesion', min_value=0.0, max_value=30.0, value=2.5)
    
with col2:
    pm25 = st.number_input('PM2.5 Pollution (μg/m³)', min_value=0.0, max_value=1000.0, value=45.0)
    lvedd = st.number_input('Left Ventricular End Diastolic Diameter (mm)', min_value=20.0, max_value=100.0, value=50.0)
    ef = st.number_input('Ejection Fraction (%)', min_value=20.0, max_value=100.0, value=55.0)

# Input fields - categorical variables with descriptive options
st.subheader("Clinical Laboratory Parameters:")

col1, col2, col3 = st.columns(3)
with col1:
    # Cardiac troponin I (cTnI) - 注意：编码从1开始
    ctni_options = {
        1: "Within 10 times ULN",
        2: "Within 100 times ULN",；
        3: "Within 1,000 times ULN",
        4: "Within 10,000 times ULN",
        5: "More than 10,000 times ULN"
    }
    ctni = st.selectbox('Cardiac Troponin I (cTnI)', options=list(ctni_options.keys()), 
                         format_func=lambda x: ctni_options[x], index=0)
    
    # LDL Cholesterol (LDL_C) - 编码从1开始
    ldl_c_options = {
        1: "<1.40 mmol/L",
        2: "1.40-1.79 mmol/L",
        3: "1.80-2.59 mmol/L",
        4: "2.60-3.39 mmol/L",
        5: "3.40-4.09 mmol/L",
        6: "4.10-4.89 mmol/L",
        7: "≥4.90 mmol/L"
    }
    ldl_c = st.selectbox('LDL Cholesterol', options=list(ldl_c_options.keys()), 
                         format_func=lambda x: ldl_c_options[x], index=3)
    
    # Glucose (GLU) - 编码从0开始
    glu_options = {
        0: "Normal",
        1: ">Normal and <7 mmol/L",
        2: "≥7 and <16.7 mmol/L",
        3: "≥16.7 mmol/L"
    }
    glu = st.selectbox('Glucose', options=list(glu_options.keys()), 
                       format_func=lambda x: glu_options[x], index=0)
    
    # Serum Creatinine (Scr) - 编码从0开始
    scr_options = {
        0: "Normal",
        1: "1-3 times ULN",
        2: ">3 times ULN"
    }
    scr = st.selectbox('Serum Creatinine (Scr)', options=list(scr_options.keys()), 
                       format_func=lambda x: scr_options[x], index=0)

with col2:
    # B-type Natriuretic Peptide (BNP) - 编码从1开始
    bnp_options = {
        1: "Normal",
        2: ">Normal and ≤1000 pg/ml",
        3: "1001-5000 pg/ml",
        4: ">5000 pg/ml"
    }
    bnp = st.selectbox('B-type Natriuretic Peptide (BNP)', options=list(bnp_options.keys()), 
                       format_func=lambda x: bnp_options[x], index=0)
    
    # D-dimer - 编码从0开始
    d_dimmer_options = {
        0: "Normal",
        1: ">Normal and <500 ng/mL",
        2: "≥500 ng/mL"
    }
    d_dimmer = st.selectbox('D-dimer', options=list(d_dimmer_options.keys()), 
                            format_func=lambda x: d_dimmer_options[x], index=0)
    
    # Smoking status - 编码从1开始
    smoking_options = {
        1: "Non-smoker",
        2: "Former smoker",
        3: "Current smoker"
    }
    smoking = st.selectbox('Smoking Status', options=list(smoking_options.keys()), 
                           format_func=lambda x: smoking_options[x], index=0)

# Create DataFrame
input_data = {
    'Smoking': smoking,
    'cTnI': ctni,
    'LDL_C': ldl_c,
    'BNP': bnp,
    'D_dimmer': d_dimmer,
    'GLU': glu,
    'Scr': scr,
    'Gensini_score': gensini_score,
    'Weighted_value_of_diffuse_lesion': weighted_value_of_diffuse_lesion,
    'Age': age,
    'BMI': bmi,
    'Temperature': temperature,
    'PM2.5': pm25,
    'LVEDD': lvedd,
    'EF': ef
}
df = pd.DataFrame([input_data])

# Prediction button
if st.button('Calculate Cardiovascular Risk'):
    # 定义模型期望的37个特征
    final_features = [
        'num__Gensini_score',
        'num__Weighted_value_of_diffuse_lesion',
        'num__Age',
        'num__BMI',
        'num__Temperature',
        'num__PM2.5',
        'num__LVEDD',
        'num__EF',
        'cat__Smoking_1',
        'cat__Smoking_2',
        'cat__Smoking_3',
        'cat__cTnI_1',
        'cat__cTnI_2',
        'cat__cTnI_3',
        'cat__cTnI_4',
        'cat__cTnI_5',
        'cat__LDL_C_1',
        'cat__LDL_C_2',
        'cat__LDL_C_3',
        'cat__LDL_C_4',
        'cat__LDL_C_5',
        'cat__LDL_C_6',
        'cat__LDL_C_7',
        'cat__BNP_1',
        'cat__BNP_2',
        'cat__BNP_3',
        'cat__BNP_4',
        'cat__D_dimmer_0',
        'cat__D_dimmer_1',
        'cat__D_dimmer_2',
        'cat__GLU_0',
        'cat__GLU_1',
        'cat__GLU_2',
        'cat__GLU_3',
        'cat__Scr_0',
        'cat__Scr_1',
        'cat__Scr_2'
    ]

    # Initialize a series for the final data
    final_series = pd.Series(0, index=final_features)

    # Fill in continuous variables
    final_series['num__Gensini_score'] = gensini_score
    final_series['num__Weighted_value_of_diffuse_lesion'] = weighted_value_of_diffuse_lesion
    final_series['num__Age'] = age
    final_series['num__BMI'] = bmi
    final_series['num__Temperature'] = temperature
    final_series['num__PM2.5'] = pm25
    final_series['num__LVEDD'] = lvedd
    final_series['num__EF'] = ef

    # Fill in categorical variables with one-hot encoding
    # 注意：根据变量起始点正确编码
    final_series['cat__Smoking_1'] = int(smoking == 1)
    final_series['cat__Smoking_2'] = int(smoking == 2)
    final_series['cat__Smoking_3'] = int(smoking == 3)
    
    final_series['cat__cTnI_1'] = int(ctni == 1)
    final_series['cat__cTnI_2'] = int(ctni == 2)
    final_series['cat__cTnI_3'] = int(ctni == 3)
    final_series['cat__cTnI_4'] = int(ctni == 4)
    final_series['cat__cTnI_5'] = int(ctni == 5)
    
    final_series['cat__LDL_C_1'] = int(ldl_c == 1)
    final_series['cat__LDL_C_2'] = int(ldl_c == 2)
    final_series['cat__LDL_C_3'] = int(ldl_c == 3)
    final_series['cat__LDL_C_4'] = int(ldl_c == 4)
    final_series['cat__LDL_C_5'] = int(ldl_c == 5)
    final_series['cat__LDL_C_6'] = int(ldl_c == 6)
    final_series['cat__LDL_C_7'] = int(ldl_c == 7)
    
    final_series['cat__BNP_1'] = int(bnp == 1)
    final_series['cat__BNP_2'] = int(bnp == 2)
    final_series['cat__BNP_3'] = int(bnp == 3)
    final_series['cat__BNP_4'] = int(bnp == 4)
    
    final_series['cat__D_dimmer_0'] = int(d_dimmer == 0)
    final_series['cat__D_dimmer_1'] = int(d_dimmer == 1)
    final_series['cat__D_dimmer_2'] = int(d_dimmer == 2)
    
    final_series['cat__GLU_0'] = int(glu == 0)
    final_series['cat__GLU_1'] = int(glu == 1)
    final_series['cat__GLU_2'] = int(glu == 2)
    final_series['cat__GLU_3'] = int(glu == 3)
    
    final_series['cat__Scr_0'] = int(scr == 0)
    final_series['cat__Scr_1'] = int(scr == 1)
    final_series['cat__Scr_2'] = int(scr == 2)

    # Convert to DataFrame for prediction
    final_df = pd.DataFrame([final_series.values], columns=final_series.index)

    # 调试信息：显示所有特征值
    if st.checkbox('Show model input features'):
        st.subheader("Model Input Features")
        st.dataframe(final_df.T.rename(columns={0: 'Value'}))
    
    # Make prediction
    prediction = model.named_steps['classifier'].predict_proba(final_df)
    st.success("### Prediction Results")
    
    risk_percent = prediction[0][1] * 100
    
    # Display probabilities with visual indicators
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Probability")
        st.metric(label="Cardiovascular Risk Probability", value=f"{risk_percent:.1f}%")
        
        # 使用颜色增强的风险指示器
        fig, ax = plt.subplots(figsize=(10, 1))
        ax.barh(['Risk'], [risk_percent], color='#e74c3c' if risk_percent > 50 else '#f0f0f0')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Clinical Interpretation")
        if risk_percent > HIGH_RISK_THRESHOLD * 100:
            st.error("##### 🚨 High Risk Alert")
            st.markdown(f"""
            - **Probability > {HIGH_RISK_THRESHOLD*100:.1f}%**
            - High likelihood of cardiovascular event
            - Immediate clinical evaluation recommended
            - Consider urgent interventions
            """)
        elif risk_percent > LOW_RISK_THRESHOLD * 100:
            st.warning("##### ⚠️ Moderate Risk")
            st.markdown(f"""
            - **Probability > {LOW_RISK_THRESHOLD*100:.1f}% and ≤ {HIGH_RISK_THRESHOLD*100:.1f}%**
            - Moderate risk of cardiovascular event
            - Lifestyle modifications advised
            - Regular medical follow-up needed
            """)
        else:
            st.success("##### ✅ Low Risk")
            st.markdown(f"""
            - **Probability ≤ {LOW_RISK_THRESHOLD*100:.1f}%**
            - Low risk of cardiovascular event
            - Maintain healthy lifestyle
            - Annual cardiovascular screening recommended
            """)

    # SHAP explainer
    try:
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer(final_df)
        values = shap_values.values[0]
    except Exception as e:
        st.warning(f"Feature contribution analysis could not be performed: {str(e)}")
        st.stop()

    # Human-readable feature name mapping
    human_readable_names = {
        'num__Gensini_score': 'Gensini Score',
        'num__Weighted_value_of_diffuse_lesion': 'Diffuse Lesion Weight',
        'num__Age': 'Age',
        'num__BMI': 'Body Mass Index (BMI)',
        'num__Temperature': 'Environmental Temperature',
        'num__PM2.5': 'PM2.5 Exposure',
        'num__LVEDD': 'Left Ventricular End Diastolic Diameter',
        'num__EF': 'Ejection Fraction',
        'cat__Smoking_1': 'Non-smoker',
        'cat__Smoking_2': 'Former smoker',
        'cat__Smoking_3': 'Current smoker',
        'cat__cTnI_1': 'cTnI: Within 10x ULN',
        'cat__cTnI_2': 'cTnI: Within 100x ULN',
        'cat__cTnI_3': 'cTnI: Within 1000x ULN',
        'cat__cTnI_4': 'cTnI: Within 10000x ULN',
        'cat__cTnI_5': 'cTnI: >10000x ULN',
        'cat__LDL_C_1': 'LDL-C: <1.40 mmol/L',
        'cat__LDL_C_2': 'LDL-C: 1.40-1.79 mmol/L',
        'cat__LDL_C_3': 'LDL-C: 1.80-2.59 mmol/L',
        'cat__LDL_C_4': 'LDL-C: 2.60-3.39 mmol/L',
        'cat__LDL_C_5': 'LDL-C: 3.40-4.09 mmol/L',
        'cat__LDL_C_6': 'LDL-C: 4.10-4.89 mmol/L',
        'cat__LDL_C_7': 'LDL-C: ≥4.90 mmol/L',
        'cat__BNP_1': 'BNP: Normal',
        'cat__BNP_2': 'BNP: >Normal and ≤1000 pg/ml',
        'cat__BNP_3': 'BNP: 1001-5000 pg/ml',
        'cat__BNP_4': 'BNP: >5000 pg/ml',
        'cat__D_dimmer_0': 'D-dimer: Normal',
        'cat__D_dimmer_1': 'D-dimer: >Normal and <500 ng/mL',
        'cat__D_dimmer_2': 'D-dimer: ≥500 ng/mL',
        'cat__GLU_0': 'Glucose: Normal',
        'cat__GLU_1': 'Glucose: >Normal and <7 mmol/L',
        'cat__GLU_2': 'Glucose: ≥7 and <16.7 mmol/L',
        'cat__GLU_3': 'Glucose: ≥16.7 mmol/L',
        'cat__Scr_0': 'Creatinine: Normal',
        'cat__Scr_1': 'Creatinine: 1-3x ULN',
        'cat__Scr_2': 'Creatinine: >3x ULN',
    }
    
    # 修正编码后的临床参数值映射
    clinical_params = {
        'Gensini Score': gensini_score,
        'Diffuse Lesion Weight': weighted_value_of_diffuse_lesion,
        'Age': age,
        'BMI': bmi,
        'Environmental Temperature': temperature,
        'PM2.5 Exposure': pm25,
        'LVEDD': lvedd,
        'Ejection Fraction': ef,
        'Smoking': smoking_options[smoking],
        'cTnI': ctni_options[ctni],
        'LDL Cholesterol': ldl_c_options[ldl_c],
        'BNP': bnp_options[bnp],
        'D-dimer': d_dimmer_options[d_dimmer],
        'Glucose': glu_options[glu],
        'Creatinine': scr_options[scr]
    }
    
    # Create summary table
    st.subheader("Clinical Parameters Summary")
    param_df = pd.DataFrame(list(clinical_params.items()), columns=['Parameter', 'Value'])
    st.dataframe(param_df, use_container_width=True, height=400)
    
    # Display feature contributions as bar chart
    try:
        # 收集特征贡献度
        feature_contributions = []
        feature_names = []
        for i, col in enumerate(final_features):
            feature_name = human_readable_names.get(col, col)
            feature_contributions.append(values[i])
            feature_names.append(feature_name)
        
        # 按贡献度绝对值排序
        sorted_idx = np.argsort(np.abs(feature_contributions))[::-1]
        sorted_contributions = np.array(feature_contributions)[sorted_idx]
        sorted_names = np.array(feature_names)[sorted_idx]
        
        # 创建水平柱状图
        st.subheader('Top Risk Contributors (SHAP Values)')
        fig, ax = plt.subplots(figsize=(12, 8))
        num_to_show = min(20, len(sorted_contributions))
        
        # 使用颜色区分正负贡献
        colors = ['#e74c3c' if x > 0 else '#3498db' for x in sorted_contributions[:num_to_show]]
        
        bars = ax.barh(
            sorted_names[:num_to_show][::-1], 
            sorted_contributions[:num_to_show][::-1], 
            color=colors, 
            alpha=0.8
        )
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            value = bar.get_width()
            ax.text(
                value + (0.001 if value > 0 else -0.01), 
                i, 
                f"{value:.4f}", 
                color='black',
                fontsize=10,
                va='center'
            )
        
        ax.set_title('Impact on Cardiovascular Risk Prediction')
        ax.set_xlabel('SHAP Value (Positive = Higher Risk, Negative = Lower Risk)')
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error generating feature contributions: {str(e)}")

# Add clinical context
st.sidebar.markdown("## Clinical Context")
st.sidebar.markdown(f"""
**Key Risk Categories**:
- **High Risk (>{HIGH_RISK_THRESHOLD*100:.1f}%)**: Requires immediate clinical evaluation
- **Moderate Risk (>{LOW_RISK_THRESHOLD*100:.1f}% ≤ {HIGH_RISK_THRESHOLD*100:.1f}%)**: Needs preventive interventions
- **Low Risk (≤{LOW_RISK_THRESHOLD*100:.1f}%)**: Continue routine care

**Important Note**:
- This model can predict high-risk probabilities up to 80% in validated populations
- Ensure clinical parameters are entered accurately for reliable predictions
- The model is used for predicting cardiac mortality in patients with acute myocardial infarction
""")

# Add disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("Clinical Decision Support Tool v2.1")
st.sidebar.caption("For medical professional use only")
st.sidebar.caption("Supported by the National Key R&D Program of China(2016YFC1301105)")
st.sidebar.caption("Supported by the Key R&D Program of Heilongjiang Province(2022ZX01A28)")