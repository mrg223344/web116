import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

# -----------------------------------------------------------------------------
# 1. 页面配置 (必须是第一个 Streamlit 命令)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RVH Risk Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. 自定义 CSS (打造好看的医疗 UI)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* 主背景色微调 */
    .stApp {
        background-color: #f8f9fa;
    }
    /* 标题样式 */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* 结果卡片样式 */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border-left: 5px solid #c62828;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 5px solid #2e7d32;
    }
    /* 医疗免责声明 */
    .disclaimer {
        font-size: 0.8em;
        color: #7f8c8d;
        margin-top: 50px;
        border-top: 1px solid #ddd;
        padding-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. 加载模型
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # 加载模型
        model = joblib.load('xgboost_outcome_model.pkl')
        return model
    except Exception as e:
        st.error(f"无法加载模型文件，请确保 'xgboost_outcome_model.pkl' 在当前目录下。\n错误详情: {e}")
        return None

model = load_model()

# -----------------------------------------------------------------------------
# 4. 侧边栏 - 输入参数
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Patient Clinical Data")
st.sidebar.markdown("Please input the patient's parameters below:")

def user_input_features():
    # 1. 生理指标
    st.sidebar.subheader("Physiological Markers")
    
    # HbA1c (糖化血红蛋白)
    hba1c = st.sidebar.number_input(
        "HbA1c (%)", 
        min_value=3.0, max_value=20.0, value=7.5, step=0.1,
        help="Glycated Hemoglobin level."
    )
    
    # BMI
    bmi = st.sidebar.number_input(
        "BMI (kg/m²)", 
        min_value=10.0, max_value=60.0, value=24.5, step=0.1
    )
    
    # Haemoglobin (血红蛋白)
    haemoglobin = st.sidebar.number_input(
        "Haemoglobin (g/L)", 
        min_value=50.0, max_value=200.0, value=135.0, step=1.0,
        help="Check if your unit is g/L or g/dL. Code assumes input matches training scale."
    )

    st.sidebar.markdown("---")
    
    # 2. 临床病史 (二分类)
    st.sidebar.subheader("Clinical History")
    
    # Active neovascularisation
    active_neo_input = st.sidebar.selectbox(
        "Active Neovascularisation",
        ("No", "Yes"),
        index=0,
        help="Presence of active new blood vessels."
    )
    active_neo = 1 if active_neo_input == "Yes" else 0

    # Hypertension
    htn_input = st.sidebar.selectbox(
        "Hypertension",
        ("No", "Yes"),
        index=1,
        help="History of high blood pressure."
    )
    hypertension = 1 if htn_input == "Yes" else 0
    
    # History of cardiovascular disease
    cvd_input = st.sidebar.selectbox(
        "History of Cardiovascular Disease",
        ("No", "Yes"),
        index=0
    )
    history_cv = 1 if cvd_input == "Yes" else 0

    # -------------------------------------------------------------------------
    # 核心修正：构建 DataFrame
    # 这里的键名 (Key) 已更新为你的模型所需要的全称
    # -------------------------------------------------------------------------
    data = {
        'Haemoglobin': haemoglobin,
        'Active.neovascularisation': active_neo,
        'History.of.cardiovascular.disease': history_cv,
        'HbA1c': hba1c,
        'BMI': bmi,
        'Hypertension': hypertension
    }
    
    features = pd.DataFrame(data, index=[0])
    
    # ⚠️ 强制排序：确保列的顺序与报错信息中的 'mismatch' 列表一致
    # 这样可以防止任何顺序错误
    expected_order = [
        'Haemoglobin', 
        'Active.neovascularisation', 
        'History.of.cardiovascular.disease', 
        'HbA1c', 
        'BMI', 
        'Hypertension'
    ]
    
    # 如果列名有微小拼写错误，这里会报错提醒，便于调试
    try:
        features = features[expected_order]
    except KeyError as e:
        st.error(f"代码内部错误：列名拼写不匹配。详细信息: {e}")
        
    return features

input_df = user_input_features()

# -----------------------------------------------------------------------------
# 5. 主页面内容
# -----------------------------------------------------------------------------

# 标题栏
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Recurrent Vitreous Hemorrhage Predictor")
    st.markdown("### Post-Vitrectomy Risk Assessment")
with col2:
    # 这是一个占位符，你可以放医院logo
    st.write("") 

# 信息提示框 (Requested Note)
st.info("""
    ℹ️ **Target Population:** This tool is designed for **PDR Patients** (Proliferative Diabetic Retinopathy) undergoing vitrectomy.
    It predicts the risk of recurrent hemorrhage based on pre-operative and clinical factors.
""")

# 显示用户输入摘要
with st.expander("Show Input Summary", expanded=False):
    st.dataframe(input_df)

# -----------------------------------------------------------------------------
# 6. 预测逻辑
# -----------------------------------------------------------------------------

if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
    if model:
        try:
            # 预测概率
            prediction_proba = model.predict_proba(input_df)[0][1]
            
            # 转换为百分比
            risk_percent = prediction_proba * 100
            
            st.markdown("---")
            
            # 布局：左侧仪表盘/结果，右侧详细建议
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.subheader("Prediction Result")
                
                # 动态显示结果卡片
                if risk_percent > 50:
                    st.markdown(f"""
                        <div class="result-card high-risk">
                            <h2>High Risk</h2>
                            <h1>{risk_percent:.1f}%</h1>
                            <p>Probability of Recurrence</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card low-risk">
                            <h2>Low Risk</h2>
                            <h1>{risk_percent:.1f}%</h1>
                            <p>Probability of Recurrence</p>
                        </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.subheader("Risk Analysis")
                # 进度条展示
                st.write("Risk Confidence Level:")
                st.progress(int(min(risk_percent, 100))) # 确保不超过100
                
                st.write("**Contributing Factors:**")
                # 简单的解释逻辑
                if input_df['Active.neovascularisation'][0] == 1:
                    st.warning("⚠️ Active Neovascularisation is a significant risk factor.")
                if input_df['HbA1c'][0] > 8.0:
                    st.warning("⚠️ Elevated HbA1c suggests poor glycemic control.")
                if risk_percent < 50:
                    st.success("✅ Patient profile suggests lower likelihood of recurrence.")
        
        except Exception as e:
            st.error(f"预测过程中发生错误: {e}")
            st.write("请检查输入的特征列名是否与模型完全匹配。")

    else:
        st.error("Model not loaded. Please check if the .pkl file exists.")

# -----------------------------------------------------------------------------
# 7. 底部免责声明
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="disclaimer">
        <strong>Medical Disclaimer:</strong> This application is for research and educational purposes only. 
        It involves a machine learning model (XGBoost) and should not be used as the sole basis for clinical diagnosis or treatment decisions. 
        Always consult with a qualified ophthalmologist.
    </div>
""", unsafe_allow_html=True)