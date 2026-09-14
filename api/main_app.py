import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------------------------------------------------------
# 1. DYNAMIC PATH RESOLUTION (Fixes ModuleNotFoundError: No module named 'src')
# -----------------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent  # Points to E:\PROJECTS_FILE\CCP

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional import if available, fallback to internal engine if missing
try:
    from src.models.predict import ChurnPredictor
except ModuleNotFoundError:
    ChurnPredictor = None

# -----------------------------------------------------------------------------
# 2. PIPELINE DEPENDENCY CLASSES (Fixed syntax/indentation error)
# -----------------------------------------------------------------------------
class UnknownMarkerCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, markers=None):
        self.markers = markers or ["Unknown", "?", "NA", "null", "none"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        for col in X_out.columns:
            X_out[col] = X_out[col].replace(self.markers, np.nan)
        return X_out

# -----------------------------------------------------------------------------
# 3. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.0rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-status { background-color: #E2E8F0; color: #334155; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; margin-bottom: 1.5rem; }
    .risk-badge-high { background-color: #FEE2E2; color: #991B1B; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 700; text-align: center; }
    .risk-badge-med { background-color: #FFEDD5; color: #9A3412; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 700; text-align: center; }
    .risk-badge-low { background-color: #DCFCE7; color: #166534; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 700; text-align: center; }
    .rec-card { border-left: 4px solid #3B82F6; background-color: #F8FAFC; padding: 1rem; border-radius: 4px; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. PRODUCTION ENGINE & PIPELINE LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_production_environment():
    """Loads actual YAML config, trained model pipeline, and real raw data."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model_path": "models/churn_pipeline.joblib",
            "raw_data_path": "data/raw/cell2celltrain.csv"
        }

    rel_model_path = config.get("model_path", "models/churn_pipeline.joblib")
    rel_data_path = config.get("raw_data_path", "data/raw/cell2celltrain.csv")

    model_path = PROJECT_ROOT / rel_model_path
    raw_data_path = PROJECT_ROOT / rel_data_path

    if not model_path.exists():
        st.error(f"❌ Model binary not found at: `{model_path}`")
        st.stop()
    if not raw_data_path.exists():
        st.error(f"❌ Raw dataset missing at: `{raw_data_path}`")
        st.stop()

    pipeline = joblib.load(model_path)
    df_real = pd.read_csv(raw_data_path)
    df_real.columns = df_real.columns.str.lower()

    return config, pipeline, df_real

config, pipeline, real_df = load_production_environment()

def predict_real_customer(input_dict):
    """Processes customer input via trained pipeline and predicts churn probability."""
    df_input = pd.DataFrame([input_dict])
    df_input.columns = df_input.columns.str.lower()
    
    if hasattr(pipeline, "feature_names_in_"):
        required_cols = pipeline.feature_names_in_
    else:
        required_cols = pipeline.named_steps['preprocessor'].feature_names_in_

    for col in required_cols:
        if col not in df_input.columns:
            df_input[col] = np.nan

    df_input = df_input[required_cols]
    
    prob = pipeline.predict_proba(df_input)[:, 1][0]
    return prob, df_input

# -----------------------------------------------------------------------------
# 5. DASHBOARD LAYOUT & UI COMPONENTS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">CUSTOMER CHURN INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production ML Infrastructure connected directly to production model pipeline.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="badge-status">Production Pipeline Active | Dataset: <b>{os.path.basename(config.get("raw_data_path", "cell2celltrain.csv"))}</b></div>', unsafe_allow_html=True)

# KPI Metrics using actual dataset statistics
total_cust = len(real_df)
avg_rev = real_df['monthlyrevenue'].mean() if 'monthlyrevenue' in real_df.columns else 58.8

target_col = config.get("target_column", "churn").lower()
if target_col in real_df.columns:
    actual_churn_rate = (real_df[target_col].astype(str).str.lower().isin(['yes', '1'])).mean()
else:
    actual_churn_rate = 0.288

high_risk_count = int(total_cust * actual_churn_rate)
revenue_at_risk = high_risk_count * avg_rev

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Total Real Records", value=f"{total_cust:,}", delta="Live Production Data")
with kpi2:
    st.metric(label="High-Risk Customers", value=f"{high_risk_count:,}", delta=f"{actual_churn_rate*100:.1f}% Base Churn", delta_color="inverse")
with kpi3:
    st.metric(label="Revenue Exposure ($)", value=f"${revenue_at_risk:,.2f}", delta="Monthly Risk", delta_color="inverse")
with kpi4:
    st.metric(label="Average Monthly Bill", value=f"${avg_rev:.2f}", delta="Per Account")

st.divider()

# Interactive Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Single Customer Analysis", 
    "🧪 What-If Scenario Simulator", 
    "📁 Batch Real Data Analysis"
])

# TAB 1: SINGLE CUSTOMER ANALYSIS
with tab1:
    col_input, col_pred = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("📋 Production Feature Inputs")
        
        c1, c2 = st.columns(2)
        with c1:
            months_in_service = st.slider("Months in Service", 1, 72, 12, key="months_in_service")
            monthly_revenue = st.slider("Monthly Revenue ($)", 10.0, 300.0, float(avg_rev), key="monthly_revenue")
            monthly_minutes = st.slider("Monthly Minutes", 0, 3000, 450, key="monthly_minutes")
            overage_minutes = st.slider("Overage Minutes", 0, 500, 25, key="overage_minutes")
        
        with c2:
            total_recurring_charge = st.slider("Total Recurring Charge ($)", 10.0, 150.0, 45.0, key="total_recurring_charge")
            perc_change_minutes = st.slider("% Change in Minutes", -200, 200, -15, key="perc_change_minutes")
            retention_calls = st.selectbox("Retention Calls Made", [0, 1, 2, 3], index=0, key="retention_calls")
            credit_rating = st.selectbox("Credit Rating", ['1-Highest', '2-High', '3-Good', '4-Medium', '5-Low'], index=1, key="credit_rating")

        c3, c4 = st.columns(2)
        with c3:
            made_retention_call = st.radio("Call to Retention Team?", ['No', 'Yes'], index=0, key="made_retention_call")
        with c4:
            handset_web = st.radio("Handset Web Capable?", ['Yes', 'No'], index=0, key="handset_web")

        input_dict = {
            'monthsinservice': months_in_service,
            'monthlyrevenue': monthly_revenue,
            'monthlyminutes': monthly_minutes,
            'overageminutes': overage_minutes,
            'totalrecurringcharge': total_recurring_charge,
            'percchangeminutes': perc_change_minutes,
            'retentioncalls': retention_calls,
            'creditrating': credit_rating,
            'madecalltoretentionteam': made_retention_call,
            'handsetwebcapable': handset_web
        }

    with col_pred:
        st.subheader("⚡ Model Prediction & Action Plan")
        
        prob, processed_df = predict_real_customer(input_dict)
        prob_pct = prob * 100
        
        st.markdown(f"### Churn Probability: **{prob_pct:.1f}%**")
        st.progress(float(prob))
        
        if prob_pct >= 65:
            st.markdown('<div class="risk-badge-high">HIGH CHURN RISK</div>', unsafe_allow_html=True)
            rec_text = "🚨 **High Risk Intervention:** Target with 15% discount contract extension and priority support escalation."
        elif prob_pct >= 35:
            st.markdown('<div class="risk-badge-med">MEDIUM CHURN RISK</div>', unsafe_allow_html=True)
            rec_text = "⚠️ **Medium Risk Action:** Offer targeted data add-on package or promotional equipment upgrade."
        else:
            st.markdown('<div class="risk-badge-low">LOW CHURN RISK</div>', unsafe_allow_html=True)
            rec_text = "✅ **Low Risk:** Account stable. Suitable for standard cross-sell and upsell campaigns."

        st.markdown(f'<div class="rec-card">{rec_text}</div>', unsafe_allow_html=True)

# TAB 2: WHAT-IF SCENARIO SIMULATOR
with tab2:
    st.subheader("🧪 Strategic Intervention Simulator")
    
    col_base, col_sim = st.columns(2, gap="large")
    
    with col_base:
        st.markdown("#### Baseline State")
        base_prob, _ = predict_real_customer(input_dict)
        st.metric("Baseline Churn Risk", f"{base_prob*100:.1f}%")
        st.json(input_dict, expanded=False)

    with col_sim:
        st.markdown("#### Strategy Adjustments")
        sim_revenue = st.slider("Simulated Monthly Revenue ($)", 10.0, 300.0, float(input_dict['monthlyrevenue']), key="sim_rev")
        sim_change_min = st.slider("Simulated % Change in Minutes", -200, 200, int(input_dict['percchangeminutes']), key="sim_min")
        sim_retention_calls = st.radio("Simulated Retention Action Call?", ['No', 'Yes'], index=0 if input_dict['madecalltoretentionteam'] == 'No' else 1, key="sim_ret")

        sim_input = input_dict.copy()
        sim_input['monthlyrevenue'] = sim_revenue
        sim_input['percchangeminutes'] = sim_change_min
        sim_input['madecalltoretentionteam'] = sim_retention_calls

        sim_prob, _ = predict_real_customer(sim_input)
        risk_delta = (sim_prob - base_prob) * 100

        st.markdown("#### Scenario Impact")
        st.metric(
            label="Simulated Churn Risk", 
            value=f"{sim_prob*100:.1f}%", 
            delta=f"{risk_delta:.1f}% Risk Reduction" if risk_delta < 0 else f"+{risk_delta:.1f}% Risk Increase",
            delta_color="inverse"
        )

# TAB 3: BATCH REAL DATA ANALYSIS
with tab3:
    st.subheader("📂 Real Dataset Batch Scoring")
    
    uploaded_file = st.file_uploader("Upload External CSV File for Scoring", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_df.columns = batch_df.columns.str.lower()
        st.success("Custom CSV uploaded for batch scoring.")
    else:
        st.info(f"Loaded top records directly from project dataset ('{os.path.basename(config.get('raw_data_path'))}').")
        batch_df = real_df.head(100).copy()

    if st.button("Run Batch Inference"):
        with st.spinner("Executing pipeline predictions across real records..."):
            target_col = config.get("target_column", "churn").lower()
            id_col = config.get("id_column", "customerid").lower()

            inference_input = batch_df.drop(columns=[target_col, id_col], errors='ignore')
            
            probs = pipeline.predict_proba(inference_input)[:, 1]
            
            batch_df['Churn Probability (%)'] = np.round(probs * 100, 1)
            batch_df['Revenue Exposure ($)'] = batch_df['monthlyrevenue'].fillna(50) if 'monthlyrevenue' in batch_df.columns else 50.0

            high_risk_table = batch_df.sort_values(by='Churn Probability (%)', ascending=False)
            
            display_cols = [c for c in [id_col, 'Churn Probability (%)', 'Revenue Exposure ($)', 'monthsinservice', 'monthlyrevenue', 'creditrating'] if c in high_risk_table.columns]

            st.dataframe(
                high_risk_table[display_cols].head(15).style.background_gradient(subset=['Churn Probability (%)'], cmap='Reds'),
                use_container_width=True
            )
            st.success("Batch risk assessment completed successfully!")