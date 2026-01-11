import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Causal Impact AI",
    page_icon="🧠",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def preprocess_data(df, selected_columns):
    """Cleans data: fills NaNs, encodes text to numbers."""
    data = df[selected_columns].copy()
    data = data.dropna()
    
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
            
    return data, encoders

def train_model(data, treatment_col, outcome_col, covariates):
    # CausalForest is best for finding "Heterogeneity" (Segments)
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=6),
        discrete_treatment=True,
        random_state=42
    )
    
    Y = data[outcome_col]
    T = data[treatment_col]
    X = data[covariates]
    
    est.fit(Y, T, X=X)
    return est, X

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV (<50MB)", type="csv")
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(raw_df)} rows.")
        
        st.header("2. Configure Variables")
        all_cols = raw_df.columns.tolist()
        
        treatment_col = st.selectbox("Treatment Column (Intervention)", all_cols, index=0)
        outcome_col = st.selectbox("Outcome Column (KPI)", all_cols, index=1)
        
        # Auto-exclude treatment/outcome from covariates list
        remaining = [c for c in all_cols if c not in [treatment_col, outcome_col]]
        covariates = st.multiselect("Control Variables (Confounders)", remaining, default=remaining[:3])
        
        run_btn = st.button("🚀 Run Causal Analysis", type="primary")

# --- MAIN DASHBOARD ---
st.title("🧠 Causal Impact Intelligence")
st.markdown("Use Machine Learning (Double ML) to isolate the **true impact** of your intervention.")

if uploaded_file and run_btn:
    if not covariates:
        st.error("Please select at least one Control Variable.")
    else:
        with st.spinner("🤖 Training Causal Forests..."):
            # 1. Train
            selected_cols = [treatment_col, outcome_col] + covariates
            clean_df, encoders = preprocess_data(raw_df, selected_cols)
            model, X_test = train_model(clean_df, treatment_col, outcome_col, covariates)
            
            # 2. Results
            ate = model.ate(X_test)
            ate_interval = model.ate_interval(X_test)
            clean_df['Predicted_Impact'] = model.effect(X_test)

        # --- SUMMARY ---
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Impact (ATE)", f"{ate:.2f}")
        with col2:
            st.metric("Confidence Interval (95%)", f"[{ate_interval[0]:.2f}, {ate_interval[1]:.2f}]")
        with col3:
            res = "✅ Effective" if ate_interval[0] > 0 else "⚠️ Inconclusive/Negative"
            st.metric("Conclusion", res)

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["Segmentation", "Drivers", "Data"])
        
        with tab1:
            st.markdown("#### Impact by User Attribute")
            x_var = st.selectbox("Segment By:", covariates)
            fig = px.scatter(clean_df, x=x_var, y="Predicted_Impact", color="Predicted_Impact", 
                             title=f"Who responds best? (Impact vs {x_var})", color_continuous_scale="RdBu")
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### What drives the impact?")
            interpreter = RandomForestRegressor(max_depth=4)
            interpreter.fit(X_test, clean_df['Predicted_Impact'])
            imp_df = pd.DataFrame({'Feature': covariates, 'Importance': interpreter.feature_importances_})
            fig2 = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.dataframe(clean_df.head(100))
            st.download_button("Download CSV", clean_df.to_csv(index=False), "results.csv", "text/csv")

elif not uploaded_file:
    st.info("👈 Please upload a CSV file in the sidebar to begin.")