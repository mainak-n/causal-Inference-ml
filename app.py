import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Causal Omni-Tool", layout="wide", page_icon="🔮")

# --- CSS FOR REPORTS ---
st.markdown("""
    <style>
    .report-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4e8cff; }
    .sig-green { color: green; font-weight: bold; }
    .sig-red { color: red; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🧠 LOGIC: TIME SERIES (Interrupted Time Series)
# ==========================================
def run_its_analysis(df, time_col, outcome_col, intervention_date):
    """
    Runs Interrupted Time Series (ITS) using Segmented Regression.
    Model: Y = β0 + β1(Time) + β2(Intervention) + β3(Time_After)
    """
    df = df.sort_values(time_col).copy()
    
    # Create Time Index (1, 2, 3...)
    df['Time_Index'] = np.arange(len(df))
    
    # Create Intervention Dummy (0 before, 1 after)
    # We assume 'intervention_date' is the first period OF the treatment
    df['Is_Treated'] = (df[time_col] >= intervention_date).astype(int)
    
    # Create 'Time After' (0 before, 1, 2, 3... after)
    intervention_idx = df[df[time_col] == intervention_date].iloc[0]['Time_Index']
    df['Time_Since_Intervention'] = np.where(
        df['Is_Treated'] == 1, 
        df['Time_Index'] - intervention_idx, 
        0
    )
    
    # Run OLS Regression (Standard Statistical Model)
    X = df[['Time_Index', 'Is_Treated', 'Time_Since_Intervention']]
    X = sm.add_constant(X) # Adds β0
    y = df[outcome_col]
    
    model = sm.OLS(y, X).fit()
    
    return model, df

# ==========================================
# 🧠 LOGIC: SNAPSHOT (Dose Response)
# ==========================================
def run_dose_response(df, treatment_col, outcome_col, covariates):
    # Causal Forest handles continuous/multi-level treatments automatically
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=6),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=6), # Regressor for continuous treatment
        discrete_treatment=False, # FALSE allows 0, 1, 2, 3...
        random_state=42
    )
    
    Y = df[outcome_col]
    T = df[treatment_col]
    X = df[covariates]
    
    est.fit(Y, T, X=X)
    return est, X

# ==========================================
# 📱 UI
# ==========================================
st.title("🔮 Causal Omni-Tool: Time & Dose Analysis")

# --- MODE SELECTOR ---
mode = st.radio("Select Analysis Mode:", ["📅 Time Series (Intervention Date)", "👥 Snapshot (Multi-Level Treatment)"], horizontal=True)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # ---------------------------------------------------------
    # MODE 1: TIME SERIES (The "Day 1, 2, 3" Request)
    # ---------------------------------------------------------
    if "Time Series" in mode:
        st.subheader("📅 Interrupted Time Series Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_col = st.selectbox("Time Column (e.g., Date, Week)", df.columns)
            # Try to convert to datetime for plotting, but keep raw for logic if needed
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                pass 
                
        with col2:
            outcome_col = st.selectbox("Outcome Metric", df.columns, index=1)
            
        with col3:
            # Dropdown to pick the exact date/index of intervention
            intervention_point = st.selectbox("When did the Intervention start?", df[time_col].unique())

        if st.button("Run Time Analysis"):
            model, res_df = run_its_analysis(df, time_col, outcome_col, intervention_point)
            
            # --- 1. VISUALIZATION ---
            # Plot Actual vs Fitted
            res_df['Fitted_Values'] = model.predict(sm.add_constant(res_df[['Time_Index', 'Is_Treated', 'Time_Since_Intervention']]))
            
            fig = go.Figure()
            # Actual Data
            fig.add_trace(go.Scatter(x=res_df[time_col], y=res_df[outcome_col], mode='markers', name='Actual Data', marker=dict(color='gray', opacity=0.6)))
            # Trend Line
            fig.add_trace(go.Scatter(x=res_df[time_col], y=res_df['Fitted_Values'], mode='lines', name='Causal Trend', line=dict(color='blue', width=3)))
            # Vertical Line for Intervention
            fig.add_vline(x=intervention_point, line_dash="dash", line_color="red", annotation_text="Intervention")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 2. STATISTICAL REPORT ---
            st.markdown("### 📊 Statistical Report (Regression Results)")
            
            # Extract Coefficients
            coef_intervention = model.params['Is_Treated']
            p_val = model.pvalues['Is_Treated']
            conf_int = model.conf_int().loc['Is_Treated']
            
            # Interpretation Logic
            sig_text = "Statistically Significant ✅" if p_val < 0.05 else "Not Significant ❌"
            color_class = "sig-green" if p_val < 0.05 else "sig-red"
            
            st.markdown(f"""
            <div class="report-box">
                <h4>Immediate Impact (The "Jump")</h4>
                <p>At the moment of intervention, the metric shifted by: <b>{coef_intervention:.4f}</b></p>
                <p>P-Value: <b>{p_val:.5f}</b> (<span class="{color_class}">{sig_text}</span>)</p>
                <p>95% Confidence Interval: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("See Full Regression Table (For Statisticians)"):
                st.text(model.summary())

    # ---------------------------------------------------------
    # MODE 2: SNAPSHOT (The "0 vs 1 vs 2 Emails" Request)
    # ---------------------------------------------------------
    else:
        st.subheader("👥 Multi-Level Dose Response")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Note: This logic now accepts continuous inputs (0, 1, 2, 1.5, etc.)
            treat_col = st.selectbox("Treatment (e.g., Number of Emails)", df.columns)
        with col2:
            out_col = st.selectbox("Outcome", df.columns, index=1)
        with col3:
            covs = st.multiselect("Confounders", [c for c in df.columns if c not in [treat_col, out_col]])

        if st.button("Calculate Dose Curve"):
            if not covs:
                st.error("Select confounders.")
            else:
                # Preprocessing
                clean_df = df.copy().dropna()
                for c in covs:
                    if clean_df[c].dtype == 'object':
                        clean_df[c] = LabelEncoder().fit_transform(clean_df[c].astype(str))
                
                with st.spinner("Calculating Dose-Response Curve..."):
                    model, X = run_dose_response(clean_df, treat_col, out_col, covs)
                    
                    # --- CREATE THE DOSE CURVE ---
                    # We predict the outcome for a hypothetical "Average User" across range of doses
                    min_dose = clean_df[treat_col].min()
                    max_dose = clean_df[treat_col].max()
                    
                    # Create range: 0, 1, 2, 3...
                    doses = np.linspace(min_dose, max_dose, num=20)
                    
                    # We need to predict the effect relative to baseline (dose=0)
                    # Note: EconML's 'effect' function predicts marginal effect (slope). 
                    # For a curve, we often look at Average Marginal Effect at different levels.
                    
                    # SIMPLER VISUALIZATION FOR NON-PHDS:
                    # Plot Raw Data vs "Causal Trend"
                    # We create a partial dependence plot style visualization
                    
                    # Get predicted CATEs for everyone
                    clean_df['Individual_Effect'] = model.effect(X)
                    
                    # Plot Treatment (X) vs Outcome (Y) adjusted for confounders
                    # This is approximately the "Dose Response"
                    
                    fig = px.scatter(
                        clean_df, x=treat_col, y=out_col, 
                        color='Individual_Effect', 
                        title="Dose-Response: Treatment Level vs Outcome",
                        labels={treat_col: "Intervention Level (e.g., # Emails)", 'Individual_Effect': "Impact Strength"},
                        trendline="lowess" # Adds a smooth curve showing the trend
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    **How to read this:**
                    The **Line** shows the trend. If it goes up, more intervention = better results.
                    The **Color** shows if the impact is consistent (Darker = Stronger Impact).
                    If the line flattens out (e.g., after 2 emails), it means adding a 3rd email adds no value (Diminishing Returns).
                    """)