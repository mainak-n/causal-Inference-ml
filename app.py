import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Causal Inference Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #0e1117;
        font-weight: 600;
    }
    .metric-container {
        border: 1px solid #e6e6e6;
        padding: 20px;
        border-radius: 8px;
        background-color: #f8f9fa;
        text-align: center;
    }
    .stDataFrame {
        border: 1px solid #e6e6e6;
        border-radius: 5px;
    }
    div[data-testid="stExpander"] details summary {
        font-weight: 600;
        color: #31333F;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC FUNCTIONS ---

def preprocess_data(df, selected_columns):
    """
    Prepares data for ML: Handles missing values and encodes text columns.
    """
    data = df[selected_columns].copy()
    data = data.dropna()
    
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
            
    return data, encoders

def run_causal_model(data, treatment_col, outcome_col, controls, time_col=None, intervention_val=None):
    """
    Runs Double Machine Learning (Causal Forest).
    Logic:
    - If Time is provided: Implements Difference-in-Differences (DiD) logic.
    - If No Time: Implements standard Treatment Effect logic.
    """
    
    # 1. Setup Variables
    Y = data[outcome_col]
    X = data[controls]  # Confounders
    
    # 2. Define Treatment Definition based on Logic
    if time_col and intervention_val:
        # --- DIFFERENCE-IN-DIFFERENCES LOGIC ---
        # We need to isolate the interaction: Being in Treatment Group AND being in Post-Period.
        
        # Create Post-Period Dummy (1 if after intervention, 0 if before)
        # Assumes data is sorted or comparable. 
        # For numeric time (years/days):
        data['Is_Post'] = (data[time_col] >= intervention_val).astype(int)
        
        # The 'Treatment' for the model is the INTERACTION term.
        # T = 1 only if you are in the Treatment Group AND it is the Post Period.
        T = data[treatment_col] * data['Is_Post']
        
        # CRITICAL STEP: 
        # We must add the Main Effects (Group ID and Time ID) to the Controls (X).
        # This forces the model to remove the baseline group differences and baseline time trends.
        X = X.copy()
        X['Group_Main_Effect'] = data[treatment_col]
        X['Time_Main_Effect'] = data['Is_Post']
        
    else:
        # --- STANDARD RCT LOGIC ---
        # No time dimension. Intervention started at the beginning.
        # Simple comparison of Treatment vs Control group, controlling for X.
        T = data[treatment_col]
    
    # 3. Train Causal Forest
    # We use Random Forest for both propensity (T) and outcome (Y) models to handle non-linearities.
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        discrete_treatment=True,
        random_state=42
    )
    
    est.fit(Y, T, X=X)
    
    return est, X, T

# --- APPLICATION LAYOUT ---

st.title("Causal Effect Analysis Portal")
st.markdown("""
This tool uses **Double Machine Learning** to estimate the causal impact of an intervention. 
It supports both **Randomized Control Trials (RCT)** and **Difference-in-Differences (DiD)** designs.
""")

st.markdown("---")

# 1. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload Data File (CSV)", type="csv")
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        all_cols = raw_df.columns.tolist()
        
        st.subheader("Variable Mapping")
        
        # A. Treatment & Outcome
        treatment_col = st.selectbox(
            "Treatment Group Column", 
            all_cols,
            help="Binary column (0=Control, 1=Treatment Group)"
        )
        
        outcome_col = st.selectbox(
            "Outcome Column", 
            all_cols, 
            index=1,
            help="The metric affected by the intervention (e.g., Sales, Health Score)"
        )
        
        # B. Time Configuration (Optional)
        use_time = st.checkbox("Include Time Dimension (Pre/Post Analysis)")
        time_col = None
        intervention_val = None
        
        if use_time:
            time_col = st.selectbox("Time Column", all_cols)
            # Try to infer type for input
            if pd.api.types.is_numeric_dtype(raw_df[time_col]):
                intervention_val = st.number_input(f"Intervention Start Value ({time_col})", value=raw_df[time_col].median())
            else:
                intervention_val = st.text_input(f"Intervention Start Value ({time_col})")
                
        # C. Confounders
        # Exclude selected columns from options
        exclude = [treatment_col, outcome_col]
        if time_col: exclude.append(time_col)
        
        covariates = st.multiselect(
            "Control Variables (Confounders)", 
            [c for c in all_cols if c not in exclude],
            help="Select all other variables that influence the outcome."
        )
        
        run_btn = st.button("Run Analysis", type="primary")

# 2. MAIN CONTENT AREA
if uploaded_file:
    # DATA PREVIEW SECTION
    st.subheader("Data Inspector")
    st.markdown(f"**Filename:** {uploaded_file.name} | **Rows:** {len(raw_df)} | **Columns:** {len(raw_df.columns)}")
    
    # Display top 20 rows as requested
    st.dataframe(raw_df.head(20), use_container_width=True)
    
    if run_btn:
        if not covariates:
            st.error("Configuration Error: Please select at least one Control Variable.")
        else:
            with st.spinner("Processing Causal Models..."):
                try:
                    # 1. Prepare Data
                    cols_needed = [treatment_col, outcome_col] + covariates
                    if time_col: cols_needed.append(time_col)
                    
                    clean_df, encoders = preprocess_data(raw_df, cols_needed)
                    
                    # 2. Run Model
                    model, X_test, T_vector = run_causal_model(
                        clean_df, treatment_col, outcome_col, covariates, 
                        time_col, intervention_val
                    )
                    
                    # 3. Extract Metrics
                    ate = model.ate(X_test)
                    ate_interval = model.ate_interval(X_test)
                    clean_df['Calculated_Impact'] = model.effect(X_test)
                    
                    # --- RESULTS DASHBOARD ---
                    st.markdown("### Analysis Results")
                    
                    # Metric Cards
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.markdown(
                            f"""<div class="metric-container">
                            <div style="font-size: 14px; color: #666;">Average Treatment Effect</div>
                            <div style="font-size: 24px; font-weight: bold;">{ate:.4f}</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    
                    with c2:
                        lower, upper = ate_interval
                        st.markdown(
                            f"""<div class="metric-container">
                            <div style="font-size: 14px; color: #666;">95% Confidence Interval</div>
                            <div style="font-size: 24px; font-weight: bold;">[{lower:.3f}, {upper:.3f}]</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    
                    with c3:
                        is_sig = (lower > 0) or (upper < 0)
                        status = "Statistically Significant" if is_sig else "Inconclusive (Null Hypothesis)"
                        color = "#28a745" if is_sig else "#6c757d"
                        
                        st.markdown(
                            f"""<div class="metric-container">
                            <div style="font-size: 14px; color: #666;">Statistical Conclusion</div>
                            <div style="font-size: 20px; font-weight: bold; color: {color};">{status}</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )

                    # --- DETAILED CHARTS ---
                    st.markdown("### Visualization")
                    
                    tab1, tab2 = st.tabs(["Impact Distribution", "Feature Drivers"])
                    
                    with tab1:
                        st.markdown("**Distribution of Causal Impact across Population**")
                        fig = px.histogram(
                            clean_df, 
                            x='Calculated_Impact', 
                            nbins=30,
                            title="How much did the intervention change the outcome per row?",
                            color_discrete_sequence=['#4e8cff']
                        )
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        fig.update_layout(showlegend=False, plot_bgcolor="white")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with tab2:
                        st.markdown("**Which variables influence the effectiveness?**")
                        # Simple feature importance on the effects
                        interpreter = RandomForestRegressor(max_depth=4)
                        interpreter.fit(X_test, clean_df['Calculated_Impact'])
                        
                        imp_df = pd.DataFrame({
                            'Variable': covariates,
                            'Importance': interpreter.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig2 = px.bar(
                            imp_df, 
                            x='Importance', 
                            y='Variable', 
                            orientation='h',
                            title="Drivers of Heterogeneity",
                            color_discrete_sequence=['#4e8cff']
                        )
                        fig2.update_layout(plot_bgcolor="white")
                        st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis Error: {str(e)}")
                    st.info("Tip: Ensure your columns are numeric or valid categorical text.")

else:
    # Empty State
    st.info("Please upload a CSV file from the sidebar to begin analysis.")