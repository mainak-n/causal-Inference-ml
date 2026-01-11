import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Causal Inference Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    /* SIDEBAR TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: #f0f2f6;
        padding: 5px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        border: none;
        font-weight: 600;
        flex-grow: 1;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0d6efd !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* METRIC CARDS */
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 22px; font-weight: 700; color: #212529; }
    .metric-label { font-size: 11px; text-transform: uppercase; color: #6c757d; letter-spacing: 0.5px; }
    
    /* GENERAL */
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #212529; }
    </style>
    """, unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False

def reset_analysis():
    """Callback to reset the analysis state when inputs change"""
    st.session_state['analysis_done'] = False

# --- LOGIC FUNCTIONS ---
def preprocess_data(df, selected_columns):
    data = df[selected_columns].copy()
    data = data.dropna()
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object' or isinstance(data[col].dtype, pd.PeriodDtype):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
    return data, encoders

def run_causal_analysis(df, treatment_col, outcome_col, covariates, time_col=None, intervention_date=None):
    if not covariates:
        X = np.zeros((len(df), 1))
        feature_names = ["No_Controls"]
    else:
        X = df[covariates]
        feature_names = covariates
    
    Y = df[outcome_col]
    
    if time_col and intervention_date:
        try:
            time_series = pd.to_datetime(df[time_col])
            intervention_ts = pd.to_datetime(intervention_date)
            df['Is_Post'] = (time_series >= intervention_ts).astype(int)
            T = df[treatment_col] * df['Is_Post']
            
            if not covariates:
                X = pd.DataFrame({'Group': df[treatment_col], 'Time': df['Is_Post']})
                feature_names = ['Group', 'Time']
            else:
                X = X.copy()
                X['Group'] = df[treatment_col]
                X['Time'] = df['Is_Post']
                feature_names = covariates + ['Group', 'Time']
        except Exception as e:
            st.error(f"Date Error: {e}")
            return None, None, None, None
    else:
        T = df[treatment_col]

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=50, max_depth=6),
        discrete_treatment=True
    )
    est.fit(Y, T, X=X)
    
    X_stats = sm.add_constant(pd.concat([T.rename("Treatment"), pd.DataFrame(X, index=df.index)], axis=1))
    # Fix duplicate column names for statsmodels
    X_stats.columns = ["Const", "Treatment"] + [f"Var_{i}" for i in range(len(feature_names))]
    
    ols_model = sm.OLS(Y, X_stats).fit()
    
    return est, ols_model, X, T

# --- SIDEBAR LAYOUT ---
with st.sidebar:
    st.header("Project Config")
    
    # THE TWO TABS AS REQUESTED
    tab_setup, tab_run = st.tabs(["🛠️ Setup (Data & Logic)", "⚡ Run Analysis"])

    # --- TAB 1: SETUP (Combined Data & Logic) ---
    with tab_setup:
        st.subheader("1. Data Ingestion")
        uploaded_file = st.file_uploader("Upload CSV", type="csv", on_change=reset_analysis)
        
        cols = []
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
            cols = raw_df.columns.tolist()
            
            st.markdown("---")
            st.subheader("2. Logic Configuration")
            
            # Note: We add 'on_change=reset_analysis' to every input. 
            # This ensures that if the user changes a setting, the button resets to "Run".
            
            treatment_col = st.selectbox("Treatment Column", cols, index=0, on_change=reset_analysis)
            outcome_col = st.selectbox("Outcome Column", cols, index=1 if len(cols)>1 else 0, on_change=reset_analysis)
            
            use_time = st.checkbox("Enable Time Dimension", on_change=reset_analysis)
            time_col = None
            intervention_date = None
            
            if use_time:
                time_col = st.selectbox("Time Column", cols, on_change=reset_analysis)
                try:
                    min_d = pd.to_datetime(raw_df[time_col]).min()
                    max_d = pd.to_datetime(raw_df[time_col]).max()
                    intervention_date = st.date_input("Intervention Date", value=min_d, min_value=min_d, max_value=max_d, on_change=reset_analysis)
                except:
                    intervention_date = st.text_input("Intervention Value", on_change=reset_analysis)
            
            exclude = [treatment_col, outcome_col]
            if time_col: exclude.append(time_col)
            covariates = st.multiselect("Confounders", [c for c in cols if c not in exclude], on_change=reset_analysis)

    # --- TAB 2: EXECUTE ---
    with tab_run:
        if uploaded_file:
            st.info("Ready to Model")
            
            # DYNAMIC BUTTON TEXT
            btn_text = "🔄 Rerun Analysis" if st.session_state['analysis_done'] else "🚀 Run Analysis"
            
            if st.button(btn_text, type="primary", use_container_width=True):
                with st.spinner("Training Models..."):
                    # Process
                    needed = [treatment_col, outcome_col] + covariates
                    if time_col: needed.append(time_col)
                    
                    clean_df, encoders = preprocess_data(raw_df, needed)
                    
                    ml, stats, X_test, T_test = run_causal_analysis(
                        clean_df, treatment_col, outcome_col, covariates, time_col, intervention_date
                    )
                    
                    if ml:
                        st.session_state['ml'] = ml
                        st.session_state['stats'] = stats
                        st.session_state['X'] = X_test
                        st.session_state['df'] = clean_df
                        st.session_state['covs'] = covariates
                        st.session_state['analysis_done'] = True
                        st.rerun() # Refresh to show results immediately
        else:
            st.warning("Please upload data in the Setup tab.")

# --- MAIN SCREEN LOGIC ---

st.title("Causal Effect Analysis Portal")

# LOGIC: 
# If Analysis IS DONE -> Show Results
# If Analysis IS NOT DONE -> Show Data Preview (The "Setup" View)

if st.session_state['analysis_done']:
    # --- RESULT VIEW ---
    ml = st.session_state['ml']
    stats = st.session_state['stats']
    X = st.session_state['X']
    df = st.session_state['df']
    covs = st.session_state['covs']
    
    ate = ml.ate(X)
    lower, upper = ml.ate_interval(X)
    p_val = stats.pvalues["Treatment"]
    r2 = stats.rsquared
    
    # 1. METRICS
    st.subheader("Analysis Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">[{lower:.2f}, {upper:.2f}]</div></div>', unsafe_allow_html=True)
    with c3:
        color = "#198754" if p_val < 0.05 else "#dc3545"
        txt = "SIGNIFICANT" if p_val < 0.05 else "INCONCLUSIVE"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-value" style="color:{color}">{txt}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R²)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. TABS FOR CHARTS
    t1, t2, t3 = st.tabs(["📉 Impact Distribution", "🧠 Drivers", "📑 Stats Table"])
    
    df['Impact'] = ml.effect(X)
    
    with t1:
        fig = px.histogram(df, x='Impact', nbins=50, color_discrete_sequence=['#0d6efd'], title="Impact Variation")
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        if covs:
            interp = RandomForestRegressor(max_depth=4)
            interp.fit(X, df['Impact'])
            imp = pd.DataFrame({'Var': X.columns, 'Imp': interp.feature_importances_}).sort_values('Imp')
            fig2 = px.bar(imp, x='Imp', y='Var', orientation='h', color_discrete_sequence=['#0d6efd'])
            fig2.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No confounders selected.")
            
    with t3:
        st.text(stats.summary())

elif uploaded_file:
    # --- SETUP VIEW (Show Data Preview) ---
    st.subheader(f"Data Preview ({len(raw_df)} rows)")
    st.markdown("Configure your logic in the **Setup Tab**, then switch to **Run Analysis**.")
    st.dataframe(raw_df.head(20), use_container_width=True)
    
else:
    # --- EMPTY STATE ---
    st.info("👈 Please upload a CSV file in the 'Setup' tab to begin.")