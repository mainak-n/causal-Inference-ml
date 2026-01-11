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

# --- PROFESSIONAL CSS & LAYOUT FIXES ---
st.markdown("""
    <style>
    /* 1. REMOVE HUGE TOP PADDING */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* 2. SIDEBAR TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: transparent;
        border-bottom: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 0px;
        border: none;
        border-bottom: 3px solid transparent;
        font-weight: 600;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        color: #0d6efd !important;
        border-bottom: 3px solid #0d6efd !important;
        background-color: transparent !important;
    }

    /* 3. METRIC CARDS */
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #212529; }
    .metric-label { font-size: 11px; text-transform: uppercase; color: #6c757d; font-weight: 600; }
    
    /* 4. HEADERS */
    h1 { font-size: 28px; margin-bottom: 0px; }
    h2 { font-size: 22px; margin-top: 0px; }
    h3 { font-size: 18px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False

def reset_analysis():
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
            ts = pd.to_datetime(df[time_col])
            int_ts = pd.to_datetime(intervention_date)
            df['Is_Post'] = (ts >= int_ts).astype(int)
            T = df[treatment_col] * df['Is_Post']
            if not covariates:
                X = pd.DataFrame({'Group': df[treatment_col], 'Time': df['Is_Post']})
                feature_names = ['Group', 'Time']
            else:
                X = X.copy()
                X['Group'] = df[treatment_col]
                X['Time'] = df['Is_Post']
                feature_names = covariates + ['Group', 'Time']
        except Exception:
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
    X_stats.columns = ["Const", "Treatment"] + [f"Var_{i}" for i in range(len(feature_names))]
    ols = sm.OLS(Y, X_stats).fit()
    return est, ols, X, T

def color_columns(val, col_type):
    """Helper for Highlighting Columns in Logic Tab"""
    if col_type == 'treatment': return 'background-color: #d1e7dd; color: #0f5132' # Green
    if col_type == 'outcome': return 'background-color: #cfe2ff; color: #084298'   # Blue
    if col_type == 'covariate': return 'background-color: #fff3cd; color: #664d03' # Yellow
    return ''

# --- SIDEBAR ---
with st.sidebar:
    st.header("Project Config")
    
    # SPLIT TABS
    tab_data, tab_logic, tab_run = st.tabs(["Data", "Logic", "Analysis"])

    # --- TAB 1: DATA ---
    with tab_data:
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV", type="csv", on_change=reset_analysis)
        cols = []
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
            cols = raw_df.columns.tolist()

    # --- TAB 2: LOGIC ---
    with tab_logic:
        st.subheader("Variable Mapping")
        if uploaded_file:
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
        else:
            st.info("Upload data first.")

    # --- TAB 3: RUN ---
    with tab_run:
        st.subheader("Execution")
        if uploaded_file:
            btn_text = "Rerun Analysis" if st.session_state['analysis_done'] else "Run Analysis"
            if st.button(btn_text, type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    needed = [treatment_col, outcome_col] + covariates
                    if time_col: needed.append(time_col)
                    clean_df, encoders = preprocess_data(raw_df, needed)
                    ml, stats, X_test, T_test = run_causal_analysis(clean_df, treatment_col, outcome_col, covariates, time_col, intervention_date)
                    
                    if ml:
                        st.session_state['ml'] = ml
                        st.session_state['stats'] = stats
                        st.session_state['X'] = X_test
                        st.session_state['df'] = clean_df
                        st.session_state['covs'] = covariates
                        st.session_state['analysis_done'] = True
                        st.rerun()

# --- MAIN AREA RENDERING ---

st.title("Causal Effect Analysis Portal")

# LOGIC TO DETERMINE WHAT TO SHOW BASED ON SIDEBAR ACTIVITY
# We can infer the active view by checking Session State and File Status,
# but since Streamlit tabs don't output their state, we rely on the Button Press for Analysis.
# For Data vs Logic visuals, we simply show them below if Analysis is NOT done.

if st.session_state['analysis_done']:
    # ---------------------------
    # VIEW: RESULTS DASHBOARD
    # ---------------------------
    ml = st.session_state['ml']
    stats = st.session_state['stats']
    X = st.session_state['X']
    df = st.session_state['df']
    covs = st.session_state['covs']
    
    ate = ml.ate(X)
    lower, upper = ml.ate_interval(X)
    p_val = stats.pvalues["Treatment"]
    r2 = stats.rsquared
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">[{lower:.2f}, {upper:.2f}]</div></div>', unsafe_allow_html=True)
    with c3:
        color = "#198754" if p_val < 0.05 else "#dc3545"
        txt = "SIGNIFICANT" if p_val < 0.05 else "INCONCLUSIVE"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-value" style="color:{color}">{txt}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R²)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    t1, t2, t3 = st.tabs(["Impact Distribution", "Drivers", "Stats Table"])
    df['Impact'] = ml.effect(X)
    
    with t1:
        fig = px.histogram(df, x='Impact', nbins=50, color_discrete_sequence=['#0d6efd'], title="Impact Variation")
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        if covs:
            interp = RandomForestRegressor(max_depth=4)
            interp.fit(X, df['Impact'])
            imp = pd.DataFrame({'Var': X.columns, 'Imp': interp.feature_importances_}).sort_values('Imp')
            fig2 = px.bar(imp, x='Imp', y='Var', orientation='h', color_discrete_sequence=['#0d6efd'])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No confounders selected.")
    with t3: st.text(stats.summary())

elif uploaded_file:
    # ---------------------------
    # VIEW: SETUP (DATA OR LOGIC)
    # ---------------------------
    
    # We display both Data Preview AND Logic Highlight
    # The user asked to split them, but logically, showing the Highlighted Dataframe
    # is the best way to visualize "Logic".
    
    st.subheader(f"Dataset Overview ({len(raw_df)} rows)")
    
    # LOGIC HIGHLIGHTING:
    # Create a styler object to color code the columns based on Sidebar Selection
    st.markdown("**Configuration Preview:**")
    st.caption("Green = Treatment | Blue = Outcome | Yellow = Confounders")
    
    # 1. Create a subset dataframe for display
    display_cols = [treatment_col, outcome_col] + covariates
    if use_time and time_col: display_cols.append(time_col)
    
    # Add a few unselected columns for context (up to 3)
    extra = [c for c in cols if c not in display_cols][:3]
    final_display = raw_df[display_cols + extra].head(15)
    
    # 2. Apply Colors
    def highlight_logic(x):
        df_color = pd.DataFrame('', index=x.index, columns=x.columns)
        # Treatment = Green
        if treatment_col in df_color.columns:
            df_color[treatment_col] = 'background-color: #d1e7dd; color: #0f5132; font-weight: bold'
        # Outcome = Blue
        if outcome_col in df_color.columns:
            df_color[outcome_col] = 'background-color: #cfe2ff; color: #084298; font-weight: bold'
        # Covariates = Yellow
        for c in covariates:
            if c in df_color.columns:
                df_color[c] = 'background-color: #fff3cd; color: #664d03'
        return df_color

    st.dataframe(final_display.style.apply(highlight_logic, axis=None), use_container_width=True)

else:
    # ---------------------------
    # VIEW: EMPTY STATE
    # ---------------------------
    st.info("Please upload a CSV file in the 'Data' tab to begin.")