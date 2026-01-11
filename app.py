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

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* 1. REMOVE TITLE GAP */
    .block-container {
        padding-top: 2rem;
    }
    
    /* 2. NAVIGATION TABS (Radio Button Styled) */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: stretch;
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 5px;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        flex-grow: 1;
        text-align: center;
        background-color: transparent;
        border: none;
        padding: 10px;
        border-radius: 5px;
        font-weight: 600;
        color: #555;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: #ffffff !important;
        color: #0d6efd !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* 3. METRIC CARDS */
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #212529; }
    .metric-label { font-size: 11px; text-transform: uppercase; color: #6c757d; letter-spacing: 0.5px; }

    /* 4. CONFIG HIGHLIGHT CARD */
    .config-card {
        background-color: #e8f4fd;
        border-left: 5px solid #0d6efd;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None

def reset_analysis():
    """Reset results if config changes"""
    st.session_state['analysis_results'] = None

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
    X_stats.columns = ["Const", "Treatment"] + [f"Var_{i}" for i in range(len(feature_names))]
    ols_model = sm.OLS(Y, X_stats).fit()
    
    return est, ols_model, X, T

# --- SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.header("Project Config")
    
    # NAVIGATION TABS
    nav = st.radio("Step", ["Data", "Logic", "Analysis"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # 1. DATA INPUTS (Always visible in Logic/Analysis to allow context)
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    cols = []
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        cols = raw_df.columns.tolist()
        
        # 2. LOGIC INPUTS (Only visible when Logic or Analysis is selected)
        if nav in ["Logic", "Analysis"]:
            st.subheader("Configuration")
            treatment_col = st.selectbox("Treatment Column", cols, index=0, on_change=reset_analysis)
            outcome_col = st.selectbox("Outcome Column", cols, index=1 if len(cols)>1 else 0, on_change=reset_analysis)
            
            use_time = st.checkbox("Time Dimension", on_change=reset_analysis)
            time_col = None
            intervention_date = None
            
            if use_time:
                time_col = st.selectbox("Date Column", cols, on_change=reset_analysis)
                try:
                    min_d = pd.to_datetime(raw_df[time_col]).min()
                    max_d = pd.to_datetime(raw_df[time_col]).max()
                    intervention_date = st.date_input("Intervention Date", value=min_d, min_value=min_d, max_value=max_d, on_change=reset_analysis)
                except:
                    intervention_date = st.text_input("Intervention Value", on_change=reset_analysis)
            
            exclude = [treatment_col, outcome_col]
            if time_col: exclude.append(time_col)
            covariates = st.multiselect("Control Variables", [c for c in cols if c not in exclude], on_change=reset_analysis)

# --- MAIN PAGE CONTENT ---

st.title("Causal Effect Analysis Portal")

# VIEW 1: DATA TAB
if nav == "Data":
    if uploaded_file:
        st.subheader(f"Data Preview ({len(raw_df)} rows)")
        st.dataframe(raw_df.head(50), use_container_width=True)
    else:
        st.info("Please upload a CSV file in the sidebar.")

# VIEW 2: LOGIC TAB (HIGHLIGHT CONFIGURATION)
elif nav == "Logic":
    if uploaded_file:
        st.subheader("Logic Configuration Summary")
        
        # Highlight Card
        st.markdown(f"""
        <div class="config-card">
            <b>Treatment (Intervention):</b> {treatment_col}<br>
            <b>Outcome (Target):</b> {outcome_col}<br>
            <b>Time Logic:</b> {'Enabled' if use_time else 'Disabled'}<br>
            <b>Controls:</b> {', '.join(covariates) if covariates else 'None'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Mapped Data Sample:**")
        # Show a snippet of ONLY the selected columns to visualize the logic
        sel_cols = [treatment_col, outcome_col] + covariates
        if time_col: sel_cols.append(time_col)
        st.dataframe(raw_df[sel_cols].head(10), use_container_width=True)
        
    else:
        st.warning("Upload data in the 'Data' tab first.")

# VIEW 3: ANALYSIS TAB
elif nav == "Analysis":
    if not uploaded_file:
        st.warning("Please upload data first.")
    else:
        # SIDEBAR BUTTON
        with st.sidebar:
            st.markdown("---")
            btn_label = "Rerun Analysis" if st.session_state['analysis_results'] else "Run Analysis"
            run = st.button(btn_label, type="primary", use_container_width=True)

        if run:
            with st.spinner("Processing Causal Models..."):
                needed = [treatment_col, outcome_col] + covariates
                if time_col: needed.append(time_col)
                
                clean_df, encoders = preprocess_data(raw_df, needed)
                ml, stats, X_test, T_test = run_causal_analysis(
                    clean_df, treatment_col, outcome_col, covariates, time_col, intervention_date
                )
                
                if ml:
                    # Store results
                    st.session_state['analysis_results'] = {
                        'ml': ml, 'stats': stats, 'X': X_test, 'df': clean_df, 'covs': covariates
                    }

        # DISPLAY RESULTS (If they exist)
        results = st.session_state['analysis_results']
        
        if results:
            ml = results['ml']
            stats = results['stats']
            X = results['X']
            df = results['df']
            covs = results['covs']
            
            ate = ml.ate(X)
            lower, upper = ml.ate_interval(X)
            p_val = stats.pvalues["Treatment"]
            r2 = stats.rsquared
            
            st.subheader("Analysis Results")
            
            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">[{lower:.2f}, {upper:.2f}]</div></div>', unsafe_allow_html=True)
            with c3:
                color = "#198754" if p_val < 0.05 else "#dc3545"
                txt = "SIGNIFICANT" if p_value < 0.05 else "INCONCLUSIVE"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-value" style="color:{color}">{txt}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tabs for Charts
            t1, t2, t3 = st.tabs(["Impact Distribution", "Drivers", "Stats Table"])
            
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
        else:
            st.info("Configuration ready. Click 'Run Analysis' in the sidebar.")