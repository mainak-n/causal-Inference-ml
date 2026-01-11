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
    page_title="Causal Inference Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    /* 1. SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        min-width: 350px;
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    
    /* 2. NAVIGATION RADIO BUTTONS (LOOKS LIKE TABS) */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: stretch;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 10px 20px;
        margin-right: 5px;
        border-radius: 5px;
        flex-grow: 1;
        text-align: center;
        font-weight: 600;
        color: #495057;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: #0d6efd !important; /* Bootstrap Blue */
        color: white !important;
        border-color: #0d6efd !important;
    }

    /* 3. METRIC CARDS */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #6c757d;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #212529;
    }
    
    /* 4. GENERAL */
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #212529; }
    .stDataFrame { border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
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

def run_analysis(df, treatment_col, outcome_col, covariates, time_col=None, intervention_date=None):
    # Setup Logic
    if not covariates:
        X = np.zeros((len(df), 1))
        feature_names = ["No_Controls"]
    else:
        X = df[covariates]
        feature_names = covariates
    
    Y = df[outcome_col]
    
    # Treatment Definition
    if time_col and intervention_date:
        try:
            time_series = pd.to_datetime(df[time_col])
            intervention_ts = pd.to_datetime(intervention_date)
            df['Is_Post'] = (time_series >= intervention_ts).astype(int)
            T = df[treatment_col] * df['Is_Post']
            
            if not covariates:
                X = pd.DataFrame({'Group_Effect': df[treatment_col], 'Time_Effect': df['Is_Post']})
                feature_names = ['Group_Effect', 'Time_Effect']
            else:
                X = X.copy()
                X['Group_Effect'] = df[treatment_col]
                X['Time_Effect'] = df['Is_Post']
                feature_names = covariates + ['Group_Effect', 'Time_Effect']
        except Exception as e:
            st.error(f"Date Error: {e}")
            return None, None, None, None
    else:
        T = df[treatment_col]

    # Models
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=50, max_depth=6),
        discrete_treatment=True
    )
    est.fit(Y, T, X=X)
    
    # Stats (OLS)
    X_stats = sm.add_constant(pd.concat([T.rename("Treatment_Effect"), pd.DataFrame(X, index=df.index)], axis=1))
    X_stats.columns = ["Const", "Treatment_Effect"] + [f"Control_{i}" if isinstance(c, int) else c for i, c in enumerate(feature_names)]
    ols_model = sm.OLS(Y, X_stats).fit()
    
    return est, ols_model, X, T

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.header("Project Navigation")
    
    # Using Radio to act as Tabs so we can control Main Page View
    nav_selection = st.radio(
        "Go to step:", 
        ["Data", "Logic", "Analysis"], 
        label_visibility="collapsed"
    )
    
    st.markdown("---")

    # --- INPUTS PERSIST ACROSS TABS ---
    # We initialize variables here so they exist regardless of which tab is open
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    cols = []
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        cols = raw_df.columns.tolist()
    
    # Initialize session state for analysis results if not exists
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False

    # --- LOGIC SETTINGS (Visible in Logic or Analysis) ---
    if nav_selection in ["Logic", "Analysis"] and uploaded_file:
        st.subheader("Variable Configuration")
        treatment_col = st.selectbox("Treatment Column", cols, index=0)
        outcome_col = st.selectbox("Outcome Column", cols, index=1 if len(cols)>1 else 0)
        
        use_time = st.checkbox("Enable Time Dimension")
        time_col = None
        intervention_date = None
        
        if use_time:
            time_col = st.selectbox("Time Column", cols)
            try:
                min_d = pd.to_datetime(raw_df[time_col]).min()
                max_d = pd.to_datetime(raw_df[time_col]).max()
                intervention_date = st.date_input("Intervention Date", value=min_d, min_value=min_d, max_value=max_d)
            except:
                intervention_date = st.text_input("Intervention Value (Text)")
        
        exclude = [treatment_col, outcome_col]
        if time_col: exclude.append(time_col)
        covariates = st.multiselect("Control Variables", [c for c in cols if c not in exclude])

    # --- EXECUTE BUTTON (Visible in Analysis) ---
    run_clicked = False
    if nav_selection == "Analysis" and uploaded_file:
        st.markdown("---")
        run_clicked = st.button("RUN MODEL", type="primary", use_container_width=True)


# --- MAIN PAGE LOGIC ---

st.title("Causal Effect Analysis Portal")

# VIEW 1: DATA & LOGIC TABS (Show the Dataframe)
if nav_selection in ["Data", "Logic"]:
    if uploaded_file:
        st.subheader(f"Dataset Inspector ({len(raw_df)} rows)")
        st.markdown("Reviewing uploaded data structure.")
        st.dataframe(raw_df.head(20), use_container_width=True)
    else:
        st.info("Please upload a CSV file in the sidebar to begin.")

# VIEW 2: ANALYSIS TAB (Show Results)
elif nav_selection == "Analysis":
    if not uploaded_file:
        st.warning("Please upload data in the 'Data' tab first.")
    
    elif run_clicked:
        with st.spinner("Processing Causal Forest..."):
            # Prepare Data
            needed_cols = [treatment_col, outcome_col] + covariates
            if time_col: needed_cols.append(time_col)
            
            clean_df, encoders = preprocess_data(raw_df, needed_cols)
            
            # Run Models
            ml_model, stats_model, X_test, T_test = run_analysis(
                clean_df, treatment_col, outcome_col, covariates, time_col, intervention_date
            )
            
            if ml_model:
                # Save results to session state so they persist
                st.session_state['ml_model'] = ml_model
                st.session_state['stats_model'] = stats_model
                st.session_state['X_test'] = X_test
                st.session_state['clean_df'] = clean_df
                st.session_state['covariates'] = covariates
                st.session_state['analysis_done'] = True

    # RENDER RESULTS (If analysis has been run)
    if st.session_state['analysis_done']:
        ml_model = st.session_state['ml_model']
        stats_model = st.session_state['stats_model']
        X_test = st.session_state['X_test']
        clean_df = st.session_state['clean_df']
        covs = st.session_state['covariates']
        
        # Calculate Metrics
        ate = ml_model.ate(X_test)
        lower, upper = ml_model.ate_interval(X_test)
        p_value = stats_model.pvalues["Treatment_Effect"]
        r_squared = stats_model.rsquared
        
        st.subheader("Analysis Results")
        
        # METRICS GRID
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Impact</div>
                <div class="metric-value">{ate:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">95% Confidence Interval</div>
                <div class="metric-value">[{lower:.2f}, {upper:.2f}]</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            sig_color = "#198754" if p_value < 0.05 else "#dc3545" # Green or Red
            sig_text = "SIGNIFICANT" if p_value < 0.05 else "INCONCLUSIVE"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Statistical Certainty</div>
                <div class="metric-value" style="color: {sig_color}">{sig_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
             st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Fit (R-Squared)</div>
                <div class="metric-value">{r_squared:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # VISUALIZATIONS
        tab1, tab2, tab3 = st.tabs(["Impact Distribution", "Feature Drivers", "Statistical Details"])
        
        clean_df['Calculated_Impact'] = ml_model.effect(X_test)
        
        with tab1:
            st.markdown("**Distribution of Impact**")
            fig = px.histogram(clean_df, x='Calculated_Impact', nbins=50, 
                               color_discrete_sequence=['#0d6efd'],
                               title="Variation of Treatment Effect across Population")
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.markdown("**Drivers of Heterogeneity**")
            if covs:
                interpreter = RandomForestRegressor(max_depth=4)
                interpreter.fit(X_test, clean_df['Calculated_Impact'])
                imp = pd.DataFrame({'Variable': X_test.columns, 'Importance': interpreter.feature_importances_})
                imp = imp.sort_values('Importance', ascending=True)
                
                fig2 = px.bar(imp, x='Importance', y='Variable', orientation='h',
                              color_discrete_sequence=['#0d6efd'])
                fig2.update_layout(plot_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No control variables were selected.")
                
        with tab3:
            st.text(stats_model.summary())
            
    elif nav_selection == "Analysis" and not st.session_state['analysis_done']:
        st.info("Click 'RUN MODEL' in the sidebar to generate insights.")