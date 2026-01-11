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
    page_title="Causal Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TACTICAL CSS STYLING ---
st.markdown("""
    <style>
    /* 1. MAKE SIDEBAR WIDER & TACTICAL */
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 450px;
        background-color: #f4f5f7;
        border-right: 2px solid #d0d7de;
    }
    
    /* 2. STYLE THE TABS TO LOOK LIKE BUTTONS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        font-weight: 600;
        flex-grow: 1; /* Make them equal width */
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff !important;
        color: white !important;
        border: none;
    }

    /* 3. METRIC BOXES */
    .metric-box {
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        background-color: white;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stat-sig { color: #28a745; font-weight: bold; }
    .stat-insig { color: #dc3545; font-weight: bold; }
    
    /* 4. HEADERS */
    h1, h2, h3 { color: #1f2937; font-weight: 700; }
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
    # Setup Logic (Same as before)
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
    
    X_stats = sm.add_constant(pd.concat([T.rename("Treatment_Effect"), pd.DataFrame(X, index=df.index)], axis=1))
    X_stats.columns = ["Const", "Treatment_Effect"] + [f"Control_{i}" if isinstance(c, int) else c for i, c in enumerate(feature_names)]
    ols_model = sm.OLS(Y, X_stats).fit()
    
    return est, ols_model, X, T

# --- UI LAYOUT ---

st.title("🔮 Causal Command Center")

# --- SIDEBAR: TACTICAL TABS ---
with st.sidebar:
    st.header("Operation Panel")
    
    # THE KEY CHANGE: Tabs instead of scrolling list
    tab_data, tab_logic, tab_run = st.tabs(["📂 Data", "⚙️ Logic", "🚀 Execute"])
    
    # --- TAB 1: DATA ---
    with tab_data:
        st.info("Step 1: Ingest Dataset")
        uploaded_file = st.file_uploader("Drop CSV Here", type="csv")
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded: {len(raw_df)} Rows")
            cols = raw_df.columns.tolist()
        else:
            cols = []

    # --- TAB 2: LOGIC ---
    with tab_logic:
        if uploaded_file:
            st.info("Step 2: Map Variables")
            
            treatment_col = st.selectbox("Treatment (0/1)", cols, index=0)
            outcome_col = st.selectbox("Outcome (KPI)", cols, index=1 if len(cols)>1 else 0)
            
            st.markdown("---")
            use_time = st.checkbox("Enable Time Dimension")
            
            time_col = None
            intervention_date = None
            
            if use_time:
                time_col = st.selectbox("Date Column", cols)
                try:
                    min_d = pd.to_datetime(raw_df[time_col]).min()
                    max_d = pd.to_datetime(raw_df[time_col]).max()
                    intervention_date = st.date_input("Intervention Date", value=min_d, min_value=min_d, max_value=max_d)
                except:
                    intervention_date = st.text_input("Intervention Value")
            
            st.markdown("---")
            exclude = [treatment_col, outcome_col]
            if time_col: exclude.append(time_col)
            covariates = st.multiselect("Confounders (Controls)", [c for c in cols if c not in exclude])
        else:
            st.warning("Upload data first.")

    # --- TAB 3: EXECUTE ---
    with tab_run:
        if uploaded_file:
            st.info("Step 3: Run Models")
            st.markdown("Click below to train Double ML & OLS models.")
            run_btn = st.button("🔥 RUN ANALYSIS", type="primary", use_container_width=True)
        else:
            st.warning("Upload data first.")

# --- MAIN SCREEN ---

if uploaded_file:
    # Quick visual check of uploaded data
    with st.expander("👀 View Raw Data Source", expanded=True):
        st.dataframe(raw_df.head(10), use_container_width=True)

    if run_btn:
        with st.spinner("🤖 Simulating Counterfactuals..."):
            
            # Prepare & Run
            needed_cols = [treatment_col, outcome_col] + covariates
            if time_col: needed_cols.append(time_col)
            
            clean_df, encoders = preprocess_data(raw_df, needed_cols)
            
            ml_model, stats_model, X_test, T_test = run_analysis(
                clean_df, treatment_col, outcome_col, covariates, time_col, intervention_date
            )
            
            if ml_model:
                # Metrics
                ate = ml_model.ate(X_test)
                lower, upper = ml_model.ate_interval(X_test)
                p_value = stats_model.pvalues["Treatment_Effect"]
                r_squared = stats_model.rsquared
                
                # --- RESULTS ---
                st.markdown("### 📊 Executive Summary")
                
                # 4-Column Metric Layout
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    st.markdown(f"""<div class="metric-box">
                        <div style="font-size:12px; color:#888;">AVERAGE LIFT</div>
                        <div style="font-size:26px; font-weight:800;">{ate:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""<div class="metric-box">
                        <div style="font-size:12px; color:#888;">CONFIDENCE INTERVAL</div>
                        <div style="font-size:26px; font-weight:800;">[{lower:.2f}, {upper:.2f}]</div>
                    </div>""", unsafe_allow_html=True)
                    
                with c3:
                    color = "stat-sig" if p_value < 0.05 else "stat-insig"
                    txt = "SIGNIFICANT" if p_value < 0.05 else "INCONCLUSIVE"
                    st.markdown(f"""<div class="metric-box">
                        <div style="font-size:12px; color:#888;">STATISTICAL CHECK</div>
                        <div style="font-size:20px; font-weight:800;" class="{color}">{txt}</div>
                    </div>""", unsafe_allow_html=True)
                    
                with c4:
                     st.markdown(f"""<div class="metric-box">
                        <div style="font-size:12px; color:#888;">MODEL FIT (R²)</div>
                        <div style="font-size:26px; font-weight:800;">{r_squared:.2f}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # --- TABS FOR VISUALIZATION ---
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📉 Impact Distribution", "🧠 Feature Drivers", "📑 Full Stats"])
                
                clean_df['Calculated_Impact'] = ml_model.effect(X_test)

                with viz_tab1:
                    st.markdown("**Did everyone react the same way?**")
                    fig = px.histogram(clean_df, x='Calculated_Impact', nbins=40, color_discrete_sequence=['#4e8cff'])
                    fig.add_vline(x=0, line_dash="dash", line_color="black")
                    fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Right of line = Positive Impact. Left of line = Negative Impact.")
                    
                with viz_tab2:
                    st.markdown("**What characteristics change the outcome?**")
                    if covariates:
                        interpreter = RandomForestRegressor(max_depth=4)
                        interpreter.fit(X_test, clean_df['Calculated_Impact'])
                        imp = pd.DataFrame({'Var': X_test.columns, 'Imp': interpreter.feature_importances_}).sort_values('Imp')
                        fig2 = px.bar(imp, x='Imp', y='Var', orientation='h', color_discrete_sequence=['#4e8cff'])
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No controls selected.")
                        
                with viz_tab3:
                    st.text(stats_model.summary())

else:
    # Empty State with a nice prompt
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #888;">
        <h3>👋 Welcome to Causal Command</h3>
        <p>Please open the <b>📂 Data</b> tab in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)