import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import graphviz
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Causal Inference Portal", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 5px;
        font-weight: 600;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
    }
    .metric-card {
        background-color: white; border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 20px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #212529; }
    .metric-label { font-size: 11px; text-transform: uppercase; color: #6c757d; }
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

def generate_pdf_report(ate, lower, upper, p_val, r2):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Causal Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Average Treatment Effect (ATE): {ate:.4f}", ln=True)
    pdf.cell(200, 10, txt=f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]", ln=True)
    pdf.cell(200, 10, txt=f"P-Value: {p_val:.5f} ({'Significant' if p_val < 0.05 else 'Inconclusive'})", ln=True)
    pdf.cell(200, 10, txt=f"Model Fit (R-Squared): {r2:.4f}", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

def run_causal_analysis(df, treatment_col, outcome_col, covariates, time_col=None, intervention_date=None):
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

    # Run ML Model
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=50, max_depth=6),
        discrete_treatment=True
    )
    est.fit(Y, T, X=X)
    
    # Run Stats Model
    X_stats = sm.add_constant(pd.concat([T.rename("Treatment"), pd.DataFrame(X, index=df.index)], axis=1))
    X_stats.columns = ["Const", "Treatment"] + [f"Var_{i}" for i in range(len(feature_names))]
    ols_model = sm.OLS(Y, X_stats).fit()
    
    return est, ols_model, X, T

# --- SESSION STATE ---
if 'data' not in st.session_state: st.session_state['data'] = None
if 'cols' not in st.session_state: st.session_state['cols'] = []
if 'results' not in st.session_state: st.session_state['results'] = None

# --- MAIN LAYOUT ---
st.title("Causal Effect Analysis Portal")

# THE 3 MAIN TABS
tab_data, tab_logic, tab_analysis = st.tabs(["📂 1. Data Source", "⚙️ 2. Logic Configuration", "🚀 3. Analysis & Report"])

# ------------------------------------------------------------------
# TAB 1: DATA (File Upload Only)
# ------------------------------------------------------------------
with tab_data:
    st.subheader("Data Ingestion")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state['data'] = raw_df
        st.session_state['cols'] = raw_df.columns.tolist()
        
        st.markdown(f"**Preview ({len(raw_df)} rows):**")
        st.dataframe(raw_df.head(100), use_container_width=True)
    else:
        st.info("Please upload a CSV file to proceed.")

# ------------------------------------------------------------------
# TAB 2: LOGIC (Configuration & Flowchart)
# ------------------------------------------------------------------
with tab_logic:
    if st.session_state['data'] is not None:
        cols = st.session_state['cols']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Map Variables")
            treatment = st.selectbox("Treatment (Intervention)", cols, index=0)
            outcome = st.selectbox("Outcome (Target)", cols, index=1 if len(cols)>1 else 0)
            
            use_time = st.checkbox("Enable Time Dimension")
            time_col, intervention_date = None, None
            
            if use_time:
                time_col = st.selectbox("Time Column", cols)
                try:
                    df = st.session_state['data']
                    min_d = pd.to_datetime(df[time_col]).min()
                    max_d = pd.to_datetime(df[time_col]).max()
                    intervention_date = st.date_input("Start Date", value=min_d, min_value=min_d, max_value=max_d)
                except:
                    intervention_date = st.text_input("Start Value")

            exclude = [treatment, outcome]
            if time_col: exclude.append(time_col)
            controls = st.multiselect("Confounders (Controls)", [c for c in cols if c not in exclude])
            
            # Save config to session
            st.session_state['config'] = {
                'treatment': treatment, 'outcome': outcome, 
                'controls': controls, 'time': time_col, 'date': intervention_date
            }

        with col2:
            st.subheader("Logic Flow Visualization")
            # DYNAMIC FLOWCHART
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR')
            
            graph.node('T', f'Treatment\n({treatment})', shape='box', style='filled', fillcolor='#d1e7dd', color='#0f5132')
            graph.node('O', f'Outcome\n({outcome})', shape='box', style='filled', fillcolor='#cfe2ff', color='#084298')
            
            graph.edge('T', 'O', label='Causal Impact?')
            
            if controls:
                graph.node('C', 'Confounders\n(Controls)', shape='ellipse', style='filled', fillcolor='#fff3cd', color='#664d03')
                graph.edge('C', 'T', style='dashed')
                graph.edge('C', 'O', style='dashed')
                
            if use_time:
                 graph.node('Time', f'Time Filter\n(> {intervention_date})', shape='note')
                 graph.edge('Time', 'T', label='activates')

            st.graphviz_chart(graph)
            
    else:
        st.warning("Please upload data in the 'Data Source' tab first.")

# ------------------------------------------------------------------
# TAB 3: ANALYSIS (Run & Download)
# ------------------------------------------------------------------
with tab_analysis:
    if st.session_state['data'] is not None and 'config' in st.session_state:
        cfg = st.session_state['config']
        
        col_run, col_down = st.columns([1, 4])
        with col_run:
            run_btn = st.button("🚀 Run Analysis", type="primary")
        
        if run_btn:
            with st.spinner("Training Causal Models..."):
                needed = [cfg['treatment'], cfg['outcome']] + cfg['controls']
                if cfg['time']: needed.append(cfg['time'])
                
                clean_df, encoders = preprocess_data(st.session_state['data'], needed)
                
                ml, stats, X_test, T_test = run_causal_analysis(
                    clean_df, cfg['treatment'], cfg['outcome'], cfg['controls'], cfg['time'], cfg['date']
                )
                
                if ml:
                    st.session_state['results'] = {'ml': ml, 'stats': stats, 'X': X_test, 'df': clean_df}

        # DISPLAY RESULTS
        if st.session_state['results']:
            res = st.session_state['results']
            ml, stats = res['ml'], res['stats']
            
            # Calc Metrics
            ate = ml.ate(res['X'])
            lower, upper = ml.ate_interval(res['X'])
            
            # FIX: Use correct variable name p_val vs p_value
            p_val = stats.pvalues["Treatment"]
            r2 = stats.rsquared
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">[{lower:.2f}, {upper:.2f}]</div></div>', unsafe_allow_html=True)
            
            with c3:
                # FIX: Logic handles p_val correctly now
                color = "#198754" if p_val < 0.05 else "#dc3545"
                txt = "SIGNIFICANT" if p_val < 0.05 else "INCONCLUSIVE"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-value" style="color:{color}">{txt}</div></div>', unsafe_allow_html=True)
            
            with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # PDF DOWNLOAD BUTTON
            pdf_bytes = generate_pdf_report(ate, lower, upper, p_val, r2)
            st.download_button(
                label="📄 Download Report (PDF)",
                data=pdf_bytes,
                file_name="causal_analysis_report.pdf",
                mime="application/pdf"
            )
            
            # Charts
            t1, t2 = st.tabs(["Impact Distribution", "Stats Details"])
            res['df']['Impact'] = ml.effect(res['X'])
            
            with t1:
                fig = px.histogram(res['df'], x='Impact', nbins=50, title="Impact Variation")
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                st.text(stats.summary())

    else:
        st.info("Configure your logic in Tab 2 first.")