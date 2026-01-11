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
st.set_page_config(
    page_title="Causal Inference Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AESTHETIC CSS ---
st.markdown("""
    <style>
    /* 1. SIDEBAR REFINEMENT */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* 2. TACTICAL TABS (IN SIDEBAR) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #e9ecef;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        color: #6c757d;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0d6efd !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 3. METRIC CARDS */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #adb5bd;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #212529;
    }
    
    /* 4. GENERAL TYPOGRAPHY */
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
    
    /* 5. FLOWCHART CONTAINER */
    .flowchart-container {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
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

def generate_pdf(ate, lower, upper, p_val, r2, treat, out):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 20, "Causal Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(108, 117, 125)
    pdf.cell(0, 10, f"Intervention: {treat}  ->  Outcome: {out}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Average Impact (ATE): {ate:.4f}", ln=True)
    pdf.cell(0, 10, f"Confidence Interval (95%): [{lower:.4f}, {upper:.4f}]", ln=True)
    sig_txt = "Statistically Significant" if p_val < 0.05 else "Inconclusive / Not Significant"
    pdf.cell(0, 10, f"Statistical Result: {sig_txt} (p={p_val:.5f})", ln=True)
    pdf.cell(0, 10, f"Model Fit (R2): {r2:.4f}", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

def run_analysis_logic(df, treatment, outcome, controls, time_col=None, date_val=None):
    if not controls:
        X = np.zeros((len(df), 1))
        features = ["No_Controls"]
    else:
        X = df[controls]
        features = controls
    
    Y = df[outcome]
    
    if time_col and date_val:
        try:
            ts = pd.to_datetime(df[time_col])
            int_ts = pd.to_datetime(date_val)
            df['Is_Post'] = (ts >= int_ts).astype(int)
            T = df[treatment] * df['Is_Post']
            if not controls:
                X = pd.DataFrame({'Group': df[treatment], 'Time': df['Is_Post']})
                features = ['Group', 'Time']
            else:
                X = X.copy()
                X['Group'] = df[treatment]
                X['Time'] = df['Is_Post']
                features = features + ['Group', 'Time']
        except:
            return None, None, None, None
    else:
        T = df[treatment]

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=50, max_depth=6),
        discrete_treatment=True
    )
    est.fit(Y, T, X=X)
    
    X_s = sm.add_constant(pd.concat([T.rename("Treat"), pd.DataFrame(X, index=df.index)], axis=1))
    X_s.columns = ["Const", "Treat"] + [f"V{i}" for i in range(len(features))]
    ols = sm.OLS(Y, X_s).fit()
    
    return est, ols, X, T

# --- SESSION STATE ---
if 'results' not in st.session_state: st.session_state['results'] = None
if 'uploaded_file' not in st.session_state: st.session_state['uploaded_file'] = None

# --- SIDEBAR: TACTICAL COMMAND CENTER ---
with st.sidebar:
    st.header("Project Config")
    
    # TACTICAL TABS
    tab_data, tab_logic, tab_run = st.tabs(["📂 Data", "⚙️ Logic", "🚀 Action"])

    # 1. DATA TAB
    with tab_data:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            st.session_state['uploaded_file'] = uploaded_file
            raw_df = pd.read_csv(uploaded_file)
            cols = raw_df.columns.tolist()
            st.success(f"Loaded {len(raw_df)} rows")
        else:
            cols = []

    # 2. LOGIC TAB
    with tab_logic:
        if uploaded_file:
            treat_col = st.selectbox("Treatment", cols, index=0)
            out_col = st.selectbox("Outcome", cols, index=1 if len(cols)>1 else 0)
            
            st.markdown("---")
            use_time = st.checkbox("Time Dimension")
            time_col, int_date = None, None
            if use_time:
                time_col = st.selectbox("Date Column", cols)
                try:
                    min_d = pd.to_datetime(raw_df[time_col]).min()
                    max_d = pd.to_datetime(raw_df[time_col]).max()
                    int_date = st.date_input("Start Date", value=min_d, min_value=min_d, max_value=max_d)
                except:
                    int_date = st.text_input("Start Value")
            
            excl = [treat_col, out_col]
            if time_col: excl.append(time_col)
            covs = st.multiselect("Controls", [c for c in cols if c not in excl])
        else:
            st.info("Upload data first")

    # 3. ACTION TAB (RUN & DOWNLOAD)
    with tab_run:
        if uploaded_file:
            run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
            
            if run_btn:
                with st.spinner("Running Causal Forest..."):
                    need = [treat_col, out_col] + covs
                    if time_col: need.append(time_col)
                    clean, enc = preprocess_data(raw_df, need)
                    
                    ml, stats, X_t, T_t = run_analysis_logic(clean, treat_col, out_col, covs, time_col, int_date)
                    
                    if ml:
                        st.session_state['results'] = {
                            'ml': ml, 'stats': stats, 'X': X_t, 'df': clean,
                            'treat': treat_col, 'out': out_col
                        }
            
            # PDF DOWNLOAD (Only appears if results exist)
            if st.session_state['results']:
                res = st.session_state['results']
                ml, stats = res['ml'], res['stats']
                ate = ml.ate(res['X'])
                l, u = ml.ate_interval(res['X'])
                p = stats.pvalues["Treat"]
                r2 = stats.rsquared
                
                pdf_data = generate_pdf(ate, l, u, p, r2, res['treat'], res['out'])
                
                st.markdown("---")
                st.download_button(
                    label="📄 DOWNLOAD PDF REPORT",
                    data=pdf_data,
                    file_name="causal_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# --- MAIN PAGE ---
st.title("Causal Inference Portal")

# LOGIC VISUALIZATION (Shows when Data is loaded)
if st.session_state['uploaded_file']:
    
    # 1. AESTHETIC FLOWCHART (Using Graphviz)
    with st.expander("Logic Visualization", expanded=True):
        g = graphviz.Digraph()
        g.attr(rankdir='LR', bgcolor='transparent')
        g.attr('node', fontname='Helvetica', shape='box', style='filled', color='white')
        g.attr('edge', fontname='Helvetica', color='#6c757d')
        
        # Nodes
        g.node('T', f'Intervention\n({treat_col})', fillcolor='#e6f4ea', color='#198754', fontcolor='#198754')
        g.node('O', f'Target Metric\n({out_col})', fillcolor='#e8f0fe', color='#0d6efd', fontcolor='#0d6efd')
        
        # Edges
        g.edge('T', 'O', label=' Impact? ', penwidth='2')
        
        if covs:
            g.node('C', 'Confounders\n(Controls)', shape='ellipse', fillcolor='#fff8c5', color='#ffc107', fontcolor='#856404')
            g.edge('C', 'T', style='dashed', dir='none')
            g.edge('C', 'O', style='dashed', dir='none')
        
        if use_time:
            g.node('D', f'Timeline\n(> {int_date})', shape='note', fillcolor='#f8f9fa')
            g.edge('D', 'T', label=' triggers ')

        st.graphviz_chart(g)

    # 2. RESULTS DASHBOARD
    if st.session_state['results']:
        res = st.session_state['results']
        ml, stats = res['ml'], res['stats']
        
        ate = ml.ate(res['X'])
        l, u = ml.ate_interval(res['X'])
        p = stats.pvalues["Treat"]
        r2 = stats.rsquared
        
        st.markdown("### Analysis Results")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Lift</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">95% Range</div><div class="metric-value">[{l:.2f}, {u:.2f}]</div></div>', unsafe_allow_html=True)
        
        with c3:
            clr = "#198754" if p < 0.05 else "#dc3545"
            txt = "SIGNIFICANT" if p < 0.05 else "INCONCLUSIVE"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value" style="color:{clr}">{txt}</div></div>', unsafe_allow_html=True)
        
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R²)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        t1, t2 = st.tabs(["📉 Impact Distribution", "📊 Stats Table"])
        res['df']['Impact'] = ml.effect(res['X'])
        
        with t1:
            fig = px.histogram(res['df'], x='Impact', nbins=50, title="Population Impact Distribution", color_discrete_sequence=['#0d6efd'])
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            st.text(stats.summary())

else:
    # EMPTY STATE
    st.info("👈 Upload your dataset in the sidebar to begin.")