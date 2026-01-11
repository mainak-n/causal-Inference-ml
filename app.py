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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Causal Inference Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* 1. NAVIGATION TABS (Radio styled as Tabs) */
    div.row-widget.stRadio > div {
        flex-direction: row;
        gap: 2px;
        background-color: #f8f9fa;
        padding: 4px;
        border-radius: 8px;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        flex-grow: 1;
        justify-content: center;
        background-color: transparent;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-weight: 600;
        color: #6c757d;
        border: 1px solid transparent;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: #ffffff !important;
        color: #0d6efd !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef !important;
    }

    /* 2. SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }

    /* 3. METRIC CARDS */
    .metric-card {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #adb5bd;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #212529;
    }
    /* Smaller font specifically for the Status Text */
    .metric-status {
        font-size: 18px; /* Reduced from 24px */
        font-weight: 700;
    }

    /* 4. GENERAL */
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; letter-spacing: -0.5px; }
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

def generate_pdf(ate, lower, upper, p_val, r2, treat, out):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, "Causal Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Effect of '{treat}' on '{out}'", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Average Impact: {ate:.4f}", ln=True)
    pdf.cell(0, 10, f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]", ln=True)
    
    sig_txt = "Significant" if (not np.isnan(p_val) and p_val < 0.05) else "Inconclusive"
    p_txt = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
    
    pdf.cell(0, 10, f"Statistical Significance: {sig_txt} (p={p_txt})", ln=True)
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

    # ML Model
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=50, max_depth=6),
        discrete_treatment=True
    )
    est.fit(Y, T, X=X)
    
    # Stats Model
    ols = None
    try:
        X_df = pd.DataFrame(X, index=df.index)
        X_df.columns = [f"V{i}" for i in range(X_df.shape[1])]
        X_ols = sm.add_constant(pd.concat([T.rename("Treat"), X_df], axis=1))
        ols = sm.OLS(Y, X_ols).fit()
    except:
        pass
    
    return est, ols, X, T

# --- SESSION STATE ---
if 'results' not in st.session_state: st.session_state['results'] = None
if 'uploaded_file' not in st.session_state: st.session_state['uploaded_file'] = None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.header("Project Config")
    
    # NAVIGATION (Styled as Tabs)
    nav = st.radio("Navigation", ["Data", "Logic", "Analysis"], label_visibility="collapsed")
    st.markdown("---")

    # 1. DATA INPUTS (Always visible for context)
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file
        raw_df = pd.read_csv(uploaded_file)
        cols = raw_df.columns.tolist()
        
        # 2. LOGIC INPUTS (Visible in Logic or Analysis)
        if nav in ["Logic", "Analysis"]:
            st.subheader("Logic Setup")
            treat_col = st.selectbox("Treatment", cols, index=0)
            out_col = st.selectbox("Outcome", cols, index=1 if len(cols)>1 else 0)
            
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

    # 3. ACTION BUTTONS (Visible in Analysis)
    if nav == "Analysis" and uploaded_file:
        st.markdown("---")
        run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
        
        if run_btn:
            with st.spinner("Processing..."):
                need = [treat_col, out_col] + covs
                if time_col: need.append(time_col)
                clean, enc = preprocess_data(raw_df, need)
                ml, stats, X_t, T_t = run_analysis_logic(clean, treat_col, out_col, covs, time_col, int_date)
                
                if ml:
                    st.session_state['results'] = {
                        'ml': ml, 'stats': stats, 'X': X_t, 'df': clean,
                        'treat': treat_col, 'out': out_col
                    }
        
        if st.session_state['results']:
            res = st.session_state['results']
            ml, stats = res['ml'], res['stats']
            ate = ml.ate(res['X'])
            l, u = ml.ate_interval(res['X'])
            p = stats.pvalues["Treat"] if (stats and "Treat" in stats.pvalues) else np.nan
            r2 = stats.rsquared if stats else 0.0
            
            pdf_data = generate_pdf(ate, l, u, p, r2, res['treat'], res['out'])
            st.download_button("📄 DOWNLOAD PDF", pdf_data, "report.pdf", "application/pdf", use_container_width=True)

# --- MAIN PAGE VIEWS ---

st.title("Causal Inference Portal")

# VIEW 1: DATA TABLE
if nav == "Data":
    if st.session_state['uploaded_file']:
        st.subheader(f"Dataset Inspector ({len(raw_df)} rows)")
        st.dataframe(raw_df.head(100), use_container_width=True)
    else:
        st.info("Please upload a CSV file in the sidebar.")

# VIEW 2: LOGIC FLOWCHART
elif nav == "Logic":
    if st.session_state['uploaded_file']:
        st.subheader("Logic Visualization")
        
        g = graphviz.Digraph()
        g.attr(rankdir='LR', bgcolor='transparent')
        g.attr('node', fontname='Helvetica', shape='box', style='filled', color='white', fontcolor='#333')
        g.attr('edge', fontname='Helvetica', color='#adb5bd')
        
        # Nodes
        g.node('T', f'Intervention\n({treat_col})', fillcolor='#d1e7dd', color='#0f5132', fontcolor='#0f5132')
        g.node('O', f'Outcome\n({out_col})', fillcolor='#cfe2ff', color='#084298', fontcolor='#084298')
        g.edge('T', 'O', label=' Impact? ', penwidth='2')
        
        if covs:
            g.node('C', 'Confounders', shape='ellipse', fillcolor='#fff3cd', color='#856404', fontcolor='#856404')
            g.edge('C', 'T', style='dashed', dir='none')
            g.edge('C', 'O', style='dashed', dir='none')
            
        if use_time:
            g.node('D', f'Timeline\n(> {int_date})', shape='note', fillcolor='#f8f9fa')
            g.edge('D', 'T', label=' triggers ')

        st.graphviz_chart(g)
    else:
        st.warning("Upload data first to see the logic flow.")

# VIEW 3: RESULTS DASHBOARD
elif nav == "Analysis":
    if st.session_state['results']:
        res = st.session_state['results']
        ml, stats = res['ml'], res['stats']
        
        ate = ml.ate(res['X'])
        l, u = ml.ate_interval(res['X'])
        
        # P-Value Logic
        if stats and "Treat" in stats.pvalues:
            p = stats.pvalues["Treat"]
            r2 = stats.rsquared
            is_sig = p < 0.05
            sig_color = "#198754" if is_sig else "#dc3545"
            sig_text = "SIGNIFICANT" if is_sig else "INCONCLUSIVE"
        else:
            p = np.nan
            r2 = 0.0
            sig_color = "#6c757d"
            sig_text = "N/A"

        st.markdown("### Analysis Results")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">95% Range</div><div class="metric-value">[{l:.2f}, {u:.2f}]</div></div>', unsafe_allow_html=True)
        
        # Reduced Font Size for Status
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-status" style="color:{sig_color}">{sig_text}</div></div>', unsafe_allow_html=True)
        
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        t1, t2 = st.tabs(["Impact Distribution", "Stats Table"])
        res['df']['Impact'] = ml.effect(res['X'])
        
        with t1:
            fig = px.histogram(res['df'], x='Impact', nbins=50, title="Population Impact", color_discrete_sequence=['#0d6efd'])
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            if stats:
                st.text(stats.summary())
            else:
                st.warning("Statistical details unavailable (collinearity detected).")
    else:
        if st.session_state['uploaded_file']:
            st.info("Click 'RUN ANALYSIS' in the sidebar.")
        else:
            st.warning("Please upload data first.")