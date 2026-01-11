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

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    /* 1. NAVIGATION TABS (Radio styled as Tabs) */
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: stretch;
        background-color: #e9ecef;
        padding: 4px;
        border-radius: 8px;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        flex-grow: 1;
        text-align: center;
        background-color: transparent;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        color: #6c757d;
        box-shadow: none;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: #ffffff !important;
        color: #0d6efd !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
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
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #adb5bd;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #212529;
    }
    /* Smaller font for text-based metrics */
    .metric-text-small {
        font-size: 18px; 
        font-weight: 700;
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

def generate_pdf(ate, lower, upper, p_val, r2, treat, out):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, "Causal Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Effect of {treat} on {out}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Average Treatment Effect: {ate:.4f}", ln=True)
    pdf.cell(0, 10, f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]", ln=True)
    
    if np.isnan(p_val):
        sig = "N/A"
    else:
        sig = "Significant" if p_val < 0.05 else "Not Significant"
    
    pdf.cell(0, 10, f"Statistical Significance: {sig} (p={p_val:.4f})", ln=True)
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
    
    # Stats Model (Robust Wrapper)
    try:
        X_df = pd.DataFrame(X, index=df.index)
        X_df.columns = [f"V{i}" for i in range(X_df.shape[1])]
        X_ols = pd.concat([T.rename("Treat"), X_df], axis=1)
        X_ols = sm.add_constant(X_ols)
        ols = sm.OLS(Y, X_ols).fit()
    except:
        ols = None
    
    return est, ols, X, T

# --- SESSION STATE ---
if 'results' not in st.session_state: st.session_state['results'] = None
if 'uploaded_file' not in st.session_state: st.session_state['uploaded_file'] = None

# --- SIDEBAR NAV ---
with st.sidebar:
    st.header("Project Config")
    
    # NAVIGATION (Styled as Tabs)
    nav = st.radio("Navigation", ["Data", "Logic", "Analysis"], label_visibility="collapsed")
    st.markdown("---")

    # 1. DATA INPUTS (Show only in Data Tab)
    if nav == "Data":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            st.session_state['uploaded_file'] = uploaded_file
            raw_df = pd.read_csv(uploaded_file)
            cols = raw_df.columns.tolist()
            st.success(f"Loaded {len(raw_df)} rows")
        else:
            cols = []

    # 2. LOGIC INPUTS (Show only in Logic Tab)
    if nav == "Logic":
        if st.session_state['uploaded_file']:
            # Reload df to get columns
            raw_df = pd.read_csv(st.session_state['uploaded_file'])
            cols = raw_df.columns.tolist()
            
            treat_col = st.selectbox("Treatment Column", cols, index=0)
            out_col = st.selectbox("Outcome Column", cols, index=1 if len(cols)>1 else 0)
            
            st.markdown("---")
            use_time = st.checkbox("Enable Time Dimension")
            time_col, int_date = None, None
            if use_time:
                time_col = st.selectbox("Date Column", cols)
                try:
                    min_d = pd.to_datetime(raw_df[time_col]).min()
                    max_d = pd.to_datetime(raw_df[time_col]).max()
                    int_date = st.date_input("Intervention Date", value=min_d, min_value=min_d, max_value=max_d)
                except:
                    int_date = st.text_input("Intervention Value")
            
            excl = [treat_col, out_col]
            if time_col: excl.append(time_col)
            covs = st.multiselect("Control Variables", [c for c in cols if c not in excl])
            
            # Save config to session
            st.session_state['config'] = {
                'treat': treat_col, 'out': out_col, 'covs': covs,
                'time': time_col, 'date': int_date
            }
        else:
            st.info("Please upload data in the Data tab first.")

    # 3. ANALYSIS INPUTS (Show only in Analysis Tab)
    if nav == "Analysis":
        if st.session_state['uploaded_file'] and 'config' in st.session_state:
            run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
            
            if run_btn:
                raw_df = pd.read_csv(st.session_state['uploaded_file'])
                cfg = st.session_state['config']
                
                with st.spinner("Running Causal Models..."):
                    need = [cfg['treat'], cfg['out']] + cfg['covs']
                    if cfg['time']: need.append(cfg['time'])
                    clean, enc = preprocess_data(raw_df, need)
                    
                    ml, stats, X_t, T_t = run_analysis_logic(
                        clean, cfg['treat'], cfg['out'], cfg['covs'], cfg['time'], cfg['date']
                    )
                    
                    if ml:
                        st.session_state['results'] = {
                            'ml': ml, 'stats': stats, 'X': X_t, 'df': clean,
                            'treat': cfg['treat'], 'out': cfg['out']
                        }
            
            # PDF DOWNLOAD
            if st.session_state['results']:
                res = st.session_state['results']
                # Safe stats extraction
                ate = res['ml'].ate(res['X'])
                l, u = res['ml'].ate_interval(res['X'])
                
                if res['stats'] and "Treat" in res['stats'].pvalues:
                    p = res['stats'].pvalues["Treat"]
                    r2 = res['stats'].rsquared
                else:
                    p = np.nan
                    r2 = 0.0
                
                pdf_data = generate_pdf(ate, l, u, p, r2, res['treat'], res['out'])
                
                st.markdown("---")
                st.download_button(
                    label="DOWNLOAD PDF REPORT",
                    data=pdf_data,
                    file_name="causal_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("Configure logic in the Logic tab first.")

# --- MAIN PAGE CONTENT ---

st.title("Causal Inference Portal")

# VIEW 1: DATA TABLE
if nav == "Data":
    if st.session_state['uploaded_file']:
        df = pd.read_csv(st.session_state['uploaded_file'])
        st.subheader(f"Data Preview ({len(df)} rows)")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info("Upload a CSV file in the sidebar to view data.")

# VIEW 2: LOGIC FLOWCHART
if nav == "Logic":
    if 'config' in st.session_state:
        cfg = st.session_state['config']
        st.subheader("Logic Visualization")
        
        g = graphviz.Digraph()
        g.attr(rankdir='LR', bgcolor='transparent', margin='0')
        g.attr('node', fontname='Helvetica', shape='box', style='filled', color='white', fontcolor='#333')
        g.attr('edge', fontname='Helvetica', color='#adb5bd')
        
        g.node('T', f'Intervention\n{cfg["treat"]}', fillcolor='#d1e7dd', color='#0f5132', fontcolor='#0f5132')
        g.node('O', f'Outcome\n{cfg["out"]}', fillcolor='#cfe2ff', color='#084298', fontcolor='#084298')
        g.edge('T', 'O', label=' Impact ')
        
        if cfg['covs']:
            g.node('C', 'Controls', shape='ellipse', fillcolor='#fff3cd', color='#856404', fontcolor='#856404')
            g.edge('C', 'T', style='dashed', dir='none')
            g.edge('C', 'O', style='dashed', dir='none')
            
        st.graphviz_chart(g)
    else:
        st.info("Configure variables in the sidebar to see the logic flow.")

# VIEW 3: RESULTS
if nav == "Analysis":
    if st.session_state['results']:
        res = st.session_state['results']
        ml, stats = res['ml'], res['stats']
        
        ate = ml.ate(res['X'])
        l, u = ml.ate_interval(res['X'])
        
        # Robust P-value check
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
        
        st.subheader("Analysis Results")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">95% Range</div><div class="metric-value">[{l:.2f}, {u:.2f}]</div></div>', unsafe_allow_html=True)
        # Reduced font size for text metrics using 'metric-text-small' class
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-value metric-text-small" style="color:{sig_color}">{sig_text}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        tab1, tab2 = st.tabs(["Impact Distribution", "Stats Table"])
        res['df']['Impact'] = ml.effect(res['X'])
        
        with tab1:
            fig = px.histogram(res['df'], x='Impact', nbins=50, title="Population Impact Distribution", color_discrete_sequence=['#0d6efd'])
            fig.update_layout(plot_bgcolor='white', margin=dict(l=20, r=20, t=40, b=20))
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            if stats:
                st.text(stats.summary())
            else:
                st.warning("Statistical details unavailable (Singular Matrix or Collinearity).")
    else:
        st.info("Click 'RUN ANALYSIS' in the sidebar to generate results.")