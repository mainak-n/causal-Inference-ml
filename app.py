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

# --- STATE MANAGEMENT ---
if 'results' not in st.session_state: st.session_state['results'] = None
if 'uploaded_file' not in st.session_state: st.session_state['uploaded_file'] = None
if 'active_tab' not in st.session_state: st.session_state['active_tab'] = "Data"

def set_view(view_name):
    st.session_state['active_tab'] = view_name

# --- DYNAMIC CSS FOR UPLOADER ---
# This injects red styling only if no file is uploaded
uploader_style = ""
if st.session_state['uploaded_file'] is None:
    uploader_style = """
    /* Target the Uploader Container when empty */
    [data-testid="stFileUploader"] section {
        border: 2px dashed #ff4b4b !important;
        background-color: #fff0f0 !important;
    }
    /* Target the "Browse files" button text to make it redish to stand out */
    [data-testid="stFileUploader"] button {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
    }
    """

# --- PROFESSIONAL CSS ---
st.markdown(f"""
    <style>
    {uploader_style}
    
    /* 1. FIXED HEADER (Clean & Aligned) */
    .header-container {{
        position: fixed;
        top: 3.75rem;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        z-index: 999;
        padding: 15px 40px; /* Reduced vertical padding */
        border-bottom: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 60px; /* Slimmer header */
        display: flex;
        align-items: center;
        /* Ensure text isn't hidden behind sidebar */
        padding-left: 22rem; 
    }}
    .header-title {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; /* Matched font */
        font-size: 20px;
        font-weight: 700;
        color: #212529;
        margin: 0;
    }}
    
    /* 2. MAIN CONTENT PADDING */
    .block-container {{
        padding-top: 9rem !important;
    }}

    /* 3. SIDEBAR ALIGNMENT */
    [data-testid="stSidebar"] {{
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
        padding-top: 1rem; /* Aligns with header */
    }}

    /* 4. TABS FILL WIDTH */
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        width: 100%;
        gap: 2px;
        background-color: #e9ecef;
        padding: 4px;
        border-radius: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        flex-grow: 1;
        justify-content: center;
        height: 40px;
        background-color: transparent;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 600;
        color: #495057;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #ffffff !important;
        color: #0d6efd !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }}

    /* 5. METRIC CARDS */
    .metric-card {{
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    .metric-label {{
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        color: #adb5bd;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }}
    .metric-value {{
        font-size: 20px;
        font-weight: 700;
        color: #212529;
    }}

    @media (max-width: 992px) {{
        .header-container {{ padding-left: 60px; }}
    }}
    </style>
    
    <div class="header-container">
        <div class="header-title">Causal Inference Portal</div>
    </div>
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
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 20, "Causal Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Analysis: Effect of '{treat}' on '{out}'", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Average Treatment Effect (ATE): {ate:.4f}", ln=True)
    pdf.cell(0, 10, f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]", ln=True)
    
    if np.isnan(p_val):
        sig_txt = "Could not calculate"
    else:
        sig_txt = "Significant" if p_val < 0.05 else "Not Significant"
        
    pdf.cell(0, 10, f"Statistical Significance: {sig_txt} (p={p_val:.4f})", ln=True)
    pdf.cell(0, 10, f"Model Fit (R-Squared): {r2:.4f}", ln=True)
    
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
    
    try:
        X_df = pd.DataFrame(X, index=df.index)
        X_df.columns = [f"V{i}" for i in range(X_df.shape[1])]
        X_ols = pd.concat([T.rename("Treat"), X_df], axis=1)
        X_ols = sm.add_constant(X_ols) 
        ols = sm.OLS(Y, X_ols).fit()
    except:
        ols = None
    
    return est, ols, X, T

# --- SIDEBAR ---
with st.sidebar:
    # No Spacer needed now as padding is adjusted in CSS to align with header
    
    # TACTICAL TABS
    tab_data, tab_logic, tab_run = st.tabs(["Data", "Logic", "Action"])

    # 1. DATA TAB
    with tab_data:
        btn_type = "primary" if st.session_state['active_tab'] != "Data" else "secondary"
        st.button("Show Table View", type=btn_type, use_container_width=True, on_click=set_view, args=("Data",))
        
        # This uploader will be Red if empty (via CSS above)
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
        btn_type = "primary" if st.session_state['active_tab'] != "Logic" else "secondary"
        st.button("Visualize Logic Flow", type=btn_type, use_container_width=True, on_click=set_view, args=("Logic",))
            
        if uploaded_file:
            treat_col = st.selectbox("Treatment Column", cols, index=0)
            out_col = st.selectbox("Outcome Column", cols, index=1 if len(cols)>1 else 0)
            
            st.markdown("---")
            use_time = st.checkbox("Enable Time Logic")
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
        else:
            st.info("Upload data first")

    # 3. ACTION TAB
    with tab_run:
        if uploaded_file:
            if st.session_state['results'] is not None:
                prev_btn_type = "primary" if st.session_state['active_tab'] != "Action" else "secondary"
                st.button("Show Previous Analysis", type=prev_btn_type, use_container_width=True, on_click=set_view, args=("Action",))
            
            if st.button("RUN NEW ANALYSIS", type="primary", use_container_width=True):
                st.session_state['active_tab'] = "Action"
                with st.spinner("Calculating Impact..."):
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
                ate = res['ml'].ate(res['X'])
                l, u = res['ml'].ate_interval(res['X'])
                
                if res['stats'] and "Treat" in res['stats'].pvalues:
                    p = res['stats'].pvalues["Treat"]
                    r2 = res['stats'].rsquared
                else:
                    p = np.nan
                    r2 = 0.0
                
                pdf_data = generate_pdf(ate, l, u, p, r2, res['treat'], res['out'])
                
                st.download_button(
                    label="DOWNLOAD PDF REPORT",
                    data=pdf_data,
                    file_name="causal_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# --- MAIN PAGE RENDERING ---

# VIEW 1: DATA
if st.session_state['active_tab'] == "Data":
    if st.session_state['uploaded_file']:
        st.subheader("Data Inspector")
        st.dataframe(raw_df.head(100), use_container_width=True)
    else:
        st.info("Upload a CSV file in the sidebar Data tab.")

# VIEW 2: LOGIC
elif st.session_state['active_tab'] == "Logic":
    if st.session_state['uploaded_file']:
        st.subheader("Logic Visualization")
        
        g = graphviz.Digraph()
        g.attr(rankdir='LR', bgcolor='transparent', margin='0')
        g.attr('node', fontname='Helvetica', shape='box', style='filled', color='white', fontcolor='#333')
        g.attr('edge', fontname='Helvetica', color='#adb5bd')
        
        g.node('T', f'Intervention\n{treat_col}', fillcolor='#d1e7dd', color='#0f5132', fontcolor='#0f5132')
        g.node('O', f'Outcome\n{out_col}', fillcolor='#cfe2ff', color='#084298', fontcolor='#084298')
        g.edge('T', 'O', label=' Impact ')
        
        if covs:
            g.node('C', 'Controls', shape='ellipse', fillcolor='#fff3cd', color='#856404', fontcolor='#856404')
            g.edge('C', 'T', style='dashed', dir='none')
            g.edge('C', 'O', style='dashed', dir='none')
            
        st.graphviz_chart(g)
    else:
        st.warning("Upload data first.")

# VIEW 3: ACTION
elif st.session_state['active_tab'] == "Action":
    if st.session_state['results']:
        res = st.session_state['results']
        ml, stats = res['ml'], res['stats']
        
        ate = ml.ate(res['X'])
        l, u = ml.ate_interval(res['X'])
        
        if stats and "Treat" in stats.pvalues:
            p = stats.pvalues["Treat"]
            r2 = stats.rsquared
            is_sig = p < 0.05
            sig_color = "#198754" if is_sig else "#dc3545"
            sig_text = "Significant" if is_sig else "Inconclusive"
        else:
            p = np.nan
            r2 = 0.0
            sig_color = "#6c757d"
            sig_text = "N/A"
        
        st.subheader("Analysis Results")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">95% Range</div><div class="metric-value">[{l:.2f}, {u:.2f}]</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-value" style="color:{sig_color}">{sig_text}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        t1, t2 = st.tabs(["Impact Distribution", "Stats Table"])
        res['df']['Impact'] = ml.effect(res['X'])
        
        with t1:
            fig = px.histogram(res['df'], x='Impact', nbins=50, title="Impact Distribution", color_discrete_sequence=['#0d6efd'])
            fig.update_layout(plot_bgcolor='white', margin=dict(l=20, r=20, t=40, b=20))
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            if stats:
                st.text(stats.summary())
            else:
                st.warning("Statistical model unavailable.")
    else:
        st.info("Configure logic and click Run Analysis in the sidebar.")