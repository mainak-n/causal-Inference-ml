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
from datetime import datetime
import matplotlib.pyplot as plt
import tempfile
import os
import textwrap

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

def reset_analysis():
    st.session_state['results'] = None

# --- CACHING ---
@st.cache_data(max_entries=1)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# --- CSS STYLING ---
css = """
    <style>
    /* 1. FIXED HEADER */
    .header-container {
        position: sticky;
        top: 2rem;
        z-index: 800;
        background-color: white;
        margin-left: -5rem;
        margin-right: -5rem;
        padding: 1rem 5rem;
        border-bottom: 1px solid #f0f2f6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-title {
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        color: rgb(49, 51, 63);
        margin: 0;
        line-height: 1.2;
    }
    
    /* 2. LAYOUT ADJUSTMENTS */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 5rem;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
    }
    [data-testid="stSidebarNav"] { display: none; }
    
    /* ONLY pull up tabs in the sidebar */
    section[data-testid="stSidebar"] .stTabs { 
        margin-top: -30px; 
    } 

    /* 3. TABS STYLE */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        gap: 2px;
        background-color: #f0f2f6;
        padding: 4px;
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        justify-content: center;
        height: 40px;
        background-color: transparent;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 600;
        color: #555;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #ff4b4b !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    /* 4. METRIC CARDS */
    div.metric-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        height: 100%;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #9aa0a6;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #31333F;
    }
    
    /* 5. INSIGHT BOX */
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 25px;
        font-size: 15px;
        color: #31333F;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* 6. MAIN INFO BOX */
    .main-info-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 40px;
        text-align: left;
        max-width: 800px;
        margin: 0 auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* General */
    h1, h2, h3 { font-family: "Source Sans Pro", sans-serif; }
    </style>
    
    <div class="header-container">
        <div class="header-title">Causal Inference Portal</div>
    </div>
"""

if st.session_state['uploaded_file'] is None:
    css += """
    <style>
    [data-testid="stFileUploader"] button {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: #ff4b4b;
        color: white;
        background-color: #ff4b4b;
    }
    </style>
    """
st.markdown(css, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def preprocess_data(df, selected_columns, categorical_cols):
    data = df[selected_columns].copy()
    data = data.dropna()
    
    for cat in categorical_cols:
        if cat in data.columns:
            n_unique = data[cat].nunique()
            if n_unique > 100:
                raise ValueError(f"Column '{cat}' has {n_unique} unique values. Limit is 100.")

    if categorical_cols:
        valid_cats = [c for c in categorical_cols if c in data.columns]
        if valid_cats:
            data = pd.get_dummies(data, columns=valid_cats, drop_first=True, dtype=int)
            
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object' or isinstance(data[col].dtype, pd.PeriodDtype):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
            
    return data, encoders

def create_logic_graph(treat, out, covs, cats, use_time, time_col, int_date):
    g = graphviz.Digraph()
    g.attr(rankdir='LR', bgcolor='white', margin='0.2')
    g.attr('node', fontname='Helvetica', shape='box', style='rounded,filled', color='white', fontcolor='#333')
    g.attr('edge', fontname='Helvetica', color='#adb5bd')
    
    g.node('T', f'Intervention\n{treat}', fillcolor='#d1e7dd', color='#0f5132', fontcolor='#0f5132')
    g.node('O', f'Outcome\n{out}', fillcolor='#cfe2ff', color='#084298', fontcolor='#084298')
    g.edge('T', 'O', label=' Impact ', penwidth='1.5')
    
    if covs:
        vis_cats = [c for c in covs if c in cats]
        vis_nums = [c for c in covs if c not in cats]
        
        if vis_nums:
            label_num = "Numeric Controls\n" + "\n".join(vis_nums[:3])
            if len(vis_nums) > 3: label_num += "\n..."
            g.node('CN', label_num, fillcolor='#fff3cd', color='#856404', fontcolor='#856404', shape='box', style='rounded,filled')
            g.edge('CN', 'T', style='dashed', dir='none')
            g.edge('CN', 'O', style='dashed', dir='none')
            
        if vis_cats:
            label_cat = "Categorical Controls\n" + "\n".join(vis_cats[:3])
            if len(vis_cats) > 3: label_cat += "\n..."
            g.node('CC', label_cat, fillcolor='#f8d7da', color='#842029', fontcolor='#842029', shape='box', style='rounded,filled')
            g.edge('CC', 'T', style='dashed', dir='none')
            g.edge('CC', 'O', style='dashed', dir='none')

    if use_time and time_col:
         g.node('Time', f'Time Trigger\n{time_col}\n>= {int_date}', shape='note', fillcolor='#e2e3e5', color='#383d41', fontcolor='#383d41')
         g.edge('Time', 'T', label='activates')
         
    return g

def generate_pdf(ate, lower, upper, p_val, r2, treat, out, feats, impact_dist, graph_config, filename):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Causal Analysis Report: {filename}", ln=True, align='C')
    pdf.set_font("Arial", '', 8)
    pdf.cell(0, 5, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(5)
    
    # 1. Executive Summary
    pdf.set_font("Arial", 'B', 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, "1. Executive Summary", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 9)
    pdf.cell(0, 5, f"Intervention Variable: {treat}", ln=True)
    pdf.cell(0, 5, f"Target Outcome: {out}", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(95, 7, f"Average Impact (ATE): {ate:.4f}", border=1)
    pdf.cell(95, 7, f"Model Fit (R2): {r2:.4f}", border=1, ln=True)
    
    sig_txt = "Significant (p < 0.05)" if p_val < 0.05 else "Not Significant"
    pdf.cell(95, 7, f"Significance: {sig_txt}", border=1)
    pdf.cell(95, 7, f"95% CI: [{lower:.4f}, {upper:.4f}]", border=1, ln=True)
    pdf.ln(6)
    
    # 2. Logic Flowchart
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, "2. Logic Configuration (Flowchart)", ln=True, fill=True)
    pdf.ln(3)
    
    try:
        g_pdf = create_logic_graph(**graph_config)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_g:
            g_pdf.render(filename=tmp_g.name.replace('.png', ''), format='png', cleanup=True)
            pdf.image(tmp_g.name, x=65, w=80) 
    except Exception as e:
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 6, "Note: To render flowchart, ensure 'graphviz' is in packages.txt", ln=True)
    pdf.ln(5)

    # 3. Visual Impact Distribution
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, "3. Impact Distribution", ln=True, fill=True)
    pdf.ln(3)
    
    plt.figure(figsize=(5, 2.5))
    plt.hist(impact_dist, bins=30, color='#0d6efd', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title("Distribution of Causal Impact", fontsize=9)
    plt.xlabel("Impact Value", fontsize=7)
    plt.ylabel("Frequency", fontsize=7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_p:
        plt.savefig(tmp_p.name, format="png", dpi=100)
        pdf.image(tmp_p.name, x=65, w=80)
    pdf.ln(3)
    
    pdf.set_font("Arial", '', 8)
    pdf.cell(0, 5, f"Min: {impact_dist.min():.2f} | Max: {impact_dist.max():.2f} | Median: {impact_dist.median():.2f}", ln=True)
    pdf.ln(6)

    # 4. Top Drivers
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, "4. Top Drivers of Impact", ln=True, fill=True)
    pdf.ln(3)
    
    pdf.set_font("Arial", '', 9)
    if not feats.empty:
        pdf.cell(0, 5, "Most influential variables:", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(120, 6, "Variable Name", border=1)
        pdf.cell(40, 6, "Importance", border=1, ln=True)
        pdf.set_font("Arial", '', 8)
        for index, row in feats.head(8).iterrows():
            pdf.cell(120, 6, str(row['Feature']), border=1)
            pdf.cell(40, 6, f"{row['Importance']:.4f}", border=1, ln=True)
    else:
        pdf.cell(0, 5, "No control variables were used.", ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

def run_analysis_logic(df, treatment, outcome, controls):
    X_cols = [c for c in df.columns if c not in [treatment, outcome, 'Is_Post']]
    
    if not X_cols:
        X = np.zeros((len(df), 1))
        features = ["No_Controls"]
    else:
        X = df[X_cols]
        features = X_cols
    
    Y = df[outcome]
    
    if 'Is_Post' in df.columns:
        T = df[treatment] * df['Is_Post']
        X = pd.concat([X, df[treatment].rename("Group_Effect"), df['Is_Post'].rename("Time_Effect")], axis=1)
        features = features + ["Group_Effect", "Time_Effect"]
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
    
    effects = est.effect(X)
    interpreter = RandomForestRegressor(max_depth=4)
    interpreter.fit(X, effects)
    importances = pd.DataFrame({
        'Feature': features, 
        'Importance': interpreter.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return est, ols, X, T, importances

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    tab_data, tab_logic, tab_run = st.tabs(["Data", "Logic", "Action"])

    # 1. DATA TAB
    with tab_data:
        btn_type = "primary" if st.session_state['active_tab'] != "Data" else "secondary"
        st.button("Show Table View", type=btn_type, use_container_width=True, on_click=set_view, args=("Data",))
            
        uploaded_file = st.file_uploader("Upload CSV", type="csv", help="Max file size: 200MB")
        
        if uploaded_file:
            # Load and Cache Data
            raw_df = load_data(uploaded_file)
            st.session_state['uploaded_file'] = uploaded_file 
            
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
            
            st.markdown("---")
            
            t_num, t_cat = st.tabs(["123 Numerical", "Abc Categorical"])
            
            excl = [treat_col, out_col]
            if time_col: excl.append(time_col)
            available_cols = [c for c in cols if c not in excl]
            
            with t_num:
                num_covs = st.multiselect("Select Numeric Controls", available_cols, default=[])
            
            with t_cat:
                cat_covs = st.multiselect("Select Categorical Controls", available_cols, default=[], help="Max 100 unique values.")

            covs = list(set(num_covs + cat_covs))
            cats = cat_covs 

        else:
            st.info("Upload data first")

    # 3. ACTION TAB
    with tab_run:
        if uploaded_file:
            if st.button("RUN NEW ANALYSIS", type="primary", use_container_width=True):
                st.session_state['active_tab'] = "Action"
                try:
                    with st.spinner("Calculating Impact..."):
                        prep_df = raw_df.copy()
                        if use_time and time_col and int_date:
                            try:
                                ts = pd.to_datetime(prep_df[time_col])
                                ids = pd.to_datetime(int_date)
                                prep_df['Is_Post'] = (ts >= ids).astype(int)
                            except:
                                prep_df['Is_Post'] = 0
                        
                        need = [treat_col, out_col] + covs
                        if 'Is_Post' in prep_df.columns: need.append('Is_Post')
                        
                        clean, enc = preprocess_data(prep_df, need, cats)
                        ml, stats, X_t, T_t, feats = run_analysis_logic(clean, treat_col, out_col, covs)
                        
                        if ml:
                            st.session_state['results'] = {
                                'ml': ml, 'stats': stats, 'X': X_t, 'df': clean,
                                'treat': treat_col, 'out': out_col, 'feats': feats,
                                'graph_config': {
                                    'treat': treat_col, 'out': out_col, 'covs': covs, 'cats': cats,
                                    'use_time': use_time, 'time_col': time_col, 'int_date': int_date
                                }
                            }
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
            
            if st.session_state['results'] is not None and st.session_state['active_tab'] != "Action":
                 st.button("Show Previous Analysis", type="secondary", use_container_width=True, on_click=set_view, args=("Action",))
            
            st.button("Reset / Stop Analysis", type="secondary", use_container_width=True, on_click=reset_analysis)
            
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
                
                impact_dist = res['ml'].effect(res['X'])
                
                # Get Filename safely
                fname = st.session_state['uploaded_file'].name
                
                pdf_data = generate_pdf(ate, l, u, p, r2, res['treat'], res['out'], res['feats'], pd.Series(impact_dist), res['graph_config'], fname)
                st.download_button("DOWNLOAD PDF REPORT", pdf_data, "causal_report.pdf", "application/pdf", use_container_width=True)

# --- MAIN PAGE ---

if st.session_state['active_tab'] == "Data":
    if st.session_state['uploaded_file']:
        fname = st.session_state['uploaded_file'].name
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0; padding: 0; font-family: 'Source Sans Pro', sans-serif; font-weight: 600; color: #31333F;">Data Inspector</h3>
            <span style="color: #adb5bd; font-size: 1.2rem; font-weight: 400; padding-top: 2px;">: {fname}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(raw_df.head(100), use_container_width=True)
    else:
        # Use simple markdown for the main landing page to ensure clean rendering
        st.markdown("""
        <div style="background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 40px; text-align: left; max-width: 800px; margin: 0 auto; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <div style="font-size: 24px; font-weight: 700; color: #31333F; margin-bottom: 15px; text-align: center;">Welcome to the Causal Inference Portal</div>
            <div style="font-size: 16px; color: #555; line-height: 1.6; margin-bottom: 25px;">
                This tool allows you to measure the <b>true impact</b> of interventions (like marketing campaigns, feature launches, or policy changes) by separating cause from correlation using advanced <b>Double Machine Learning</b>.
            </div>
            
            <div style="font-weight: 600; color: #31333F; font-size: 16px; margin-bottom: 15px;">📋 Required Data Format (CSV):</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Standard Markdown Table (Outside HTML block for safety)
        st.markdown("""
        | Treatment (0/1) | Outcome ($) | Control 1 (Age) | Control 2 (Region) |
        | :--- | :--- | :--- | :--- |
        | 1 | 120.50 | 25 | North |
        | 0 | 85.00 | 32 | South |
        | 1 | 135.20 | 45 | East |
        """)
        
        st.markdown("""
        1. **Treatment Column:** 0 or 1 (Who got the intervention?)
        2. **Outcome Column:** Numeric (Sales, clicks, retention)
        3. **Control Variables:** User attributes (Age, Region, etc.)
        """)

elif st.session_state['active_tab'] == "Logic":
    if st.session_state['uploaded_file']:
        fname = st.session_state['uploaded_file'].name
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0; padding: 0; font-family: 'Source Sans Pro', sans-serif; font-weight: 600; color: #31333F;">Logic Visualization</h3>
            <span style="color: #adb5bd; font-size: 1.2rem; font-weight: 400; padding-top: 2px;">: {fname}</span>
        </div>
        """, unsafe_allow_html=True)
        
        g = create_logic_graph(treat_col, out_col, covs, cats, use_time, time_col, int_date)
        st.graphviz_chart(g)
    else:
        st.warning("Upload data first.")

elif st.session_state['active_tab'] == "Action":
    if st.session_state['results']:
        res = st.session_state['results']
        ml, stats = res['ml'], res['stats']
        feats = res['feats']
        
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
        
        fname = st.session_state['uploaded_file'].name
        
        # Header with Filename on SAME LINE
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0; padding: 0; font-family: 'Source Sans Pro', sans-serif; font-weight: 600; color: #31333F;">Analysis Results</h3>
            <span style="color: #adb5bd; font-size: 1.2rem; font-weight: 400; padding-top: 2px;">: {fname}</span>
        </div>
        """, unsafe_allow_html=True)
        
        direction = "INCREASE" if ate > 0 else "DECREASE"
        sig_phrase = "statistically significant" if is_sig else "not statistically conclusive"
        
        st.markdown(f"""
        <div class="insight-box">
            <b>💡 Automated Insight:</b><br>
            The intervention led to an average <b>{direction}</b> of <b>{abs(ate):.2f}</b> in <b>{out_col}</b>. 
            This result is <b>{sig_phrase}</b> (Confidence: {100*(1-p):.1f}%). 
            The model explains <b>{r2:.1%}</b> of the variation in the data.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: 
             st.markdown(f'<div class="metric-container"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
        with c2: 
             st.markdown(f'<div class="metric-container"><div class="metric-label">95% Range</div><div class="metric-value">[{l:.2f}, {u:.2f}]</div></div>', unsafe_allow_html=True)
        with c3: 
             st.markdown(f'<div class="metric-container"><div class="metric-label">Certainty</div><div class="metric-value" style="color:{sig_color}">{sig_text}</div></div>', unsafe_allow_html=True)
        with c4: 
             st.markdown(f'<div class="metric-container"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
        
        # SPACER to prevent overlap
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        
        t1, t2, t3, t4 = st.tabs(["📉 Impact Distribution", "🧠 Drivers of Impact", "🔍 Segment Analysis", "📊 Stats Table"])
        
        res['df']['Impact'] = ml.effect(res['X'])
        
        with t1:
            st.caption("Shows how the impact varies across the population.")
            fig = px.histogram(res['df'], x='Impact', nbins=50, color_discrete_sequence=['#0d6efd'])
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            st.caption("Which variables contribute most to the outcome change?")
            if not feats.empty:
                fig2 = px.bar(feats.head(10), x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#0d6efd'])
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No controls used, so no feature drivers available.")

        with t3:
            st.caption("Does the impact depend on a specific variable?")
            if not feats.empty:
                seg_var = st.selectbox("Select Variable to Segment By:", feats['Feature'].unique())
                if seg_var in res['df'].columns:
                    fig3 = px.scatter(res['df'], x=seg_var, y='Impact', title=f"Impact vs {seg_var}", color='Impact', color_continuous_scale='RdBu')
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.warning(f"Variable '{seg_var}' was processed (e.g., One-Hot Encoded) and cannot be plotted directly here.")
            else:
                st.info("No variables available for segmentation.")

        with t4:
            if stats:
                st.text(stats.summary())
            else:
                st.warning("Statistical model unavailable.")
    else:
        st.info("Configure logic and click Run Analysis in the sidebar.")