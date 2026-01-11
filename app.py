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

# --- CSS STYLING ---
css = """
    <style>
    /* 1. STICKY HEADER (Dynamic Centering) */
    .header-container {
        position: sticky;
        top: 2rem; /* Sticks below the standard Streamlit decoration bar */
        z-index: 800; /* High enough to sit over content, low enough for dropdowns */
        background-color: white;
        
        /* Extend to edges of the container */
        margin-left: -5rem;
        margin-right: -5rem;
        padding: 1rem 5rem;
        
        border-bottom: 1px solid #f0f2f6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Font Matching "Data Inspector" (Streamlit Subheader style) */
    .header-title {
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 600;
        font-size: 1.5rem; /* ~24px */
        color: rgb(49, 51, 63);
        margin: 0;
        line-height: 1.2;
    }
    
    /* 2. ADJUST TOP PADDING FOR CONTENT */
    .block-container {
        padding-top: 3rem !important; /* Reduced since we use sticky, not fixed */
        padding-bottom: 5rem;
    }

    /* 3. SIDEBAR ALIGNMENT */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
    }
    [data-testid="stSidebarNav"] { display: none; }
    .stTabs { margin-top: -20px; }

    /* 4. TABS STYLE */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        gap: 2px;
        background-color: #f0f2f6; /* Lighter gray */
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
        color: #ff4b4b !important; /* Streamlit Red/Primary */
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    /* 5. MAIN PAGE INFO BOX */
    .main-info-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .info-icon { font-size: 40px; margin-bottom: 20px; }
    .info-header { font-size: 24px; font-weight: 700; color: #31333F; margin-bottom: 15px; }
    .info-text { font-size: 16px; color: #6c757d; line-height: 1.6; }
    .info-list { text-align: left; display: inline-block; margin-top: 20px; color: #31333F; }

    /* General */
    h1, h2, h3 { font-family: "Source Sans Pro", sans-serif; }
    </style>
    
    <div class="header-container">
        <div class="header-title">Causal Inference Portal</div>
    </div>
"""

# Dynamic CSS for Uploader
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

def generate_pdf(ate, lower, upper, p_val, r2, treat, out, feats, impact_dist):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, "Causal Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    # 1. Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, "1. Executive Summary", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Intervention Variable: {treat}", ln=True)
    pdf.cell(0, 8, f"Target Outcome: {out}", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(95, 10, f"Average Impact (ATE): {ate:.4f}", border=1)
    pdf.cell(95, 10, f"Model Fit (R2): {r2:.4f}", border=1, ln=True)
    sig_txt = "Significant (p < 0.05)" if p_val < 0.05 else "Not Significant"
    pdf.cell(95, 10, f"Significance: {sig_txt}", border=1)
    pdf.cell(95, 10, f"95% CI: [{lower:.4f}, {upper:.4f}]", border=1, ln=True)
    pdf.ln(10)
    
    # 2. Visual Impact Distribution
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Impact Distribution (Visual)", ln=True, fill=True)
    pdf.ln(2)
    
    # Generate Chart using Matplotlib
    plt.figure(figsize=(6, 3))
    plt.hist(impact_dist, bins=30, color='#0d6efd', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title("Distribution of Causal Impact")
    plt.xlabel("Impact Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, format="png", dpi=100)
        tmp_path = tmp_file.name
    
    # Embed in PDF
    pdf.image(tmp_path, x=10, w=190)
    pdf.ln(5)
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    # Stats Text
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Min: {impact_dist.min():.2f} | Max: {impact_dist.max():.2f} | Median: {impact_dist.median():.2f}", ln=True)
    pdf.ln(10)

    # 3. Top Drivers
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Top Drivers of Impact", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 11)
    if not feats.empty:
        pdf.cell(0, 8, "Most influential variables:", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(140, 8, "Variable Name", border=1)
        pdf.cell(50, 8, "Importance Score", border=1, ln=True)
        pdf.set_font("Arial", '', 10)
        for index, row in feats.head(8).iterrows():
            pdf.cell(140, 8, str(row['Feature']), border=1)
            pdf.cell(50, 8, f"{row['Importance']:.4f}", border=1, ln=True)
    else:
        pdf.cell(0, 8, "No control variables were used.", ln=True)
        
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
        # Run Button
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
                            'treat': treat_col, 'out': out_col, 'feats': feats
                        }
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
        
        # Show Previous (Only if results exist AND we are NOT on Action view)
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
            pdf_data = generate_pdf(ate, l, u, p, r2, res['treat'], res['out'], res['feats'], pd.Series(impact_dist))
            st.download_button("DOWNLOAD PDF REPORT", pdf_data, "causal_report.pdf", "application/pdf", use_container_width=True)

# --- MAIN PAGE ---

if st.session_state['active_tab'] == "Data":
    if st.session_state['uploaded_file']:
        st.subheader("Data Inspector")
        st.dataframe(raw_df.head(100), use_container_width=True)
    else:
        st.markdown("""
        <div class="main-info-box">
            <div class="info-icon">👋</div>
            <div class="info-header">Welcome to the Causal Inference Portal</div>
            <div class="info-text">
                This tool allows you to measure the <b>true impact</b> of interventions (like marketing campaigns, feature launches, or policy changes) by separating cause from correlation using advanced <b>Double Machine Learning</b>.
            </div>
            <div class="info-list">
                <b>📋 Required Data Format (CSV):</b><br>
                1. <b>Treatment Column:</b> 0/1 or True/False (Who got the intervention?)<br>
                2. <b>Outcome Column:</b> Numeric (Sales, clicks, retention)<br>
                3. <b>Control Variables:</b> User details (Age, Region, etc.)
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            vis_cats = [c for c in covs if c in cats]
            vis_nums = [c for c in covs if c not in cats]
            if vis_nums:
                label_num = "Numeric Controls\n" + "\n".join(vis_nums[:3])
                if len(vis_nums) > 3: label_num += "\n..."
                g.node('CN', label_num, shape='ellipse', fillcolor='#fff3cd', color='#856404', fontcolor='#856404')
                g.edge('CN', 'T', style='dashed', dir='none')
                g.edge('CN', 'O', style='dashed', dir='none')
            if vis_cats:
                label_cat = "Categorical Controls\n" + "\n".join(vis_cats[:3])
                if len(vis_cats) > 3: label_cat += "\n..."
                g.node('CC', label_cat, shape='ellipse', fillcolor='#f8d7da', color='#842029', fontcolor='#842029')
                g.edge('CC', 'T', style='dashed', dir='none')
                g.edge('CC', 'O', style='dashed', dir='none')

        if use_time and time_col:
             g.node('Time', f'Time Trigger\n{time_col}\n>= {int_date}', shape='note', fillcolor='#e2e3e5', color='#383d41', fontcolor='#383d41')
             g.edge('Time', 'T', label='activates')
            
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
        
        st.subheader("Analysis Results")
        
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
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">95% Range</div><div class="metric-value">[{l:.2f}, {u:.2f}]</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Certainty</div><div class="metric-value" style="color:{sig_color}">{sig_text}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
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
        # If result is None, but user is on Action tab, show prompt
        st.info("Configure logic and click Run Analysis in the sidebar.")