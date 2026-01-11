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

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* 1. FIXED HEADER (Clean & Aligned) */
    .header-container {
        position: fixed;
        top: 3.75rem;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        z-index: 999;
        padding: 15px 40px;
        border-bottom: 1px solid #e0e0e0;
        height: 70px; /* Reduced height since subtitle is gone */
        display: flex;
        align-items: center;
        padding-left: 22rem; 
    }
    .header-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; /* Same as body */
        font-size: 20px;
        font-weight: 700;
        color: #212529;
        margin: 0;
    }
    
    /* 2. PUSH MAIN CONTENT DOWN */
    .block-container {
        padding-top: 9rem !important;
    }

    /* 3. SIDEBAR ALIGNMENT (Push Up) */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem; /* Aligns with the header */
    }

    /* 4. RED UPLOAD BUTTON (CSS Hack) */
    /* Target the button inside the uploader when it's in 'secondary' state (default) */
    [data-testid="stFileUploader"] button {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: #ff4b4b;
        color: white;
        background-color: #ff4b4b;
    }

    /* 5. TABS */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
        gap: 2px;
        background-color: #e9ecef;
        padding: 4px;
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        justify-content: center;
        height: 40px;
        background-color: transparent;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 600;
        color: #495057;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0d6efd !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    /* 6. METRIC CARDS */
    .metric-card {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-label {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        color: #adb5bd;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #212529;
    }

    h1, h2, h3 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #212529; }
    </style>
    
    <div class="header-container">
        <div class="header-title">Causal Inference Portal</div>
    </div>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def preprocess_data(df, selected_columns, categorical_cols):
    """
    Handles numeric and categorical data properly.
    """
    data = df[selected_columns].copy()
    data = data.dropna()
    
    # 1. One-Hot Encode user-defined categoricals
    if categorical_cols:
        # Check if selected categoricals are actually in the subset
        valid_cats = [c for c in categorical_cols if c in data.columns]
        if valid_cats:
            data = pd.get_dummies(data, columns=valid_cats, drop_first=True, dtype=int)
            
    # 2. Label Encode remaining object columns (fallback)
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
    # Prepare X (Confounders)
    # Filter columns that are NOT treatment or outcome (handles the dummy columns created by OHE)
    X_cols = [c for c in df.columns if c not in [treatment, outcome, 'Is_Post']]
    
    if not X_cols:
        X = np.zeros((len(df), 1))
        features = ["No_Controls"]
    else:
        X = df[X_cols]
        features = X_cols
    
    Y = df[outcome]
    
    if time_col and date_val:
        try:
            # We need to access the original time column for the mask
            # But 'df' here is already processed. We pass 'time_col' in original df logic usually.
            # Here we assume time_col was kept or we handle the interaction differently.
            # Simplified: We assume user handled time split or we don't have the raw time col in 'df' anymore if it was encoded.
            # FIX: We rely on the mask created BEFORE preprocessing or recreate it if time_col is numeric.
            
            # Since preprocessing might have dropped/encoded the time column, robust approach:
            # We assume time_col logic was handled in the 'Is_Post' creation step or T calculation.
            # For this simplified app, we'll assume T is passed correctly or created here if possible.
            
            # Re-creating T based on the interaction logic requires the Time info.
            # NOTE: In robust systems, we'd pass the 'Is_Post' mask separately.
            # Here, we will try to find 'Is_Post' if it exists, or fail gracefully.
            pass
        except:
            return None, None, None, None
            
    # For DiD, we usually construct T beforehand or handle it here. 
    # To keep this function clean, we will calculate T *outside* and pass it, or calculate it here if 'Is_Post' exists.
    
    if 'Is_Post' in df.columns:
        T = df[treatment] * df['Is_Post']
        # Add main effects to X
        X = pd.concat([X, df[treatment].rename("Group_Effect"), df['Is_Post'].rename("Time_Effect")], axis=1)
        features = features + ["Group_Effect", "Time_Effect"]
    else:
        T = df[treatment]

    # ML Model
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=50, max_depth=6),
        discrete_treatment=True
    )
    est.fit(Y, T, X=X)
    
    try:
        X_df = pd.DataFrame(X, index=df.index)
        # Ensure unique columns
        X_df.columns = [f"V{i}" for i in range(X_df.shape[1])]
        X_ols = pd.concat([T.rename("Treat"), X_df], axis=1)
        X_ols = sm.add_constant(X_ols) 
        ols = sm.OLS(Y, X_ols).fit()
    except:
        ols = None
    
    return est, ols, X, T

# --- SIDEBAR ---
with st.sidebar:
    # TACTICAL TABS
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
            
            # Exclude treatment/outcome from controls
            excl = [treat_col, out_col]
            if time_col: excl.append(time_col)
            possible_controls = [c for c in cols if c not in excl]
            
            # 1. Select Controls
            covs = st.multiselect("Control Variables", possible_controls)
            
            # 2. Select Categoricals (Subset of Controls)
            cats = []
            if covs:
                cats = st.multiselect("Which of these are Categorical?", covs, help="Select variables like Region, Gender, etc. to One-Hot Encode them.")

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
                    
                    # 1. Preprocess (Handling Time Logic inside dataframe prep)
                    prep_df = raw_df.copy()
                    
                    # Create Is_Post mask BEFORE encoding if time exists
                    if use_time and time_col and int_date:
                        try:
                            ts = pd.to_datetime(prep_df[time_col])
                            ids = pd.to_datetime(int_date)
                            prep_df['Is_Post'] = (ts >= ids).astype(int)
                        except:
                            prep_df['Is_Post'] = 0 # Fallback
                    
                    # Define columns to keep
                    need = [treat_col, out_col] + covs
                    if 'Is_Post' in prep_df.columns: need.append('Is_Post')
                    
                    # Preprocess (OHE)
                    clean, enc = preprocess_data(prep_df, need, cats)
                    
                    ml, stats, X_t, T_t = run_analysis_logic(clean, treat_col, out_col, covs)
                    
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
        
        # Treatment & Outcome
        g.node('T', f'Intervention\n{treat_col}', fillcolor='#d1e7dd', color='#0f5132', fontcolor='#0f5132')
        g.node('O', f'Outcome\n{out_col}', fillcolor='#cfe2ff', color='#084298', fontcolor='#084298')
        g.edge('T', 'O', label=' Impact ')
        
        # Controls (Numeric vs Categorical)
        if covs:
            # Separate cats vs nums for visualization
            cat_vars = cats
            num_vars = [c for c in covs if c not in cats]
            
            if num_vars:
                label_num = "Numeric Controls\n" + "\n".join(num_vars[:3])
                if len(num_vars) > 3: label_num += "\n..."
                g.node('CN', label_num, shape='ellipse', fillcolor='#fff3cd', color='#856404', fontcolor='#856404')
                g.edge('CN', 'T', style='dashed', dir='none')
                g.edge('CN', 'O', style='dashed', dir='none')
                
            if cat_vars:
                label_cat = "Categorical Controls\n(One-Hot Encoded)\n" + "\n".join(cat_vars[:3])
                if len(cat_vars) > 3: label_cat += "\n..."
                g.node('CC', label_cat, shape='ellipse', fillcolor='#f8d7da', color='#842029', fontcolor='#842029')
                g.edge('CC', 'T', style='dashed', dir='none')
                g.edge('CC', 'O', style='dashed', dir='none')

        # Time Logic
        if use_time and time_col:
             g.node('Time', f'Time Trigger\n{time_col}\n>= {int_date}', shape='note', fillcolor='#e2e3e5', color='#383d41', fontcolor='#383d41')
             g.edge('Time', 'T', label='activates')
            
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