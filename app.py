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
import matplotlib.dates as mdates  # Added for PDF chart formatting
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
    
    /* 7. EXAMPLE TABLE STYLE */
    .example-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        font-family: "Source Sans Pro", sans-serif;
        color: #333;
        background-color: #fff;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .example-table th {
        background-color: #f8f9fa;
        color: #555;
        font-weight: 600;
        text-align: left;
        padding: 12px;
        border-bottom: 2px solid #e0e0e0;
    }
    .example-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #f0f0f0;
    }
    .example-table tr:last-child td { border-bottom: none; }

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
    """Basic preprocessing: drop NaNs and One-Hot Encoding."""
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
        # Check if the column is NOT a datetime before trying to label encode it.
        if (data[col].dtype == 'object' or isinstance(data[col].dtype, pd.PeriodDtype)) and not pd.api.types.is_datetime64_any_dtype(data[col]):
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

def generate_pdf(ate, lower, upper, p_val, r2, treat, out, feats, impact_dist, graph_config, filename, df):
    """
    Generates a PDF report. 
    If Time Logic is ON: Replaces Impact Distribution Histogram with Treatment vs Control Trend Line Chart.
    """
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
    
    sig_txt = "Significant" if (lower > 0 or upper < 0) else "Inconclusive"
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

    # 3. Visual Section (CONDITIONAL)
    pdf.set_font("Arial", 'B', 11)
    
    # Check if Time Logic was used and the column exists
    if graph_config['use_time'] and graph_config['time_col'] in df.columns:
        # --- SHOW TREATMENT VS CONTROL TRENDS ---
        pdf.cell(0, 7, "3. Treatment vs Control Trends", ln=True, fill=True)
        pdf.ln(3)
        
        plt.figure(figsize=(6, 3))
        
        # Prepare Data for Plotting
        plot_df = df.copy()
        time_c = graph_config['time_col']
        
        # Ensure date format for plotting using dayfirst=True
        try:
            plot_df[time_c] = pd.to_datetime(plot_df[time_c], dayfirst=True)
        except:
            pass # Keep as is if conversion fails
            
        # Aggregate
        trend = plot_df.groupby([time_c, treat])[out].mean().reset_index()
        
        # Plot using standard Matplotlib
        # Treated
        treated_data = trend[trend[treat] == 1]
        plt.plot(treated_data[time_c], treated_data[out], label='Treated', color='#28a745', marker='.', linewidth=2)
        
        # Control
        control_data = trend[trend[treat] == 0]
        plt.plot(control_data[time_c], control_data[out], label='Control', color='#6c757d', marker='.', linewidth=2)
        
        plt.title("Parallel Trends Check", fontsize=10)
        plt.xlabel("Time", fontsize=8)
        plt.ylabel(out, fontsize=8)
        plt.legend(fontsize=8)
        plt.xticks(fontsize=7, rotation=45)
        
        # --- FIX: Set Date Format for PDF Chart ---
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        
        plt.yticks(fontsize=7)
        plt.grid(color='#f0f0f0', linestyle='--')
        plt.tight_layout()
        
    else:
        # --- SHOW IMPACT DISTRIBUTION (HISTOGRAM) ---
        pdf.cell(0, 7, "3. Impact Distribution", ln=True, fill=True)
        pdf.ln(3)
        
        plt.figure(figsize=(5, 2.5))
        if isinstance(impact_dist, (int, float)):
            # If scalar (DiD but missing time col in df), draw line
            plt.axvline(x=impact_dist, color='#0d6efd', linewidth=4, label='Impact')
            plt.xlim(impact_dist - 10, impact_dist + 10)
        else:
            plt.hist(impact_dist, bins=30, color='#0d6efd', alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--')
        
        plt.title("Distribution of Causal Impact", fontsize=9)
        plt.xlabel("Impact Value", fontsize=7)
        plt.ylabel("Frequency", fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
    
    # Save Plot to Temp File for PDF insertion
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_p:
        plt.savefig(tmp_p.name, format="png", dpi=100)
        pdf.image(tmp_p.name, x=65, w=80)
    pdf.ln(3)
    
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

def run_analysis_logic(df, treatment, outcome, controls, time_col=None):
    """
    Robust logic for Causal Inference.
    - If Time Logic: Uses Statsmodels DiD Regression (OLS) for robustness.
    - If No Time Logic: Uses CausalForestDML for Heterogeneity.
    """
    
    # 1. Feature Engineering (For ML controls)
    if time_col and time_col in df.columns:
        try:
            # FIX: Ensure proper date parsing with dayfirst
            df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
            df['Month'] = df[time_col].dt.month
            df['DayOfWeek'] = df[time_col].dt.dayofweek
            df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
            new_feats = ['Month', 'DayOfWeek', 'Is_Weekend']
            for f in new_feats:
                if f not in controls: controls.append(f)
        except Exception as e:
            pass

    df = df.dropna()
    valid_controls = [c for c in controls if c in df.columns]
    
    # --- BRANCH 1: TIME LOGIC ENABLED (DiD) ---
    if 'Is_Post' in df.columns:
        # Standard Difference-in-Differences via OLS
        # Formula: Y ~ Treatment + Post + Treatment*Post + Controls
        
        # Create Interaction Term (The real causal effect)
        df['T_Interaction'] = df[treatment] * df['Is_Post']
        
        # Prepare X matrix for OLS
        # We need: Constant, Treatment (Group), Is_Post (Time), Interaction, Controls
        X_ols = df[[treatment, 'Is_Post', 'T_Interaction'] + valid_controls].copy()
        X_ols = sm.add_constant(X_ols)
        Y_ols = df[outcome]
        
        # Fit OLS
        model = sm.OLS(Y_ols, X_ols).fit()
        
        # Extract Results
        ate = model.params['T_Interaction']
        conf = model.conf_int().loc['T_Interaction']
        lower, upper = conf[0], conf[1]
        
        # Package into object that looks like the ML result for compatibility
        class DiDResult:
            def ate(self, X): return ate
            def ate_interval(self, X): return lower, upper
            def effect(self, X): return np.full(len(X), ate) # Constant effect for DiD
            
        # Feature Importance (use t-values from OLS)
        params = model.params.drop(['const', 'T_Interaction', treatment, 'Is_Post'], errors='ignore')
        importances = pd.DataFrame({'Feature': params.index, 'Importance': params.abs()}).sort_values('Importance', ascending=False)
        
        return DiDResult(), model, None, None, importances

    # --- BRANCH 2: NO TIME LOGIC (Cross-Sectional DML) ---
    else:
        if not valid_controls:
            X = np.zeros((len(df), 1))
            features = ["No_Controls"]
        else:
            X = df[valid_controls]
            features = valid_controls
        
        Y = df[outcome]
        T = df[treatment]

        # FIX: Updated n_estimators to 500 for stability
        # FIX: Added random_state=42 for reproducibility
        SEED = 42
        
        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=5, random_state=SEED),
            model_t=RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5, random_state=SEED),
            discrete_treatment=True,
            n_estimators=500,
            random_state=SEED
        )
        est.fit(Y, T, X=X)
        
        # Statsmodels for summary table only
        try:
            X_ols = pd.concat([T.rename("Treat"), pd.DataFrame(X, index=df.index)], axis=1)
            X_ols = sm.add_constant(X_ols)
            ols = sm.OLS(Y, X_ols).fit()
        except:
            ols = None
        
        effects = est.effect(X)
        interpreter = RandomForestRegressor(max_depth=4, random_state=SEED)
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
                    # Fix: Ensure date conversion for Date Input with dayfirst=True
                    if raw_df[time_col].dtype == 'object':
                         temp_dates = pd.to_datetime(raw_df[time_col], dayfirst=True)
                    else:
                         temp_dates = raw_df[time_col]
                         
                    min_d = temp_dates.min().date()
                    max_d = temp_dates.max().date()
                    default_d = min_d + (max_d - min_d) // 2
                    
                    # --- FIX: Set format to DD/MM/YYYY in sidebar ---
                    int_date = st.date_input("Intervention Date", value=default_d, min_value=min_d, max_value=max_d, format="DD/MM/YYYY")
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
                        
                        # --- 1970 FIX & DD-MM-YYYY FIX ---
                        if use_time and time_col:
                            try:
                                # Ensure parsing respects DD-MM-YYYY
                                prep_df[time_col] = pd.to_datetime(prep_df[time_col], dayfirst=True)
                            except:
                                pass 

                        # Handle DiD Logic
                        if use_time and time_col and int_date:
                            try:
                                # Ensure parsing respects DD-MM-YYYY
                                ids = pd.to_datetime(int_date, dayfirst=True)
                                prep_df['Is_Post'] = (prep_df[time_col] >= ids).astype(int)
                            except:
                                prep_df['Is_Post'] = 0
                        
                        need = [treat_col, out_col] + covs
                        if use_time and time_col: need.append(time_col) 
                        if 'Is_Post' in prep_df.columns: need.append('Is_Post')
                        
                        # Preprocess
                        clean, enc = preprocess_data(prep_df, need, cats)
                        
                        t_col_arg = time_col if use_time else None
                        
                        # RUN ANALYSIS
                        ml, stats, X_t, T_t, feats = run_analysis_logic(clean, treat_col, out_col, covs, time_col=t_col_arg)
                        
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
                # FIX: Unpack variables from results dict
                ml, stats = res['ml'], res['stats']
                feats = res['feats']
                
                # Check if it's DiD (Scalar) or DML (Forest)
                if hasattr(res['ml'], 'ate_interval'):
                    ate = res['ml'].ate(None) if 'Is_Post' in res['df'].columns else res['ml'].ate(res['X'])
                    l, u = res['ml'].ate_interval(None) if 'Is_Post' in res['df'].columns else res['ml'].ate_interval(res['X'])
                else:
                    ate, l, u = 0, 0, 0
                
                # Determine Significance directly from the Interval
                # This depends ONLY on 'l' and 'u', so it will always be defined
                if (l > 0) or (u < 0):
                    is_sig = True
                    sig_color = "#198754" # Green
                    sig_text = "Significant"
                    confidence_msg = "The 95% Confidence Interval excludes 0."
                else:
                    is_sig = False
                    sig_color = "#dc3545" # Red
                    sig_text = "Inconclusive"
                    confidence_msg = "The 95% Confidence Interval includes 0."

                # Get R2 safely
                r2_val = stats.rsquared if stats else 0.0
                
                fname = st.session_state['uploaded_file'].name
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                    <h3 style="margin: 0; padding: 0; font-family: 'Source Sans Pro', sans-serif; font-weight: 600; color: #31333F;">Analysis Results</h3>
                    <span style="color: #adb5bd; font-size: 1.2rem; font-weight: 400; padding-top: 2px;">: {fname}</span>
                </div>
                """, unsafe_allow_html=True)
                
                direction = "INCREASE" if ate > 0 else "DECREASE"
                
                st.markdown(f"""
                <div class="insight-box">
                    <b>💡 Automated Insight:</b><br>
                    The intervention led to an average <b>{direction}</b> of <b>{abs(ate):.2f}</b> in <b>{out_col}</b>. 
                    This result is <b>{sig_text}</b>. {confidence_msg}
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f'<div class="metric-container"><div class="metric-label">Average Impact</div><div class="metric-value">{ate:.2f}</div></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="metric-container"><div class="metric-label">95% Range</div><div class="metric-value">[{l:.2f}, {u:.2f}]</div></div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="metric-container"><div class="metric-label">Result</div><div class="metric-value" style="color:{sig_color}">{sig_text}</div></div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="metric-container"><div class="metric-label">Model Fit (R2)</div><div class="metric-value">{r2_val:.2f}</div></div>', unsafe_allow_html=True)
                
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                
                t0, t1, t2, t3, t4 = st.tabs(["📈 Treat vs Control", "📉 Impact Distribution", "🧠 Drivers of Impact", "🔍 Segment Analysis", "📊 Stats Table"])
                
                # --- NEW TAB: TREAT VS CONTROL VISUAL ---
                with t0:
                    st.caption("Visual check: How do the groups compare?")
                    plot_df = res['df'].copy()
                    
                    # Map labels for clearer plotting
                    plot_df['Group'] = plot_df[res['treat']].map({1: 'Treated', 0: 'Control'})
                    
                    # Scenario 1: TIME SERIES (Line Chart)
                    if res['graph_config']['use_time'] and res['graph_config']['time_col'] in raw_df.columns:
                         # Re-fetch the original time column from raw_df to avoid preprocessing issues
                         t_c = res['graph_config']['time_col']
                         try:
                             plot_df[t_c] = pd.to_datetime(raw_df[t_c], dayfirst=True)
                         except:
                             plot_df[t_c] = raw_df[t_c]
                         
                         # Group by Time and Group
                         trend = plot_df.groupby([t_c, 'Group'])[res['out']].mean().reset_index()
                         
                         fig = px.line(trend, x=t_c, y=res['out'], color='Group', 
                                      title="Average Outcome Trends (Parallel Trends Check)",
                                      color_discrete_map={'Treated': '#28a745', 'Control': '#6c757d'},
                                      markers=True)
                         st.plotly_chart(fig, use_container_width=True)
                         
                    # Scenario 2: NO TIME (Box Plot)
                    else:
                         fig = px.box(plot_df, x='Group', y=res['out'], color='Group',
                                     title="Outcome Distribution by Group",
                                     color_discrete_map={'Treated': '#28a745', 'Control': '#6c757d'})
                         st.plotly_chart(fig, use_container_width=True)

                with t1:
                    # Impact Dist
                    if 'Is_Post' in res['df'].columns:
                        impact_vals = pd.Series([ate] * len(res['df'])) # Constant impact
                        impact_dist = float(ate)
                    else:
                        impact_vals = ml.effect(res['X'])
                        impact_dist = impact_vals

                    fig = px.histogram(x=impact_vals, nbins=30, color_discrete_sequence=['#0d6efd'], labels={'x': 'Impact Value'})
                    fig.add_vline(x=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with t2:
                    if not feats.empty:
                        fig2 = px.bar(feats.head(10), x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#0d6efd'])
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No drivers available.")

                with t3:
                    if not feats.empty:
                        seg = st.selectbox("Segment By:", feats['Feature'].unique())
                        if seg in res['df'].columns:
                            fig3 = px.scatter(res['df'], x=seg, y=impact_vals, title=f"Impact vs {seg}")
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("No segments available.")

                with t4:
                    if stats: st.text(stats.summary())
                
                # PASS DF TO GENERATE PDF
                pdf_data = generate_pdf(ate, l, u, 0.05, r2_val, res['treat'], res['out'], res['feats'], pd.Series(impact_dist), res['graph_config'], fname, res['df'])
                st.download_button("DOWNLOAD PDF REPORT", pdf_data, "causal_report.pdf", "application/pdf", use_container_width=True)

    else:
        st.info("Configure logic and click Run Analysis in the sidebar.")