import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility (so you get the same numbers every time)
np.random.seed(42)

# --- CONFIGURATION ---
N_SAMPLES = 2000
START_DATE = datetime(2023, 1, 1)
INTERVENTION_DATE = datetime(2023, 6, 15)  # The date the "policy" changed

# --- 1. GENERATE TIME SERIES DATA ---
# Random dates distributed throughout the year
dates = [START_DATE + timedelta(days=np.random.randint(0, 365)) for _ in range(N_SAMPLES)]
dates.sort() # Sort chronologically

# --- 2. GENERATE COVARIATES (CONTROLS) ---

# Numerical Control: User Age (18-70)
age = np.random.randint(18, 70, size=N_SAMPLES)

# Numerical Control: User Tenure (months)
# We make tenure slightly correlated with age (older people tend to stay longer)
tenure = np.clip(np.random.normal(age * 0.5, 10), 1, 120).astype(int)

# Categorical Control: Region (4 categories)
regions = np.random.choice(['North', 'South', 'East', 'West'], size=N_SAMPLES, p=[0.25, 0.35, 0.2, 0.2])

# Categorical Control: Device Type (3 categories)
devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=N_SAMPLES, p=[0.6, 0.3, 0.1])

# --- 3. GENERATE TREATMENT ASSIGNMENT ---
# This simulates "Observational Data" where treatment isn't random.
# Example: Older users and Desktop users are MORE likely to get the 'promotion'.
prob_treatment = 1 / (1 + np.exp(-( (age - 40)/10 + (devices == 'Desktop').astype(int)*0.5 )))
treatment = np.random.binomial(1, prob_treatment)

# --- 4. GENERATE OUTCOME (Sales Amount) ---
# Base Sales (random noise + baseline)
base_sales = np.random.normal(50, 10, size=N_SAMPLES)

# Confounding Effect: Controls affecting the Outcome
# Older people spend more, Desktop users spend more, North region spends more.
control_effect = (age * 0.5) + (tenure * 0.2) + (np.where(regions=='North', 5, 0)) + (np.where(devices=='Desktop', 10, 0))

# TRUE CAUSAL EFFECT (This is the "Hidden Truth" we want the ML model to find)
# Let's say the promotion actually increases sales by exactly $15.00
true_causal_effect = 15.0

# TIME EFFECT (Seasonality)
# Sales naturally jump by $5 after June 15th regardless of promotion (Time Confounder)
is_post_intervention = np.array([1 if d >= INTERVENTION_DATE else 0 for d in dates])
time_effect = is_post_intervention * 5.0 

# FINAL OUTCOME EQUATION
# Sales = Base + Controls + Treatment_Effect + Time_Trend + Random_Noise
sales = base_sales + control_effect + (treatment * true_causal_effect) + time_effect + np.random.normal(0, 5, size=N_SAMPLES)

# --- 5. CREATE DATAFRAME ---
df = pd.DataFrame({
    'Date': dates,
    'Age': age,
    'Tenure_Months': tenure,
    'Region': regions,
    'Device': devices,
    'Got_Promotion': treatment, # Treatment (0/1)
    'Sales_Amount': np.round(sales, 2) # Outcome ($)
})

# --- 6. SAVE TO CSV ---
file_name = "synthetic_marketing_data.csv"
df.to_csv(file_name, index=False)

print(f"Data generated successfully: {file_name}")
print(f"True Causal Effect embedded: +${true_causal_effect}")
print("-" * 30)
print(df.head())