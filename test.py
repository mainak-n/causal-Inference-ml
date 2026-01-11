import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. SETUP
np.random.seed(999) 
n_samples = 6000    

# Define the Intervention
intervention_date = datetime(2023, 6, 1)
TARGET_IMPACT = 50.00  

# 2. GENERATE TIME SERIES
base_date = datetime(2023, 1, 1)
dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
dates.sort()

# 3. GENERATE CONTROLS
activity_score = np.random.randint(10, 90, size=n_samples)
regions = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)

# 4. ASSIGN TREATMENT (Biased)
prob_treat = 1 / (1 + np.exp(-( (activity_score - 50)/10 )))
treatment = np.random.binomial(1, prob_treat)

# 5. GENERATE OUTCOME (Revenue)
revenue = np.random.normal(100, 5, size=n_samples)
revenue += (activity_score * 0.5)

# Time Trend
is_post = np.array([1 if d >= intervention_date else 0 for d in dates])
revenue -= (is_post * 5.0) 

# --- ADD CAUSAL IMPACT ---
actual_impact_vector = treatment * is_post * TARGET_IMPACT
revenue += actual_impact_vector
revenue += np.random.normal(0, 2, size=n_samples)

# 6. SAVE
df = pd.DataFrame({
    'Date': dates,
    'Activity_Score': activity_score,
    'Region': regions,
    'Treatment_Group': treatment, 
    'Revenue': np.round(revenue, 2)
})

# --- FIX: Convert Date format to DD-MM-YYYY ---
df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')

filename = "positive_impact_data.csv"
df.to_csv(filename, index=False)

print(f"Data Generated: {filename}")
print(f"Target Positive Impact: +${TARGET_IMPACT}")
print(f"Intervention Date (DD-MM-YYYY): {intervention_date.strftime('%d-%m-%Y')}")