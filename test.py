import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. SETUP
np.random.seed(999) # Changed seed to ensure fresh distribution
n_samples = 6000    # More samples for better stability

# Define the Intervention
intervention_date = datetime(2023, 6, 1)
TARGET_IMPACT = 50.00  # HUGE positive impact to be absolutely sure

# 2. GENERATE TIME SERIES
base_date = datetime(2023, 1, 1)
dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
dates.sort()

# 3. GENERATE CONTROLS
# User Activity Score (0-100)
activity_score = np.random.randint(10, 90, size=n_samples)
# User Region
regions = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)

# 4. ASSIGN TREATMENT (Biased)
# High activity users are more likely to get the treatment
prob_treat = 1 / (1 + np.exp(-( (activity_score - 50)/10 )))
treatment = np.random.binomial(1, prob_treat)

# 5. GENERATE OUTCOME (Revenue)
# Base revenue
revenue = np.random.normal(100, 5, size=n_samples)

# Add Control Effect (Active users spend more naturally)
revenue += (activity_score * 0.5)

# Add Time Trend (Revenue drops slightly in second half of year - Negative Trend)
# This tests if the model can ignore the general drop and still find the positive impact.
is_post = np.array([1 if d >= intervention_date else 0 for d in dates])
revenue -= (is_post * 5.0) 

# --- ADD THE MASSIVE POSITIVE CAUSAL IMPACT ---
# Only applied if Treated AND Post-Intervention
actual_impact_vector = treatment * is_post * TARGET_IMPACT
revenue += actual_impact_vector

# Add very little random noise to make the signal clear
revenue += np.random.normal(0, 2, size=n_samples)

# 6. SAVE
df = pd.DataFrame({
    'Date': dates,
    'Activity_Score': activity_score,
    'Region': regions,
    'Treatment_Group': treatment,  # 0 or 1
    'Revenue': np.round(revenue, 2)
})

filename = "positive_impact_data.csv"
df.to_csv(filename, index=False)

print(f"Data Generated: {filename}")
print(f"Target Positive Impact: +${TARGET_IMPACT}")
print(f"Intervention Date: {intervention_date.strftime('%Y-%m-%d')}")