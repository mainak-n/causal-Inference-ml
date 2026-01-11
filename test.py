import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. SETUP
np.random.seed(42) # Ensures you get the same numbers every time
n_samples = 5000   # Enough data points to ensure statistical significance

# Define the Intervention (The "Policy Change")
intervention_date = datetime(2023, 7, 1)
true_causal_impact = 25.00  # We want the tool to find this number!

# 2. GENERATE TIME SERIES DATA
# Dates ranging from Jan 1, 2023 to Dec 31, 2023
base_date = datetime(2023, 1, 1)
dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
dates.sort()

# 3. GENERATE CONTROLS (Confounders)
# Numerical
age = np.random.randint(18, 65, size=n_samples)
tenure = np.random.randint(1, 120, size=n_samples)

# Categorical
regions = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
devices = np.random.choice(['Mobile', 'Desktop'], size=n_samples, p=[0.6, 0.4])

# 4. ASSIGN TREATMENT (Biased Assignment)
# "Rich" users (Older, Desktop users) are MORE likely to get the treatment.
# This tests if your model can remove this bias.
prob_treat = 1 / (1 + np.exp(-( (age - 40)/10 + (devices == 'Desktop').astype(int) )))
treatment = np.random.binomial(1, prob_treat)

# 5. GENERATE OUTCOME ($ Sales)
# Start with a base value
sales = np.random.normal(100, 10, size=n_samples)

# Add Control Effects (Rich people spend more naturally)
sales += (age * 0.5) 
sales += (np.where(devices=='Desktop', 15, 0)) 

# Add Time Trend (Inflation/Seasonality) - Applies to EVERYONE
# Sales go up by $10 naturally after July 1st
is_post = np.array([1 if d >= intervention_date else 0 for d in dates])
sales += (is_post * 10)

# Add the TRUE CAUSAL IMPACT (The Signal)
# Only applied if you are Treated AND it is Post-Intervention
actual_impact_vector = treatment * is_post * true_causal_impact
sales += actual_impact_vector

# Add some random noise
sales += np.random.normal(0, 5, size=n_samples)

# 6. SAVE
df = pd.DataFrame({
    'Date': dates,
    'User_Age': age,
    'User_Tenure': tenure,
    'Region': regions,
    'Device_Type': devices,
    'Got_Promotion': treatment,  # 0 or 1
    'Sales_Amount': np.round(sales, 2)
})

filename = "test_data_significant.csv"
df.to_csv(filename, index=False)

print(f"Data Generated: {filename}")
print(f"Target Impact to find: ${true_causal_impact}")
print(f"Intervention Date: {intervention_date.strftime('%Y-%m-%d')}")