import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. GENERATE TIME SERIES DATA (For "Days 1, 2, 3...")
# ==========================================
def generate_time_series():
    # Setup: 100 Days of data
    days = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(days)]
    
    # Logic: Intervention happens on Day 50
    intervention_start = datetime(2024, 1, 1) + timedelta(days=50)
    
    data = []
    base_traffic = 1000
    
    for i, date in enumerate(dates):
        # 1. Natural Trend: Traffic grows slightly every day (+2 users/day)
        trend = i * 2
        
        # 2. Seasonality: Traffic is higher on weekends
        is_weekend = 1 if date.weekday() >= 5 else 0
        weekend_bump = is_weekend * 100
        
        # 3. INTERVENTION EFFECT:
        # After Day 50, traffic jumps by 300 users instantly (The "Level Shift")
        # AND grows faster (+5 users/day instead of +2) (The "Slope Change")
        if date >= intervention_start:
            treatment_effect = 300 + ((i - 50) * 3)
        else:
            treatment_effect = 0
            
        # 4. Noise: Random fluctuation
        noise = np.random.normal(0, 50)
        
        daily_traffic = base_traffic + trend + weekend_bump + treatment_effect + noise
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Daily_Visitors': int(daily_traffic),
            'Marketing_Spend': np.random.randint(500, 1000) # Control variable
        })
        
    df = pd.DataFrame(data)
    df.to_csv('time_series_test.csv', index=False)
    print("Generated 'time_series_test.csv' (Intervention Date: 2024-02-20)")

# ==========================================
# 2. GENERATE DOSE RESPONSE DATA (For "0 vs 1 vs 2 Emails")
# ==========================================
def generate_dose_response():
    # Setup: 2000 Customers
    n = 2000
    np.random.seed(42)
    
    # Covariates (Confounders)
    # Engagement Score: 1-10 (High engagement users get more emails naturally)
    engagement = np.random.randint(1, 11, n)
    
    # TREATMENT: Number of Emails Sent (0, 1, 2, 3, or 4)
    # Bias: More engaged users are MORE likely to get more emails
    # This creates a "Confounder" your model must fix.
    prob_weights = [0.4, 0.3, 0.2, 0.05, 0.05] # Base probs
    emails_sent = []
    
    for score in engagement:
        # If score is high, push probability towards higher email counts
        shift = score / 20.0 
        # (Simplified logic to correlate treatment with engagement)
        if score > 7:
            emails_sent.append(np.random.choice([2, 3, 4]))
        elif score > 4:
            emails_sent.append(np.random.choice([1, 2]))
        else:
            emails_sent.append(np.random.choice([0, 1]))
            
    emails_sent = np.array(emails_sent)
    
    # OUTCOME: Purchase Amount ($)
    # Logic:
    # - Baseline: Everyone spends $50 + (Engagement * 10)
    # - Email 1: Adds $30 (Huge lift)
    # - Email 2: Adds $15 (Diminishing returns)
    # - Email 3: Adds $0 (Plateau)
    # - Email 4: Subtracts $10 (Annoyance!)
    
    base_spend = 50 + (engagement * 10)
    
    # Calculate causal impact per user
    impact = np.zeros(n)
    impact = np.where(emails_sent == 1, 30, impact)
    impact = np.where(emails_sent == 2, 45, impact) # 30 + 15
    impact = np.where(emails_sent == 3, 45, impact) # Plateau
    impact = np.where(emails_sent == 4, 35, impact) # Drop due to spam
    
    noise = np.random.normal(0, 15, n)
    total_spend = base_spend + impact + noise
    
    df = pd.DataFrame({
        'customer_id': range(1, n+1),
        'engagement_score': engagement,
        'emails_received': emails_sent, # TREATMENT (0-4)
        'purchase_amount': total_spend  # OUTCOME
    })
    
    df.to_csv('dose_response_test.csv', index=False)
    print("Generated 'dose_response_test.csv' (Look for the curve flattening after 2 emails)")

if __name__ == "__main__":
    generate_time_series()
    generate_dose_response()