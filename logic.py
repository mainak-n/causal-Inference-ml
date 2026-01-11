import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, selected_columns):
    """
    Cleans the data: drops missing values, converts text to numbers.
    """
    data = df[selected_columns].copy()
    data = data.dropna()
    
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
            
    return data, encoders

def train_causal_model(data, treatment_col, outcome_col, covariates):
    """
    Trains a Double Machine Learning (Causal Forest) model.
    """
    # 1. Define the estimator
    # We use Random Forest for both models to capture non-linearities
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=6),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=6),
        discrete_treatment=True,
        random_state=42
    )
    
    # 2. Prepare data matrices
    Y = data[outcome_col] # The Output
    T = data[treatment_col] # The Treatment (0 or 1)
    X = data[covariates] # The Controls
    
    # 3. Fit the model
    est.fit(Y, T, X=X)
    
    return est, X