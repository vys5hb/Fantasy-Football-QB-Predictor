import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Train model with validation to prevent overfitting, tree depth = 6
def train_model(X_train, y_train, X_val, y_val,
                n_estimators = 500,
                learning_rate = .04,
                max_depth = 6,
                early_stopping_rounds = 20,
                model_path = 'models/xgboost.json',
                verbose = False):
    
    model = XGBRegressor(
        n_estimators = n_estimators,
        learning_rate = learning_rate,
        max_depth = max_depth,
        early_stopping_rounds = early_stopping_rounds,
        random_state = 42,
        verbosity = 1 if verbose else 0)
    
    # Fit the XGBoost model on training data with validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose = verbose)
    
    # Save model to XGBoost model to "models" folder
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    return model