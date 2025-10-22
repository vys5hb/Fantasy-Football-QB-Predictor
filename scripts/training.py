import os
from xgboost import XGBRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error

def train_model_for_selection(X_train, y_train, 
                                n_estimators = 400, 
                                learning_rate = .03, 
                                max_depth = 4, 
                                random_state = 42,
                                reg_alpha = 1,
                                reg_lambda = 4,
                                gamma = .3,
                                subsample = .8,
                                colsample_bytree = .8,
                                min_child_weight = 5,
                                k_features = (12,20),
                                cv = 10,
                                n_jobs = -1,
                                verbose = False):
    
    model_for_selection = XGBRegressor(
        n_estimators = n_estimators, 
        learning_rate = learning_rate,
        max_depth = max_depth,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        gamma = gamma,
        subsample = subsample,                           
        colsample_bytree = colsample_bytree,
        min_child_weight = min_child_weight,
        random_state = random_state,
        verbosity = 1 if verbose else 0)
    
    sfs = SFS(model_for_selection,
              k_features = k_features,
              forward = True,
              floating = True,
              scoring = 'neg_root_mean_squared_error',
              cv = cv,
              n_jobs = n_jobs,
              verbose = verbose)
    
    sfs = sfs.fit(X_train, y_train)
    
    return sfs

def train_model(X_train, y_train, X_val, y_val,
                n_estimators = 400,
                learning_rate = .03,
                max_depth = 4,
                reg_alpha = 1,
                reg_lambda = 4,
                gamma = .3,
                subsample = .8,
                colsample_bytree = .8,
                min_child_weight = 5,
                early_stopping_rounds = 30,
                random_state = 42,
                model_path = 'models/xgboost.json',
                verbose = False):
    
    model = XGBRegressor(
        n_estimators = n_estimators,
        learning_rate = learning_rate,
        max_depth = max_depth,
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        gamma = gamma,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        min_child_weight = min_child_weight,
        random_state = random_state,
        verbosity = 1 if verbose else 0)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds = early_stopping_rounds,
        verbose = verbose)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    return model