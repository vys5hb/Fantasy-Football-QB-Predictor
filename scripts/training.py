import os
from xgboost import XGBRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

DEFAULT_XGB_PARAMS = {
    'n_estimators': 400,
    'learning_rate': 0.03,
    'max_depth': 4,
    'reg_alpha': 1,
    'reg_lambda': 4,
    'gamma': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'random_state': 42,
}


def train_model_for_selection(X_train, y_train,
                              k_features=(12, 20),
                              cv=10,
                              n_jobs=-1,
                              verbose=False,
                              **xgb_overrides):
    xgb_params = {**DEFAULT_XGB_PARAMS, **xgb_overrides}

    model = XGBRegressor(
        **xgb_params,
        verbosity=1 if verbose else 0,
    )

    sfs = SFS(
        model,
        k_features=k_features,
        forward=True,
        floating=True,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    sfs = sfs.fit(X_train, y_train)

    return sfs


def train_model(X_train, y_train, X_val, y_val,
                early_stopping_rounds=30,
                model_path='models/xgboost.json',
                verbose=False,
                **xgb_overrides):
    xgb_params = {**DEFAULT_XGB_PARAMS, **xgb_overrides}

    model = XGBRegressor(
        **xgb_params,
        verbosity=1 if verbose else 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    return model
