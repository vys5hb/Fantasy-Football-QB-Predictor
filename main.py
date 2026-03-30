import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.feature_engineering import cleaning_data, season_count, rolling_features, next_season_pts
from scripts.training import train_model, train_model_for_selection
from scripts.feature_importance import plot_xgboost_feature_importance
from scripts.prediction_2025 import save_predictions


def evaluate_model(model, X, y, label):
    """Print RMSE, MAE, and R2 for a given data split."""
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'{label} RMSE = {rmse:.2f}, {label} MAE = {mae:.2f}, {label} R2 = {r2:.2f}')


df = pd.read_csv('data/qb_data.csv')
prediction_path = 'results/predictions_2025.csv'

df = cleaning_data(df)
df = season_count(df)
df = rolling_features(df)

# Separate 2024 data for 2025 prediction before shifting target variable
df_2024 = df[df['Year'] == 2024].copy()
player_names_2024 = df_2024['Player'].reset_index(drop=True)
df_2024 = df_2024.drop(columns=['Player'])
df = next_season_pts(df)

initial_features = [
    'Age', 'G', 'GS', 'Cmp', 'PassAtts', 'PassYds',
    'PassTDs', 'Ints', 'RushAtts', 'RushYds', 'RushTDs', 'Fmb', 'FL', '2PM',
    '2PP', 'FantPts', 'Year', 'CompProp', 'PassAtts/G', 'PassYds/G',
    'PassTD/G', 'Turnovers/G', 'RushTD/G', 'RushYds/G', 'FantPts/G',
    'ProBowl', 'AllPro', 'SeasonNumber', '#ofY', 'CompPropLast2Y',
    'CompPropLast3Y', 'PassAtts/GLast2Y', 'PassAtts/GLast3Y',
    'PassYds/GLast2Y', 'PassYds/GLast3Y', 'PassTD/GLast2Y',
    'PassTD/GLast3Y', 'Turnovers/GLast2Y', 'Turnovers/GLast3Y',
    'RushTD/GLast2Y', 'RushTD/GLast3Y', 'RushYds/GLast2Y',
    'RushYds/GLast3Y', 'FantPts/GLast2Y', 'FantPts/GLast3Y',
]
target = 'NextYearFantPt/G'

X_init = df[initial_features]
y = df[target].values.ravel()

# Train/Validation/Test split: 60%/20%/20%
X_train, X_temp, y_train, y_temp = train_test_split(X_init, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Forward feature selection using SFS
model_for_selection = train_model_for_selection(X_train, y_train, verbose=False)
feature_names = list(model_for_selection.k_feature_names_)
print('Selected Features:', feature_names)

X_train_final = X_train[feature_names]
X_val_final = X_val[feature_names]
X_test_final = X_test[feature_names]

model = train_model(X_train_final, y_train, X_val_final, y_val, verbose=False)

importance_df = plot_xgboost_feature_importance(model, feature_names)

evaluate_model(model, X_train_final, y_train, 'Training')
evaluate_model(model, X_val_final, y_val, 'Validation')
evaluate_model(model, X_test_final, y_test, 'Testing')

# Predict 2025 fantasy points per game using 2024 season data
X_2024 = df_2024[feature_names]
predictions_2025 = model.predict(X_2024)
predictions_df = pd.DataFrame({
    'Player': player_names_2024,
    'PredictedFantPts/G': predictions_2025.round(2),
})
predictions_df = predictions_df.sort_values(by='PredictedFantPts/G', ascending=False).reset_index(drop=True)

save_predictions(predictions_df, prediction_path)
print(f'Predictions for 2025 saved to {prediction_path}')
