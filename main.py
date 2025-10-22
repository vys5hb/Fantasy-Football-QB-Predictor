# Import libraries & functions from scripts folder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.feature_engineering import cleaning_data, season_count, rolling_features, next_season_pts
from scripts.training import train_model, train_model_for_selection
from scripts.feature_importance import plot_xgboost_feature_importance
from scripts.prediction_2025 import save_predictions

# Load messy data & initialize paths
df = pd.read_csv('data/qb_data.csv')
prediction_path = 'results/predictions_2025.csv'

# Clean data & engineer new features
df = cleaning_data(df)
df = season_count(df)
df = rolling_features(df)

# Separate 2024 data for future prediction use
df_2024 = df[df['Year'] == 2024].copy()
player_names_2024 = df_2024['Player'].reset_index(drop=True)
df_2024 = df_2024.drop(columns=['Player'])
# Shift next season fantasy points per game to line up with correct year
df = next_season_pts(df)

# All quantitative features in data
initial_features = ['Age', 'G', 'GS', 'Cmp', 'PassAtts', 'PassYds',
       'PassTDs', 'Ints', 'RushAtts', 'RushYds', 'RushTDs', 'Fmb', 'FL', '2PM',
       '2PP', 'FantPts', 'Year', 'CompProp', 'PassAtts/G', 'PassYds/G',
       'PassTD/G', 'Turnovers/G', 'RushTD/G', 'RushYds/G', 'FantPts/G',
       'ProBowl', 'AllPro', 'SeasonNumber', '#ofY', 'CompPropLast2Y',
       'CompPropLast3Y', 'PassAtts/GLast2Y', 'PassAtts/GLast3Y',
       'PassYds/GLast2Y', 'PassYds/GLast3Y', 'PassTD/GLast2Y',
       'PassTD/GLast3Y', 'Turnovers/GLast2Y', 'Turnovers/GLast3Y',
       'RushTD/GLast2Y', 'RushTD/GLast3Y', 'RushYds/GLast2Y',
       'RushYds/GLast3Y', 'FantPts/GLast2Y', 'FantPts/GLast3Y']
target = ['NextYearFantPt/G']

# Run forward selection testing using SFS to find the most effective combination of features for model
X_init = df[initial_features]
y = df[target].values.ravel()

# Train/Validation/Test split of 60%/20%/20%
X_train, X_temp, y_train, y_temp = train_test_split(X_init, y, test_size = .4, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = .5, random_state = 42)

# Train the feature selection model
model_for_selection = train_model_for_selection(X_train, y_train, verbose=False)

# Save results to a final list of features for model training
feature_names = list(model_for_selection.k_feature_names_)
print('Selected Features:', feature_names)

# Filter the data splits to only contain selected features
X_train_final = X_train[feature_names]
X_val_final = X_val[feature_names]
X_test_final = X_test[feature_names]

# Train the prediction model
model = train_model(X_train_final, y_train, X_val_final, y_val, verbose=False)

# Plot feature importance and save to results folder
importance_df = plot_xgboost_feature_importance(model, feature_names)

# Display Training, Validation, Testing regression statistics
y_train_pred = model.predict(X_train_final)
train_rmse = mean_squared_error(y_train,y_train_pred) ** .5
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Training RMSE = {train_rmse:.2f}, Train MAE = {train_mae:.2f}, Train R2 = {train_r2:.2f}')

y_val_pred = model.predict(X_val_final)
val_rmse = mean_squared_error(y_val, y_val_pred) ** .5
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)
print(f'Validation RMSE = {val_rmse:.2f}, Validation MAE = {val_mae:.2f}, Validation R2 = {val_r2:.2f}')

y_test_pred = model.predict(X_test_final)
test_rmse = mean_squared_error(y_test, y_test_pred) ** .5
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Testing RMSE = {test_rmse:.2f}, Test MAE = {test_mae:.2f}, Test R2 = {test_r2:.2f}')

# Use separated 2024 data to predict QB average fantasy points per game for upcoming 2025 season
X_2024 = df_2024[feature_names]
predictions_2025 = model.predict(X_2024)
predictions_df = pd.DataFrame(predictions_2025, columns=['PredictedFantPts/G'])
predictions_df['Player'] = player_names_2024
predictions_df['PredictedFantPts/G'] = round(predictions_df['PredictedFantPts/G'], 2)
predictions_df = predictions_df.sort_values(by='PredictedFantPts/G', ascending=False).reset_index(drop=True)

# Save 2025 predictions to csv in results folder
save_predictions(predictions_df, prediction_path)
print(f'Predictions for 2025 saved to {prediction_path}')