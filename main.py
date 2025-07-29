# Import libraries & functions from scripts folder
import pandas as pd
from xgboost import XGBRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.feature_engineering import cleaning_data, season_count, rolling_features, next_season_pts
from scripts.training import train_model
from scripts.feature_importance import plot_xgboost_feature_importance
from scripts.prediction_2025 import save_predictions

# Load messy data & initialize paths
df = pd.read_csv('data/qb_data.csv')
model_path = 'models/xgboost.json'
prediction_path = 'results/predictions_2025.csv'
plot_path = 'results/distribution_curve_2025.png'

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

# Train/Validation/Test split of 50%/35%/15% for feature selection
X_train, X_temp, y_train, y_temp = train_test_split(X_init, y, test_size = .5, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = .3, random_state = 42)

model_for_selection = XGBRegressor(n_estimators = 200, learning_rate = .075, max_depth = 3, random_state = 42)
sfs = SFS(model_for_selection,
          k_features = 15,
          forward = True,
          floating = True,
          scoring = 'neg_root_mean_squared_error',
          cv = 4,
          n_jobs = -1,
          verbose = False)

sfs = sfs.fit(X_train, y_train)

# Save results to a final list of features for model training
feature_names = list(sfs.k_feature_names_)
print('Selected Features:', feature_names)

# Train the model
X_final = df[feature_names]

# Train/Validation/Test split of 50%/35%/15% for final model
X_train, X_temp, y_train, y_temp = train_test_split(X_final, y, test_size = .5, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = .3, random_state = 42)

model = train_model(X_train, y_train, X_val, y_val, verbose=False)

# Plot feature importance and save to results folder
importance_df = plot_xgboost_feature_importance(model, feature_names)

# Display Testing & Training regression statistics
y_test_pred = model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred) ** .5
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test RMSE = {test_rmse:.2f}, Test MAE = {test_mae:.2f}, Test R2 = {test_r2:.2f}')

y_train_pred = model.predict(X_train)
train_rmse = mean_squared_error(y_train,y_train_pred) ** .5
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Train RMSE = {train_rmse:.2f}, Train MAE = {train_mae:.2f}, Train R2 = {train_r2:.2f}')

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