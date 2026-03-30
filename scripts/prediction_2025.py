def save_predictions(predictions_df, file_path):
    predictions_df[['Player', 'PredictedFantPts/G']].to_csv(file_path, index=False)
