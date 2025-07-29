import pandas as pd

def save_predictions(predictions_df, file_path):
    # Initialize DataFrame for 2025 predictions
    df = pd.DataFrame({
        'Player': predictions_df['Player'],
        'PredictedFantPts/G': predictions_df['PredictedFantPts/G']
    })
    # Save to .csv in "results" folder
    df.to_csv(file_path, index=False)
