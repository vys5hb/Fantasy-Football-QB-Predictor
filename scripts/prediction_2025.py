import pandas as pd

def save_predictions(predictions_df, file_path):
    df = pd.DataFrame({
        'Player': predictions_df['Player'],
        'PredictedFantPts/G': predictions_df['PredictedFantPts/G']
    })
    
    df.to_csv(file_path, index=False)
