import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_xgboost_feature_importance(model, feature_names, results_path='results'):
    os.makedirs(results_path, exist_ok=True)
    # Get feature importances from XGBoost model
    importance = model.get_booster().get_score(importance_type='gain')
    # Create a DataFrame which maps each feature to its importance 
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': [importance.get(f,0) for f in feature_names]    
    }).sort_values(by='Importance', ascending=False)
    
    # Create a barplot displaying feature importance in descending order
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    # Save plot to "results" folder
    plt.savefig(os.path.join(results_path, 'feature_importance.png'))
    plt.close()
    
    return importance_df