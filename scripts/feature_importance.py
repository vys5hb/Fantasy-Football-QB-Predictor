import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_xgboost_feature_importance(model, feature_names, results_path='results'):
    os.makedirs(results_path, exist_ok=True)

    importance = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': [importance.get(f, 0) for f in feature_names],
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'feature_importance.png'))
    plt.close()

    return importance_df
