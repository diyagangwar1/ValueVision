import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

df = pd.read_csv("raw_data/cleaned_encoded_property_data.csv")
columns_to_drop = ['ID', 'address', 'suburb', 'date', 'agency', 'lat', 'long']
X = df.drop(columns=columns_to_drop + ['price'], errors='ignore')
y = np.log1p(df['price'])  # Log-transform target

# feature engineering (MHPP-inspired)
if 'BedroomAbvGr' in X and 'FullBath' in X:
    X['bed_bath_ratio'] = X['BedroomAbvGr'] / (X['FullBath'] + 1e-6)
    
if 'YrSold' in X and 'YearBuilt' in X:
    X['age_at_sale'] = X['YrSold'] - X['YearBuilt']

X = X.select_dtypes(include=[np.number])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# feature selection (MHPP-style optimization)
selector = RFECV(RandomForestRegressor(n_estimators=100), step=1, cv=3, scoring='neg_mean_absolute_error')
selector.fit(X_train_scaled, y_train)
X_train_reduced = selector.transform(X_train_scaled)
X_test_reduced = selector.transform(X_test_scaled)

# model configuration
models = {
    "XGBoost": XGBRegressor(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.4,
        gamma=0.1
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=50,
        min_samples_leaf=10,
        max_features=0.3,
        bootstrap=True
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8
    )
}

# training and evalution
results = []
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_reduced, y_train)
    
    # Save trained model
    joblib.dump(model, f"{name}_model.pkl")
    
    # Predict and transform back from log scale
    y_pred = np.expm1(model.predict(X_test_reduced))
    y_true = np.expm1(y_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "Features Used": X_train_reduced.shape[1]
    })

    # Diagnostic Plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f"{name} Predictions\nMAE: ${mae:,.0f}, RMSE: ${rmse:,.0f}")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residual Distribution")
    
    plt.tight_layout()
    plt.savefig(f"{name}_diagnostics.png")
    plt.close()

# SHAP Analysis (MHPP-style interpretability)
print("\n=== SHAP Analysis ===")
explainer = shap.TreeExplainer(models["XGBoost"])
shap_values = explainer.shap_values(X_test_reduced)

shap.summary_plot(shap_values, X_test_reduced, feature_names=X.columns[selector.support_], show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.close()
results_df = pd.DataFrame(results)
print("\n=== Final Results ===")
print(results_df.to_string(index=False))
results_df.to_csv("structured_only_results.csv", index=False)
