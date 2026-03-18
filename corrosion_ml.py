# """
# Corrosion rate prediction — load data and run ML pipeline.
# Finds the Excel file in project root or data/ and uses robust paths.
# """

# import sys
# from pathlib import Path

# import pandas as pd

# # ---------------------------------------------------------------------------
# # Excel path: works from project root or from script location
# # ---------------------------------------------------------------------------
# def find_excel_path() -> Path:
#     """Locate the corrosion dataset Excel file in project or data/."""
#     script_dir = Path(__file__).resolve().parent
#     candidates = [
#         "Synthetic_Corrosion_Data_10000_Samples.xlsx",
#         "Synthetic_Corrosion_Data_10000_Samples (1).xlsx",
#     ]
#     for base in (script_dir, script_dir / "data"):
#         for name in candidates:
#             path = base / name
#             if path.is_file():
#                 return path
#     searched = [str(script_dir), str(script_dir / "data")]
#     raise FileNotFoundError(
#         f"Excel file not found. Looked for {candidates} in:\n  " + "\n  ".join(searched)
#     )


# def main() -> None:
#     excel_path = find_excel_path()
#     print(f"Loading: {excel_path.name}")

#     df = pd.read_excel(excel_path, engine="openpyxl")
#     print(f"Shape: {df.shape}")
#     print(f"Columns: {list(df.columns)}")

#     # TODO: add train/test split, feature/target, model (e.g. XGBoost), evaluation
#     return


# if __name__ == "__main__":
#     main()
#     sys.exit(0)

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


def _find_excel_path() -> Path:
    """Find the corrosion dataset in script folder or data/ subfolder."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        "Synthetic_Corrosion_Data_10000_Samples.xlsx",
        "Synthetic_Corrosion_Data_10000_Samples (1).xlsx",
    ]
    for base in (script_dir, script_dir / "data"):
        for name in candidates:
            path = base / name
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"Excel file not found. Tried {candidates} in:\n  {script_dir}\n  {script_dir / 'data'}"
    )


print("="*80)
print("CORROSION ML PIPELINE - START")
print("="*80)
 
# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")
 
try:
    excel_path = _find_excel_path()
    df = pd.read_excel(excel_path, engine="openpyxl")
    print(f"✓ Data loaded: {excel_path.name} — {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError as e:
    print(f" ERROR: {e}")
    exit()
 
# Display basic info
print(f"Columns: {df.columns.tolist()}")
print(f"\nData shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
 
# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("\n[2/5] Preparing data...")
 
# Select features (exclude text columns and target)
feature_cols = [
    'SiO2_Content_%', 'Epoxy_Type_Code', 'Curing_Agent_Ratio',
    'Coating_Thickness_um', 'Coating_Age_days', 'Surface_Prep_Quality_0_100',
    'Temperature_C', 'Humidity_percent', 'pH', 'Chloride_Concentration_ppm',
    'Dissolved_Oxygen_ppm', 'Sulfide_H2S_ppm', 'Applied_Potential_mV_vs_SCE',
    'Exposure_Time_hours', 'Substrate_Type_Code'
]
 
target_col = 'Corrosion_Rate_uA_cm2'
 
# Extract features and target
X = df[feature_cols]
y = df[target_col]
 
print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Target: {target_col}")
print(f"✓ Feature ranges:")
for col in feature_cols:
    print(f"   {col}: {X[col].min():.2f} - {X[col].max():.2f}")
 # ============================================================================
# STEP 3: SPLIT DATA
# ============================================================================
print("\n[3/5] Splitting data...")
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
 
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
 
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
print(f"✓ Features scaled")
# ============================================================================
# STEP 4: TRAIN MODELS
# ============================================================================
print("\n[4/5] Training 5 models...")
 
models = {}
results_list = []
 
# MODEL 1: RANDOM FOREST
print("\n  Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
rf_cv = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
 
models['Random Forest'] = {
    'model': rf_model,
    'predictions': rf_pred,
    'r2': rf_r2,
    'rmse': rf_rmse,
    'mape': rf_mape,
    'cv': rf_cv
}
results_list.append(['Random Forest', rf_r2, rf_rmse, rf_mape, rf_cv])
print(f"  ✓ R² = {rf_r2:.4f}, RMSE = {rf_rmse:.2f}, MAPE = {rf_mape:.4f}")
 
# MODEL 2: GRADIENT BOOSTING (XGBoost-like)
print("  Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mape = mean_absolute_percentage_error(y_test, gb_pred)
gb_cv = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
 
models['Gradient Boosting'] = {
    'model': gb_model,
    'predictions': gb_pred,
    'r2': gb_r2,
    'rmse': gb_rmse,
    'mape': gb_mape,
    'cv': gb_cv
}
results_list.append(['Gradient Boosting', gb_r2, gb_rmse, gb_mape, gb_cv])
print(f"  ✓ R² = {gb_r2:.4f}, RMSE = {gb_rmse:.2f}, MAPE = {gb_mape:.4f}")
 
# MODEL 3: SUPPORT VECTOR REGRESSION
print("  Training SVR...")
svr_model = SVR(kernel='rbf', C=100, gamma='scale')
svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model.predict(X_test_scaled)
svr_r2 = r2_score(y_test, svr_pred)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
svr_mape = mean_absolute_percentage_error(y_test, svr_pred)
svr_cv = cross_val_score(svr_model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
 
models['SVR'] = {
    'model': svr_model,
    'predictions': svr_pred,
    'r2': svr_r2,
    'rmse': svr_rmse,
    'mape': svr_mape,
    'cv': svr_cv
}
results_list.append(['SVR', svr_r2, svr_rmse, svr_mape, svr_cv])
print(f"  ✓ R² = {svr_r2:.4f}, RMSE = {svr_rmse:.2f}, MAPE = {svr_mape:.4f}")
 
# MODEL 4: LINEAR REGRESSION (BASELINE)
print("  Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mape = mean_absolute_percentage_error(y_test, lr_pred)
lr_cv = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
 
models['Linear Regression'] = {
    'model': lr_model,
    'predictions': lr_pred,
    'r2': lr_r2,
    'rmse': lr_rmse,
    'mape': lr_mape,
    'cv': lr_cv
}
results_list.append(['Linear Regression', lr_r2, lr_rmse, lr_mape, lr_cv])
print(f"  ✓ R² = {lr_r2:.4f}, RMSE = {lr_rmse:.2f}, MAPE = {lr_mape:.4f}")
 
# MODEL 5: ENSEMBLE (Average of best 2)
print("  Creating Ensemble (RF + GB average)...")
ensemble_pred = (rf_pred + gb_pred) / 2
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred)
 
models['Ensemble'] = {
    'predictions': ensemble_pred,
    'r2': ensemble_r2,
    'rmse': ensemble_rmse,
    'mape': ensemble_mape,
    'cv': (rf_cv + gb_cv) / 2
}
results_list.append(['Ensemble', ensemble_r2, ensemble_rmse, ensemble_mape, (rf_cv + gb_cv) / 2])
print(f"  ✓ R² = {ensemble_r2:.4f}, RMSE = {ensemble_rmse:.2f}, MAPE = {ensemble_mape:.4f}")
 
# ============================================================================
# STEP 5: IDENTIFY BEST MODEL
# ============================================================================
print("\n[5/5] Finalizing results...")
 
# Create results dataframe
results_df = pd.DataFrame(results_list, columns=['Model', 'R² Score', 'RMSE', 'MAPE', 'CV R²'])
best_model_idx = results_df['R² Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
 
print(f"\n✓ BEST MODEL: {best_model_name}")
print(f"  R² = {results_df.loc[best_model_idx, 'R² Score']:.4f}")
print(f"  RMSE = {results_df.loc[best_model_idx, 'RMSE']:.2f}")
print(f"  MAPE = {results_df.loc[best_model_idx, 'MAPE']:.4f}")
 
# ============================================================================
# FEATURE IMPORTANCE (from best model)
# ============================================================================
print("\nCalculating feature importance...")
 
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    best_model_obj = models[best_model_name]['model']
    feature_importance = best_model_obj.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']*100:.1f}%")
else:
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': np.abs(models[best_model_name]['model'].coef_) / np.sum(np.abs(models[best_model_name]['model'].coef_))
    }).sort_values('Importance', ascending=False)
 
# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\nGenerating 4 publication-quality charts...")
 
# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
 
# CHART 1: MODEL COMPARISON
print("  Creating Chart 1: Model Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))
models_names = results_df['Model'].tolist()
r2_scores = results_df['R² Score'].tolist()
colors = ['#2ecc71' if i == best_model_idx else '#3498db' for i in range(len(models_names))]
 
bars = ax.bar(models_names, r2_scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target (R²=0.85)')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Machine Learning Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
 
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
 
plt.tight_layout()
plt.savefig('01_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 01_model_comparison.png")
plt.close()
 
# CHART 2: ACTUAL VS PREDICTED
print("  Creating Chart 2: Actual vs Predicted...")
best_pred = models[best_model_name]['predictions']
 
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, best_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
 
# Perfect prediction line
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
 
ax.set_xlabel('Actual Corrosion Rate (μA/cm²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Corrosion Rate (μA/cm²)', fontsize=12, fontweight='bold')
ax.set_title(f'Best Model: {best_model_name}\nActual vs Predicted (R² = {results_df.loc[best_model_idx, "R² Score"]:.3f})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig('02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 02_actual_vs_predicted.png")
plt.close()
 
# CHART 3: FEATURE IMPORTANCE
print("  Creating Chart 3: Feature Importance...")
top_n = 10
top_features = importance_df.head(top_n)
 
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top_features)), top_features['Importance'].values, 
               color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values, fontsize=11)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title(f'Top {top_n} Features Influencing Corrosion Rate Prediction', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
 
# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.3f}',
            ha='left', va='center', fontsize=9, fontweight='bold')
 
plt.tight_layout()
plt.savefig('03_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 03_feature_importance.png")
plt.close()
 
# CHART 4: RESIDUALS PLOT
print("  Creating Chart 4: Residuals Plot...")
residuals = y_test - best_pred
 
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(best_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Corrosion Rate (μA/cm²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
ax.set_title('Residual Plot - Model Error Analysis', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig('04_residuals_plot.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_residuals_plot.png")
plt.close()
# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\nSaving results...")
 
# Results table
results_df.to_csv('ML_Results_Table.csv', index=False)
print(f"✓ Saved: ML_Results_Table.csv")
 
# Feature importance
importance_df.to_csv('Feature_Importance.csv', index=False)
print(f"✓ Saved: Feature_Importance.csv")
 
# ============================================================================
# PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
 
print(f"\nBest Model: {best_model_name}")
print(f"R² Score: {results_df.loc[best_model_idx, 'R² Score']:.4f} ({results_df.loc[best_model_idx, 'R² Score']*100:.1f}% accuracy)")
print(f"RMSE: {results_df.loc[best_model_idx, 'RMSE']:.2f} μA/cm²")
print(f"MAPE: {results_df.loc[best_model_idx, 'MAPE']:.4f} ({results_df.loc[best_model_idx, 'MAPE']*100:.1f}%)")
print(f"Cross-Validation R²: {results_df.loc[best_model_idx, 'CV R²']:.4f}")
 
print(f"\nAll Results:")
print(results_df.to_string(index=False))
 
print(f"\nTop 5 Most Important Features:")
for idx, (_, row) in enumerate(importance_df.head(5).iterrows()):
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']*100:.1f}%")
 
print(f"\nGenerated Files:")
print(f"  ✓ 01_model_comparison.png")
print(f"  ✓ 02_actual_vs_predicted.png")
print(f"  ✓ 03_feature_importance.png")
print(f"  ✓ 04_residuals_plot.png")
print(f"  ✓ ML_Results_Table.csv")
print(f"  ✓ Feature_Importance.csv")
 