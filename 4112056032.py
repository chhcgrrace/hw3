# -*- coding: utf-8 -*-
"""
Title: Red Wine Quality Prediction using Multiple Linear Regression
Assignment: HW2 (Regression Analysis under CRISP-DM framework)
Author: 4112056032 (黃喻琦)
Date: May 2026
Description: This script performs multiple linear regression on the Red Wine Quality dataset.
             It includes EDA, data preprocessing, feature selection using backward elimination,
             model evaluation, and advanced visualization of confidence and prediction intervals.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import statsmodels.api as sm

# Set premium visualization styles
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial'] # Support Chinese characters
plt.rcParams['axes.unicode_minus'] = False # Support minus signs

def run_regression_pipeline():
    print("=" * 60)
    print("  CRISP-DM Red Wine Quality Multiple Linear Regression Analysis")
    print("=" * 60)
    
    # -------------------------------------------------------------
    # 1. DATA UNDERSTANDING (資料理解)
    # -------------------------------------------------------------
    print("\n[Step 1] Loading and understanding the dataset...")
    csv_path = "winequality-red.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find '{csv_path}' in the current directory.")
        
    df = pd.read_csv(csv_path)
    
    print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nDataset columns & data types:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe().round(2).to_string())
    
    # Missing values check
    missing_count = df.isnull().sum().sum()
    print(f"\nMissing values check: {missing_count} missing values found.")
    
    # Target distribution
    print("\nRed Wine Quality Distribution:")
    print(df['quality'].value_counts().sort_index().to_string())
    
    # Plot 1: Correlation Matrix Heatmap
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": .8})
    plt.title("Red Wine Quality Features - Correlation Matrix", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    heatmap_filename = "correlation_matrix.png"
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_filename}")
    
    # -------------------------------------------------------------
    # 2. DATA PREPARATION (資料準備)
    # -------------------------------------------------------------
    print("\n[Step 2] Preparing the data (Train-Test Split & Standardization)...")
    
    # Separate features and target
    X = df.drop(columns=['quality'])
    y = df['quality']
    
    # Split to train and test sets (80% train, 20% test)
    # Fixing random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Standardize the features (so coefficients are directly comparable)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    # -------------------------------------------------------------
    # 3. MODELING & FEATURE SELECTION (建模與特徵選擇)
    # -------------------------------------------------------------
    print("\n[Step 3] Modeling and Feature Selection (Backward Elimination based on p-value)...")
    
    # Helper function for backward elimination using statsmodels OLS
    def backward_elimination(X_data, y_data, alpha=0.05):
        features = list(X_data.columns)
        iteration = 1
        
        while len(features) > 0:
            # Statsmodels OLS requires an explicit constant column for intercept
            X_with_const = sm.add_constant(X_data[features])
            model = sm.OLS(y_data, X_with_const).fit()
            
            # Extract p-values (skipping intercept)
            p_values = model.pvalues.drop('const')
            max_p = p_values.max()
            max_feature = p_values.idxmax()
            
            if max_p > alpha:
                print(f"  Iteration {iteration}: Removing feature '{max_feature}' (p-value: {max_p:.4f} > {alpha})")
                features.remove(max_feature)
                iteration += 1
            else:
                print(f"  All remaining features have p-value <= {alpha}. Feature selection complete.")
                break
                
        return features, model
    
    # Perform feature selection on standardized training set
    selected_features, initial_model = backward_elimination(X_train_scaled, y_train)
    
    print("\nSelected Features:")
    for idx, col in enumerate(selected_features, 1):
        print(f"  {idx}. {col}")
        
    # Fit final OLS model on training set with selected features
    X_train_selected = X_train_scaled[selected_features]
    X_train_selected_const = sm.add_constant(X_train_selected)
    final_model = sm.OLS(y_train, X_train_selected_const).fit()
    
    print("\n" + "=" * 60)
    print("  FINAL OLS REGRESSION SUMMARY")
    print("=" * 60)
    print(final_model.summary())
    print("=" * 60)
    
    # -------------------------------------------------------------
    # 4. MODEL EVALUATION (模型評估)
    # -------------------------------------------------------------
    print("\n[Step 4] Evaluating the model on Training and Testing sets...")
    
    # Add constant to test set for prediction
    X_test_selected = X_test_scaled[selected_features]
    X_test_selected_const = sm.add_constant(X_test_selected)
    
    # Predictions
    y_train_pred = final_model.predict(X_train_selected_const)
    y_test_pred = final_model.predict(X_test_selected_const)
    
    # Train Metrics
    r2_train = final_model.rsquared
    adj_r2_train = final_model.rsquared_adj
    mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
    mse_train = metrics.mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    
    # Test Metrics
    r2_test = metrics.r2_score(y_test, y_test_pred)
    # Adjusted R2 for test set: 1 - [(1-R2)*(n-1)/(n-p-1)]
    n_test = len(y_test)
    p_features = len(selected_features)
    adj_r2_test = 1 - ((1 - r2_test) * (n_test - 1) / (n_test - p_features - 1))
    
    mae_test = metrics.mean_absolute_error(y_test, y_test_pred)
    mse_test = metrics.mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    
    # Print metrics table
    metrics_summary = pd.DataFrame({
        'Metric': ['R-squared (R2)', 'Adjusted R-squared', 'MAE', 'MSE', 'RMSE'],
        'Training Set': [r2_train, adj_r2_train, mae_train, mse_train, rmse_train],
        'Testing Set': [r2_test, adj_r2_test, mae_test, mse_test, rmse_test]
    }).round(4)
    
    print("\nPerformance Comparison:")
    print(metrics_summary.to_string(index=False))
    
    # -------------------------------------------------------------
    # 5. PREMIUM VISUALIZATION WITH CONFIDENCE & PREDICTION INTERVALS
    # -------------------------------------------------------------
    print("\n[Step 5] Generating premium visualization of prediction intervals...")
    
    # Get predictions and their standard errors/intervals
    # statsmodels get_prediction yields confidence intervals (for the mean) and prediction intervals (for single observations)
    predictions_obj = final_model.get_prediction(X_test_selected_const)
    predictions_summary = predictions_obj.summary_frame(alpha=0.05) # 95% confidence level
    
    # Columns in predictions_summary:
    # 'mean' (predicted value), 'mean_se' (std error of mean), 
    # 'mean_ci_lower', 'mean_ci_upper' (Confidence Interval of the mean)
    # 'obs_ci_lower', 'obs_ci_upper' (Prediction Interval for a new observation)
    
    # For a beautiful, ordered visualization, we sort the test instances by their predicted value
    plot_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions_summary['mean'].values,
        'CI_Lower': predictions_summary['mean_ci_lower'].values,
        'CI_Upper': predictions_summary['mean_ci_upper'].values,
        'PI_Lower': predictions_summary['obs_ci_lower'].values,
        'PI_Upper': predictions_summary['obs_ci_upper'].values
    }).sort_values(by='Predicted').reset_index(drop=True)
    
    # Plotting actual vs. predicted with shaded CI and PI
    plt.figure(figsize=(12, 7))
    
    # Plot Actual data points (add jitter on X axis to prevent complete overlap since quality is discrete)
    x_indices = np.arange(len(plot_df))
    plt.scatter(x_indices, plot_df['Actual'], color='#34495e', alpha=0.4, label='Actual Quality (with jitter)', s=20)
    
    # Plot the predicted line (ordered)
    plt.plot(x_indices, plot_df['Predicted'], color='#e74c3c', linewidth=2.5, label='Predicted Quality (MLR Fit)')
    
    # Shade 95% Confidence Interval (CI) - uncertainty of the fit
    plt.fill_between(x_indices, plot_df['CI_Lower'], plot_df['CI_Upper'], 
                     color='#f39c12', alpha=0.4, label='95% Confidence Interval (Mean)')
    
    # Shade 95% Prediction Interval (PI) - uncertainty of single observations
    plt.fill_between(x_indices, plot_df['PI_Lower'], plot_df['PI_Upper'], 
                     color='#3498db', alpha=0.15, label='95% Prediction Interval (Observation)')
    
    plt.xlabel('Test Samples (Sorted by Predicted Quality)', fontsize=12, labelpad=10)
    plt.ylabel('Wine Quality Score (3 - 8)', fontsize=12, labelpad=10)
    plt.title('Red Wine Quality Prediction with 95% Confidence & Prediction Intervals', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', shadow=True)
    plt.ylim(1.5, 9.5)
    plt.tight_layout()
    
    prediction_plot_filename = "prediction_intervals.png"
    plt.savefig(prediction_plot_filename, dpi=300)
    plt.close()
    print(f"Saved: {prediction_plot_filename}")
    
    # -------------------------------------------------------------
    # 6. RESIDUALS DIAGNOSTICS (殘差分析 - 驗證迴歸假設)
    # -------------------------------------------------------------
    print("\n[Step 6] Generating residual diagnostics...")
    residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Residual Histogram & KDE / Normal Fit
    sns.histplot(residuals, kde=True, ax=axes[0], color='#2c3e50', stat="density", bins=15)
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axes[0].plot(x, p, 'r--', linewidth=2, label=f'Normal Fit\n(μ={mu:.2f}, σ={std:.2f})')
    axes[0].set_xlabel('Residuals (Errors)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Residuals Distribution vs. Normal Curve', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    
    # Subplot 2: Q-Q Plot using Statsmodels
    sm.qqplot(residuals, line='45', fit=True, ax=axes[1])
    # Style the Q-Q plot dots and line to look highly premium
    axes[1].get_lines()[0].set_markerfacecolor('#3498db')
    axes[1].get_lines()[0].set_markeredgecolor('#2980b9')
    axes[1].get_lines()[0].set_alpha(0.6)
    axes[1].get_lines()[0].set_markersize(5)
    axes[1].get_lines()[1].set_color('#e74c3c')
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].set_title('Normal Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    
    plt.suptitle('Residual Diagnostics: Verifying OLS Normality Assumption', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    residuals_plot_filename = "residuals_analysis.png"
    plt.savefig(residuals_plot_filename, dpi=300)
    plt.close()
    print(f"Saved: {residuals_plot_filename}")
    
    # -------------------------------------------------------------
    # 7. MODEL DIAGNOSTICS: ACTUAL VS PREDICTED & RESIDUALS VS FITTED
    # -------------------------------------------------------------
    print("\n[Step 7] Generating actual vs predicted and residuals vs fitted diagnostics...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Actual vs. Predicted Scatter
    axes[0].scatter(y_test, y_test_pred, color='#2c3e50', alpha=0.5, edgecolors='none', s=40)
    lims = [
        min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
        max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    ]
    axes[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Prediction (Y = X)')
    axes[0].set_xlabel('Actual Quality', fontsize=11)
    axes[0].set_ylabel('Predicted Quality', fontsize=11)
    axes[0].set_title('Actual vs. Predicted Quality', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left')
    
    # Subplot 2: Residuals vs. Fitted (Classic Residual Plot for Homoscedasticity)
    axes[1].scatter(y_test_pred, residuals, color='#e74c3c', alpha=0.5, edgecolors='none', s=40)
    axes[1].axhline(y=0, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.75)
    axes[1].set_xlabel('Predicted Quality (Fitted Values)', fontsize=11)
    axes[1].set_ylabel('Residuals (Errors)', fontsize=11)
    axes[1].set_title('Residuals vs. Fitted Values', fontsize=12, fontweight='bold')
    
    plt.suptitle('Model Diagnostics: Prediction Accuracy & Residuals Homoscedasticity', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    diagnostics_plot_filename = "model_diagnostics.png"
    plt.savefig(diagnostics_plot_filename, dpi=300)
    plt.close()
    print(f"Saved: {diagnostics_plot_filename}")
    
    # -------------------------------------------------------------
    # 8. COEFFICIENT PLOT (特徵權重圖 - 呈現特徵重要性與信心區間)
    # -------------------------------------------------------------
    print("\n[Step 8] Generating standardized regression coefficients plot...")
    coef_df = pd.DataFrame({
        'Feature': final_model.params.index,
        'Coefficient': final_model.params.values,
        'CI_Lower': final_model.conf_int()[0].values,
        'CI_Upper': final_model.conf_int()[1].values
    })
    coef_df = coef_df[coef_df['Feature'] != 'const'].sort_values(by='Coefficient', ascending=True)
    
    plt.figure(figsize=(10, 6))
    error_left = coef_df['Coefficient'] - coef_df['CI_Lower']
    error_right = coef_df['CI_Upper'] - coef_df['Coefficient']
    errors = np.array(list(zip(error_left, error_right))).T
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in coef_df['Coefficient']]
    bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], xerr=errors, 
                    color=colors, alpha=0.8, edgecolor='none', height=0.5,
                    error_kw=dict(ecolor='#2c3e50', lw=1.5, capsize=4, capthick=1.5))
    
    plt.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=1, alpha=0.5)
    plt.xlabel('Standardized Coefficient Weight (Impact Size)', fontsize=12, labelpad=10)
    plt.ylabel('Selected 理化特徵 (Selected Features)', fontsize=12, labelpad=10)
    plt.title('Standardized Regression Coefficients (Feature Weights) with 95% Confidence Bars', fontsize=14, fontweight='bold', pad=15)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (0.01 if width >= 0 else -0.05), bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 va='center', ha='left' if width >= 0 else 'right', 
                 fontsize=10, fontweight='bold', color='#2c3e50')
                 
    plt.tight_layout()
    coefficient_plot_filename = "coefficient_plot.png"
    plt.savefig(coefficient_plot_filename, dpi=300)
    plt.close()
    print(f"Saved: {coefficient_plot_filename}")
    
    print("\n" + "=" * 60)
    print("  Analysis Pipeline Completed Successfully!")
    print("=" * 60)
    print("Generated files:")
    print(f"  1. {heatmap_filename} (Exploratory correlation heatmap)")
    print(f"  2. {prediction_plot_filename} (Actual vs Predicted with CI & PI)")
    print(f"  3. {residuals_plot_filename} (Residual diagnostics normality check)")
    print(f"  4. {diagnostics_plot_filename} (Actual vs Predicted and Residual plot)")
    print(f"  5. {coefficient_plot_filename} (Coefficient feature weights plot)")
    print("=" * 60)

if __name__ == "__main__":
    run_regression_pipeline()
