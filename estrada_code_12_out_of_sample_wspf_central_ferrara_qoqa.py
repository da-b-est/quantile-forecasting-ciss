#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:02:53 2025

@author: dianniadreiestrada
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import t
import warnings
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor



# Set the working directory (useful to choose the folder where to export output files)
os.chdir('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise')

# Load the data
spf_df = pd.read_csv('Data/spf_df.csv')

# Ensure numeric columns are properly recognized
numeric_columns = spf_df.columns.difference(['Date', 'FCT_SOURCE'])
spf_df[numeric_columns] = spf_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group data by Date and FCT_SOURCE to visualize distribution
def plot_distributions(df, date_col, source_col, point_col):
    plt.figure(figsize=(14, 7))
    sns.boxplot(x=date_col, y=point_col, hue=source_col, data=df, showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Distribution of Point Forecasts by Date and Source")
    plt.ylabel("Point Forecasts")
    plt.xlabel("Date")
    plt.legend(title="Source", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Plot the distribution of Point Forecasts
plot_distributions(spf_df, "Date", "FCT_SOURCE", "POINT")

# Calculate Average, Median, and Percentiles (5th, 25th, 75th, and 90th) for each quarter
def calculate_statistics(df, date_col):
    # Group by date
    stats = df.groupby(date_col).agg(
        average_point=('POINT', 'mean'),
        median_point=('POINT', 'median'),
        iqr_point=('POINT', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        p5_point=('POINT', lambda x: x.quantile(0.05)),
        p25_point=('POINT', lambda x: x.quantile(0.25)),
        p75_point=('POINT', lambda x: x.quantile(0.75)),
        p90_point=('POINT', lambda x: x.quantile(0.90)),
        **{f'avg_{col}': (col, 'mean') for col in numeric_columns},
        **{f'median_{col}': (col, 'median') for col in numeric_columns},
        **{f'iqr_{col}': (col, lambda x: x.quantile(0.75) - x.quantile(0.25)) for col in numeric_columns}
    ).reset_index()
    return stats

# Get the statistics for each quarter
statistics_df = calculate_statistics(spf_df, "Date")

# Display statistics
print(statistics_df)

statistics_df.to_csv('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise/Data/statistics_df.csv', index=False)


#%% Functions
warnings.filterwarnings('ignore')  # Suppress all warnings

# Set a limit for minimization iterations
minimize_iter_max = 100

# Function to calculate the cumulative distribution function of the skewed t-distribution
def pskt(x, mu, sigma, alpha, nu):
    z = (x - mu) / sigma
    cdf_value = t.cdf(z * (1 + alpha * np.sign(z)), df=nu)
    return cdf_value

# Function to compute the probability density function of the skewed t-distribution
def dskt(x, mu, sigma, alpha, nu):
    z = (x - mu) / sigma
    adjusted_z = z * (1 + alpha * np.sign(z))
    pdf_value = t.pdf(adjusted_z, df=nu) / sigma
    return pdf_value

# Logarithmic Score function for the skewed t-distribution
def log_score(y, mu, sigma, alpha, nu):
    pdf_value = dskt(y, mu, sigma, alpha, nu)
    if pdf_value <= 0:
        return np.inf
    log_score_value = -np.log(max(pdf_value, 1e-10))
    return log_score_value

# CRPS function for skewed-t distribution (numerical approximation)
def crps_skewed_t(y, mu, sigma, alpha, nu):
    # Limit the integration to a reasonable range around the observation
    lower_bound = df[ycol].min()
    upper_bound = df[ycol].max()

    def integrand(x):
        F_forecast = pskt(x, mu, sigma, alpha, nu)
        F_obs = 1.0 if y < x else 0.0
        return (F_forecast - F_obs) ** 2

    crps_value, _ = quad(integrand, lower_bound, upper_bound, epsabs=1e-6, epsrel=1e-6)
    return crps_value


# Function to compute the quantile for the skewed t-distribution
def qskt(p, mu, sigma, alpha, nu):
    quantile = mu + sigma * t.ppf(p, df=nu) * (1 + alpha * np.sign(p - 0.5))
    return quantile

# Function to perform quantile interpolation for skewed t-distribution
def quantiles_interpolation(quantiles, quantile_targets):
    def objective(params):
        mu, sigma, alpha, nu = params
        theoretical_quantiles = [qskt(q, mu, sigma, alpha, nu) for q in quantile_targets]
        error = np.sum((np.array(theoretical_quantiles) - np.array(quantiles)) ** 2)
        return error

    initial_guess = [
        np.mean(quantiles),  
        max(np.std(quantiles), 1e-2), 
        0.1,  
        4  
    ]
    
    bounds = [
        (-np.inf, np.inf),  
        (1e-6, np.inf),  
        (-10, 10),  
        (2, 30)  
    ]
    
    result = minimize(objective, initial_guess, bounds=bounds, options={'maxiter': minimize_iter_max})
    mu, sigma, alpha, nu = result.x
    return mu, sigma, alpha, nu

# Function to check multicollinearity using VIF
def check_multicollinearity(df, predictors):
    X = sm.add_constant(df[predictors])
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nMulticollinearity Check (VIF):")
    print(vif_data)
 
    

#%% Load the data
df = pd.read_excel('data/Data_Adrian_2019.xlsx', sheet_name='EA_Quarterly')

# Ensure the 'Date' column in 'df' is formatted correctly to quarterly dates in the desired format
def adjust_dates(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.to_period('Q').dt.to_timestamp()
    return df

# Convert all 'Date' columns to datetime format for consistency
df = adjust_dates(df)
statistics_df = adjust_dates(statistics_df)

# Add percentiles (5th, 25th, 75th, 90th)
percentile_columns = ['average_point', 'median_point', 'p5_point', 'p25_point', 'p75_point', 'p90_point']
selected_quantiles = statistics_df[['Date'] + percentile_columns]

# Merge SPF data with the empirical dataset (outer join)
merged_df = pd.merge(df, selected_quantiles, on='Date', how='outer')

# Rename columns for clarity
merged_df.rename(columns={
    'average_point': 'SPF_mean_forecast',
    'median_point': 'SPF_median_forecast',
    'p5_point': 'SPF_5th_percentile',
    'p25_point': 'SPF_25th_percentile',
    'p75_point': 'SPF_75th_percentile',
    'p90_point': 'SPF_90th_percentile'
}, inplace=True)

# Fill missing values with NaN to indicate non-intersecting dates
merged_df.fillna(value=pd.NA, inplace=True)

# Display or export the resulting dataframe
print(merged_df)

# Set index for the merged dataframe
merged_df.set_index('Date', inplace=True)

#%% Define forecasting horizon
h = 1
ycol = 'g_RGDP' if h == 1 else 'g_RGDPyoy'

# Create original and shifted versions of the GDP column
merged_df['g_RGDP_original'] = merged_df[ycol]  # Store the original values
merged_df[ycol] = merged_df[ycol].shift(-h)  # Shift the target variable by `-h` steps (lead)

# Drop rows where the key columns have NaN values due to shifting
merged_df.dropna(subset=['CISS', ycol, 'g_RGDP_original'], inplace=True)

# Identify the index of the first out-of-sample observation
try:
    jtFirstOOS = merged_df.index.get_loc(pd.Timestamp('1999-07-01'))
except KeyError:
    print("The specified date '1999-07-01' is not in the index. Please check the data range.")
    jtFirstOOS = 0  # Default to the start if the date is not found

# Ensure necessary columns are available and drop missing rows for SPF-related columns
required_columns = [
    'g_RGDP_original', 'CISS',
    'SPF_mean_forecast', 'SPF_median_forecast', 'SPF_5th_percentile',
    'SPF_25th_percentile', 'SPF_75th_percentile', 'SPF_90th_percentile'
]

# Drop rows with missing values in the required columns
rows_with_missing = merged_df[required_columns].isnull().any(axis=1).sum()  # Count rows with missing values
if rows_with_missing > 0:
    print(f"Dropping {rows_with_missing} rows due to missing values in required columns...")
    merged_df.dropna(subset=required_columns, inplace=True)
else:
    print("No missing values in required columns. No rows dropped.")

# Ensure there are still enough data points after dropping
if merged_df.shape[0] <= jtFirstOOS:
    raise ValueError("Insufficient data points after dropping missing values. Please check your dataset.")
else:
    print(f"Dataset after dropping missing values: {merged_df.shape[0]} rows remaining.")


rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
quantile_targets = [0.05, 0.25, 0.75, 0.95]
T_oos = len(merged_df) - jtFirstOOS
y_oos = merged_df[ycol].iloc[jtFirstOOS:].values

models = {
    'Model 1: GDP': ['g_RGDP_original'],
    'Model 2: CISS': ['CISS'],
    'Model 3: GDP + CISS': ['g_RGDP_original', 'CISS'],
    'Model 4: SPF (Mean)': ['SPF_mean_forecast'],
    'Model 5: GDP + SPF (Mean)': ['g_RGDP_original', 'SPF_mean_forecast'],
    'Model 6: CISS + SPF (Mean)': ['CISS', 'SPF_mean_forecast'],
    'Model 7: GDP + CISS + SPF (Mean)': ['g_RGDP_original', 'CISS', 'SPF_mean_forecast'],
    'Model 8: SPF (Median)': ['SPF_median_forecast'],
    'Model 9: GDP + SPF (Median)': ['g_RGDP_original', 'SPF_median_forecast'],
    'Model 10: CISS + SPF (Median)': ['CISS', 'SPF_median_forecast'],
    'Model 11: GDP + CISS + SPF (Median)': ['g_RGDP_original', 'CISS', 'SPF_median_forecast'],
    
}


results = pd.DataFrame(columns=['Model', 'Average LS', 'Average CRPS'])
# Initialize a dictionary to store CRPS scores for each model
crps_scores_dict = {model_name: np.zeros(T_oos) for model_name in models.keys()}
# Initialize dictionary for storing fitted quantiles for each model
quantile_fitted_dict = {model_name: np.zeros((T_oos, len(quantile_levels))) for model_name in models.keys()}


# Rolling quantile regression for out-of-sample prediction
for model_name, predictors in models.items():
    print(f"\nEvaluating {model_name}...")

    # Check multicollinearity
    check_multicollinearity(merged_df, predictors)

    ls_scores = np.zeros(T_oos)
    crps_scores = np.zeros(T_oos)
    quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))

    for ti in range(jtFirstOOS, len(merged_df)):
        train_df = merged_df.iloc[ti - rolling_window_size:ti]

        # Check for missing predictors in training data
        missing_predictors = [col for col in predictors if col not in train_df.columns]
        if missing_predictors:
            print(f"Warning: Missing predictors in train_df: {missing_predictors}")
            continue  # Skip if necessary columns are missing

        # Drop rows with NaN in predictors or target values
        X_train = sm.add_constant(train_df[predictors], has_constant='add')
        y_train = train_df[ycol]
        valid_rows = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        X_train = X_train[valid_rows]
        y_train = y_train[valid_rows]

        if len(y_train) == 0:  # Skip if no valid data points
            print(f"Skipping index {ti}: No valid data points in rolling window.")
            continue

        for i, q in enumerate(quantile_levels):
            quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
            quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict(
                [1] + list(merged_df[predictors].iloc[ti])
            )[0]
            quantile_fitted_dict[model_name][ti - jtFirstOOS, i] = quantile_model.predict(
                [1] + list(merged_df[predictors].iloc[ti])
            )[0]

    tdist_mu_oos = np.zeros(T_oos)
    tdist_sigma_oos = np.zeros(T_oos)
    tdist_alpha_oos = np.zeros(T_oos)
    tdist_nu_oos = np.zeros(T_oos)

    # Quantile interpolation and parameter estimation
    for ti in range(T_oos):
        quantiles_to_fit = quantile_fitted_oos[ti, [0, 1, 3, 4]]
        mu, sigma, alpha, nu = quantiles_interpolation(quantiles_to_fit, quantile_targets)
        tdist_mu_oos[ti], tdist_sigma_oos[ti], tdist_alpha_oos[ti], tdist_nu_oos[ti] = mu, sigma, alpha, nu

    # Calculate Log Score and CRPS for out-of-sample predictions
    for i in range(T_oos):
        mu, sigma, alpha, nu = tdist_mu_oos[i], tdist_sigma_oos[i], tdist_alpha_oos[i], tdist_nu_oos[i]
        ls_scores[i] = log_score(y_oos[i], mu, sigma, alpha, nu)
        crps_scores[i] = crps_skewed_t(y_oos[i], mu, sigma, alpha, nu)

    # Store CRPS scores for the current model
    crps_scores_dict[model_name] = crps_scores

    average_ls = np.mean(ls_scores)
    average_crps = np.mean(crps_scores)

    new_row = pd.DataFrame({'Model': [model_name], 'Average LS': [average_ls], 'Average CRPS': [average_crps]})
    print(new_row)
    results = pd.concat([results, new_row], ignore_index=True)

results.sort_values(by=['Average CRPS'], ascending=True, inplace=True)

print("\nModel Comparison:")
print(results.to_latex(index=False))
#results.to_csv('vulnerable_growth_model_comparison.csv')

#%% Plot the time series of the target variable (merged_df[ycol]) and the CRPS scores for each model
# Create a figure and primary axis for the GDP growth
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot the observed GDP growth on the primary y-axis
ax1.plot(merged_df.index[jtFirstOOS:], y_oos, label='Observed ' + ycol, color='black', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('GDP Growth')
ax1.tick_params(axis='y')

# Create a secondary y-axis for the CRPS scores
ax2 = ax1.twinx()
ax2.set_ylabel('CRPS Score')

# Plot the CRPS scores for each model on the secondary y-axis
for model_name, crps_score in crps_scores_dict.items():
    ax2.plot(merged_df.index[jtFirstOOS:], crps_score, label=f'{model_name} CRPS', linestyle='--')

# Add title and legends for both axes
fig.suptitle('Time Series of Observed GDP Growth and CRPS Scores for Each Model')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add grid and layout adjustments
ax1.grid()
fig.tight_layout()

# Save the plot as a PDF
#plt.savefig('fig/CRPS_vulnerable_growth.pdf')

# Show plot
plt.show()


# Plotting median and IQR for each model along with target variable
for model_name in models.keys():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract quantile data for the model
    median_values = quantile_fitted_dict[model_name][:, 2]  # 0.5 quantile (median)
    lower_quartile = quantile_fitted_dict[model_name][:, 1]  # 0.25 quantile
    upper_quartile = quantile_fitted_dict[model_name][:, 3]  # 0.75 quantile

    # Plot the median and interquartile range
    ax.plot(merged_df.index[jtFirstOOS:], median_values, label=f'{model_name} Median', color='blue', linewidth=2)
    ax.fill_between(merged_df.index[jtFirstOOS:], lower_quartile, upper_quartile, color='lightblue', alpha=0.5, label=f'{model_name} IQR')

    # Plot the target variable
    ax.plot(merged_df.index[jtFirstOOS:], y_oos, label=f'Observed {ycol}', color='black', linewidth=2)

    # Set plot labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('GDP Growth')
    ax.set_title(f'Median and Interquartile Range for {model_name}')
    
    # Add legend and grid
    ax.legend()
    ax.grid()
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the plot
    # plt.savefig(f'fig/{model_name.replace(" ", "_")}_median_IQR_with_target.pdf')
    
    # Show the plot
    plt.show()

