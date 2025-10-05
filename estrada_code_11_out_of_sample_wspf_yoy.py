#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 17:08:38 2025

@author: dianniadreiestrada
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import t
import numpy as np

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

# Visualization of trends for average, median, and percentile ranges
def plot_trends(stats_df, date_col, point_col):
    plt.figure(figsize=(14, 7))
    plt.plot(stats_df[date_col], stats_df['average_point'], label='Average', marker='o', color='blue')
    plt.plot(stats_df[date_col], stats_df['median_point'], label='Median', marker='o', color='orange')

    # Fill between percentile ranges
    plt.fill_between(stats_df[date_col], stats_df['p5_point'], stats_df['p90_point'], color='gray', alpha=0.2, label='5th to 90th Percentile')
    plt.fill_between(stats_df[date_col], stats_df['p25_point'], stats_df['p75_point'], color='lightblue', alpha=0.4, label='Interquartile Range (25th to 75th Percentile)')

    plt.xticks(rotation=45)
    plt.title("Trends in Average, Median, and Percentiles of Point Forecasts")
    plt.ylabel("Point Forecast")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot trends for Point Forecasts
plot_trends(statistics_df, "Date", "POINT")



#%% Function to compute the quantile for the skewed t-distribution

minimize_iter_max = 100

# Function to calculate the quantile for the skewed t-distribution
def qskt(p, mu, sigma, alpha, nu):
    """
    Calculate the quantile for the skewed t-distribution.
    """
    quantile = mu + sigma * t.ppf(p, df=nu) * (1 + alpha * np.sign(p - 0.5))
    return quantile

# Function to perform quantile interpolation for skewed t-distribution
def quantiles_interpolation(quantiles, quantile_targets):
    """
    Fit the parameters of the skewed t-distribution to the provided quantiles.
    """
    def objective(params):
        mu, sigma, alpha, nu = params
        theoretical_quantiles = [qskt(q, mu, sigma, alpha, nu) for q in quantile_targets]
        error = np.sum((np.array(theoretical_quantiles) - np.array(quantiles)) ** 2)
        return error

    initial_guess = [0, 1, 0, 5]
    bounds = [(-np.inf, np.inf), (1e-6, np.inf), (-np.inf, np.inf), (2, np.inf)]
    result = minimize(objective, initial_guess, bounds=bounds, options={'maxiter': minimize_iter_max})
    mu, sigma, alpha, nu = result.x
    return mu, sigma, alpha, nu

# Function to calculate the cumulative distribution function of the skewed t-distribution
def pskt(x, mu, sigma, alpha, nu):
    """
    Calculate the cumulative distribution function (CDF) for the skewed t-distribution.
    """
    z = (x - mu) / sigma
    cdf_value = t.cdf(z * (1 + alpha * np.sign(z)), df=nu)
    return cdf_value

# Function to calculate the empirical CDF of PIT values
def empirical_cdf(pit_values, rvec):
    """
    Compute the empirical cumulative distribution function (CDF) of the PIT values.
    """
    sorted_pits = np.sort(pit_values)
    empirical_cdf_values = np.searchsorted(sorted_pits, rvec, side='right') / len(sorted_pits)
    return empirical_cdf_values


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
h = 4  # Set to 1 for one-quarter ahead prediction
ycol = 'g_RGDP' if h == 1 else 'g_RGDPyoy'
merged_df['g_RGDP_original'] = merged_df[ycol]
merged_df[ycol] = merged_df[ycol].shift(-h)
merged_df.dropna(subset=['CISS', ycol, 'g_RGDP_original'], inplace=True)

jtFirstOOS = merged_df.index.get_loc(pd.Timestamp('1995-01-01'))


#Mean Point Forecast Trends

plt.figure(figsize=(14, 7))

# Plot the actual ycol and mean forecast, with mean forecast in red
merged_df[ycol].plot(ax=plt.gca(), marker='o', linewidth=1, label=ycol)
merged_df['SPF_mean_forecast'].plot(ax=plt.gca(), marker='o', linewidth=1, color='yellow', label='SPF Mean Forecast')

# Add shaded areas for percentiles
plt.fill_between(merged_df.index, merged_df['SPF_5th_percentile'], merged_df['SPF_90th_percentile'], alpha=0.2, color='gray', label='5th to 90th Percentile')
plt.fill_between(merged_df.index, merged_df['SPF_25th_percentile'], merged_df['SPF_75th_percentile'], alpha=0.4, color='lightblue', label='25th to 75th Percentile (IQR)')


plt.title('Mean Forecast Trends with Percentiles')
plt.xlabel('Date')
plt.ylabel('Forecast Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


#Median Point Forecast Trends

plt.figure(figsize=(14, 7))

# Plot the actual ycol and mean forecast, with mean forecast in red
merged_df[ycol].plot(ax=plt.gca(), marker='o', linewidth=1, label=ycol)
merged_df['SPF_median_forecast'].plot(ax=plt.gca(), marker='o', linewidth=1, color='orange', label='SPF Median Forecast')

# Add shaded areas for percentiles
plt.fill_between(merged_df.index, merged_df['SPF_5th_percentile'], merged_df['SPF_90th_percentile'], alpha=0.2, color='gray', label='5th to 90th Percentile')
plt.fill_between(merged_df.index, merged_df['SPF_25th_percentile'], merged_df['SPF_75th_percentile'], alpha=0.4, color='lightblue', label='25th to 75th Percentile (IQR)')

plt.title('Median Forecast Trends with Percentiles')
plt.xlabel('Date')
plt.ylabel('Forecast Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


#5th Percentile Forecast Trends

plt.figure(figsize=(14, 7))

# Plot the actual ycol and mean forecast, with mean forecast in red
merged_df[ycol].plot(ax=plt.gca(), marker='o', linewidth=1, label=ycol)
merged_df['SPF_5th_percentile'].plot(ax=plt.gca(), marker='o', linewidth=1, color='orange', label='5th Percentile Forecast')

plt.title('5th Percentile Forecast Trends')
plt.xlabel('Date')
plt.ylabel('Forecast Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


#25th Percentile Forecast Trends

plt.figure(figsize=(14, 7))

# Plot the actual ycol and mean forecast, with mean forecast in red
merged_df[ycol].plot(ax=plt.gca(), marker='o', linewidth=1, label=ycol)
merged_df['SPF_25th_percentile'].plot(ax=plt.gca(), marker='o', linewidth=1, color='orange', label='25th Percentile Forecast')

plt.title('25th Percentile Forecast Trends')
plt.xlabel('Date')
plt.ylabel('Forecast Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


#75th Percentile Forecast Trends

plt.figure(figsize=(14, 7))

# Plot the actual ycol and mean forecast, with mean forecast in red
merged_df[ycol].plot(ax=plt.gca(), marker='o', linewidth=1, label=ycol)
merged_df['SPF_75th_percentile'].plot(ax=plt.gca(), marker='o', linewidth=1, color='orange', label='75th Percentile Forecast')

plt.title('75th Percentile Forecast Trends')
plt.xlabel('Date')
plt.ylabel('Forecast Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


#90th Percentile Forecast Trends

plt.figure(figsize=(14, 7))

# Plot the actual ycol and mean forecast, with mean forecast in red
merged_df[ycol].plot(ax=plt.gca(), marker='o', linewidth=1, label=ycol)
merged_df['SPF_75th_percentile'].plot(ax=plt.gca(), marker='o', linewidth=1, color='orange', label='75th Percentile Forecast')

plt.title('75th Percentile Forecast Trends')
plt.xlabel('Date')
plt.ylabel('Forecast Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()




#%% Out-of-Sample Quantile Regression with CISS, GDP, and mean forecast
rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
T_oos = len(merged_df) - jtFirstOOS

quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))


# Perform quantile regression with a rolling window approach
for ti in range(jtFirstOOS, len(merged_df)):
    train_merged_df = merged_df.iloc[ti - rolling_window_size:ti].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_merged_df) < rolling_window_size:
        print(f"Skipping index {ti} due to insufficient data.")
        continue
    
    X_train = sm.add_constant(train_merged_df[['CISS', 'g_RGDP_original', 'SPF_mean_forecast']])
    y_train = train_merged_df[ycol]
    
    if X_train.isnull().values.any() or y_train.isnull().any():
        print(f"Skipping index {ti} due to NaN values in training data.")
        continue
    
    for i, q in enumerate(quantile_levels):
        try:
            quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
            quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict(
                [1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_mean_forecast'].iloc[ti]]
            )[0]
        except Exception as e:
            print(f"Quantile regression failed at index {ti}, quantile {q}: {e}")
            continue


# Fit Skewed t-Distribution for each out-of-sample time step
tdist_mu_oos = np.zeros(T_oos)
tdist_sigma_oos = np.zeros(T_oos)
tdist_alpha_oos = np.zeros(T_oos)
tdist_nu_oos = np.zeros(T_oos)

quantile_targets = [0.05, 0.25, 0.75, 0.95]


# Out-of-sample processing
for ti in range(T_oos):
    quantiles_to_fit = quantile_fitted_oos[ti, [0, 1, 3, 4]]
    mu, sigma, alpha, nu = quantiles_interpolation(quantiles_to_fit, quantile_targets)
    tdist_mu_oos[ti], tdist_sigma_oos[ti], tdist_alpha_oos[ti], tdist_nu_oos[ti] = mu, sigma, alpha, nu

# Calculate PIT values
def calculate_pits_oos(y_true, params):
    """
    Calculate the Probability Integral Transform (PIT) values for out-of-sample data.
    """
    pits = np.zeros(len(y_true))
    for i in range(len(y_true)):
        mu, sigma, alpha, nu = params[i]
        pits[i] = pskt(y_true[i], mu, sigma, alpha, nu)
    return pits


# Generate PITs for out-of-sample predictions
y_oos = merged_df[ycol].iloc[jtFirstOOS:].values
skewed_t_params_oos = [(tdist_mu_oos[i], tdist_sigma_oos[i], tdist_alpha_oos[i], tdist_nu_oos[i]) for i in range(T_oos)]
pits_skewed_t_oos = calculate_pits_oos(y_oos, skewed_t_params_oos)

# Set rvec for plotting
rvec = np.arange(0, 1.001, 0.001)

# Calculate empirical CDF for the PIT values
zST_ecdf = empirical_cdf(pits_skewed_t_oos, rvec)

# Determine critical values for the PIT test
kappa = 1.34  # Asymptotic critical value for h=1

# Plot the PITs empirical CDF vs. theoretical CDF
plt.figure(figsize=(12, 8))
plt.plot(rvec, zST_ecdf, '-b', label='GDP, SPF mean forecast and CISS')
plt.plot(rvec, rvec, 'k--', label='Theoretical 45-degree line')
plt.fill_between(rvec, rvec - (kappa / np.sqrt(len(pits_skewed_t_oos))), 
                 rvec + (kappa / np.sqrt(len(pits_skewed_t_oos))), color='b', alpha=0.2, label='95% Confidence Band')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('Out-of-Sample PIT Empirical CDF', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()


#%% Determine the starting index for out-of-sample prediction
jtFirstOOS = merged_df.index.get_loc(pd.Timestamp('1995-01-01'))

# Out-of-Sample Quantile Regression with CISS, GDP, and median forecast
rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
T_oos = len(merged_df) - jtFirstOOS

quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))


# Perform quantile regression with a rolling window approach
for ti in range(jtFirstOOS, len(merged_df)):
    train_merged_df = merged_df.iloc[ti - rolling_window_size:ti].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_merged_df) < rolling_window_size:
        print(f"Skipping index {ti} due to insufficient data.")
        continue
    
    X_train = sm.add_constant(train_merged_df[['CISS', 'g_RGDP_original', 'SPF_median_forecast']])
    y_train = train_merged_df[ycol]
    
    if X_train.isnull().values.any() or y_train.isnull().any():
        print(f"Skipping index {ti} due to NaN values in training data.")
        continue
    
    for i, q in enumerate(quantile_levels):
        try:
            quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
            quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict(
                [1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_median_forecast'].iloc[ti]]
            )[0]
        except Exception as e:
            print(f"Quantile regression failed at index {ti}, quantile {q}: {e}")
            continue


# Fit Skewed t-Distribution for each out-of-sample time step
tdist_mu_oos = np.zeros(T_oos)
tdist_sigma_oos = np.zeros(T_oos)
tdist_alpha_oos = np.zeros(T_oos)
tdist_nu_oos = np.zeros(T_oos)

quantile_targets = [0.05, 0.25, 0.75, 0.95]


# Out-of-sample processing
for ti in range(T_oos):
    quantiles_to_fit = quantile_fitted_oos[ti, [0, 1, 3, 4]]
    mu, sigma, alpha, nu = quantiles_interpolation(quantiles_to_fit, quantile_targets)
    tdist_mu_oos[ti], tdist_sigma_oos[ti], tdist_alpha_oos[ti], tdist_nu_oos[ti] = mu, sigma, alpha, nu

# Calculate PIT values
def calculate_pits_oos(y_true, params):
    """
    Calculate the Probability Integral Transform (PIT) values for out-of-sample data.
    """
    pits = np.zeros(len(y_true))
    for i in range(len(y_true)):
        mu, sigma, alpha, nu = params[i]
        pits[i] = pskt(y_true[i], mu, sigma, alpha, nu)
    return pits


# Generate PITs for out-of-sample predictions
y_oos = merged_df[ycol].iloc[jtFirstOOS:].values
skewed_t_params_oos = [(tdist_mu_oos[i], tdist_sigma_oos[i], tdist_alpha_oos[i], tdist_nu_oos[i]) for i in range(T_oos)]
pits_skewed_t_oos = calculate_pits_oos(y_oos, skewed_t_params_oos)

# Set rvec for plotting
rvec = np.arange(0, 1.001, 0.001)

# Calculate empirical CDF for the PIT values
zST_ecdf = empirical_cdf(pits_skewed_t_oos, rvec)

# Determine critical values for the PIT test
kappa = 1.34  # Asymptotic critical value for h=1

# Plot the PITs empirical CDF vs. theoretical CDF
plt.figure(figsize=(12, 8))
plt.plot(rvec, zST_ecdf, '-b', label='GDP, SPF median forecast and CISS')
plt.plot(rvec, rvec, 'k--', label='Theoretical 45-degree line')
plt.fill_between(rvec, rvec - (kappa / np.sqrt(len(pits_skewed_t_oos))), 
                 rvec + (kappa / np.sqrt(len(pits_skewed_t_oos))), color='b', alpha=0.2, label='95% Confidence Band')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('Out-of-Sample PIT Empirical CDF', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()


#%% Determine the starting index for out-of-sample prediction
jtFirstOOS = merged_df.index.get_loc(pd.Timestamp('1995-01-01'))

# Out-of-Sample Quantile Regression with CISS, GDP, and 5th percentile forecast
rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
T_oos = len(merged_df) - jtFirstOOS

quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))


# Perform quantile regression with a rolling window approach
for ti in range(jtFirstOOS, len(merged_df)):
    train_merged_df = merged_df.iloc[ti - rolling_window_size:ti].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_merged_df) < rolling_window_size:
        print(f"Skipping index {ti} due to insufficient data.")
        continue
    
    X_train = sm.add_constant(train_merged_df[['CISS', 'g_RGDP_original', 'SPF_5th_percentile']])
    y_train = train_merged_df[ycol]
    
    if X_train.isnull().values.any() or y_train.isnull().any():
        print(f"Skipping index {ti} due to NaN values in training data.")
        continue
    
    for i, q in enumerate(quantile_levels):
        try:
            quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
            quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict(
                [1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_5th_percentile'].iloc[ti]]
            )[0]
        except Exception as e:
            print(f"Quantile regression failed at index {ti}, quantile {q}: {e}")
            continue


# Fit Skewed t-Distribution for each out-of-sample time step
tdist_mu_oos = np.zeros(T_oos)
tdist_sigma_oos = np.zeros(T_oos)
tdist_alpha_oos = np.zeros(T_oos)
tdist_nu_oos = np.zeros(T_oos)

quantile_targets = [0.05, 0.25, 0.75, 0.95]


# Out-of-sample processing
for ti in range(T_oos):
    quantiles_to_fit = quantile_fitted_oos[ti, [0, 1, 3, 4]]
    mu, sigma, alpha, nu = quantiles_interpolation(quantiles_to_fit, quantile_targets)
    tdist_mu_oos[ti], tdist_sigma_oos[ti], tdist_alpha_oos[ti], tdist_nu_oos[ti] = mu, sigma, alpha, nu

# Calculate PIT values
def calculate_pits_oos(y_true, params):
    """
    Calculate the Probability Integral Transform (PIT) values for out-of-sample data.
    """
    pits = np.zeros(len(y_true))
    for i in range(len(y_true)):
        mu, sigma, alpha, nu = params[i]
        pits[i] = pskt(y_true[i], mu, sigma, alpha, nu)
    return pits


# Generate PITs for out-of-sample predictions
y_oos = merged_df[ycol].iloc[jtFirstOOS:].values
skewed_t_params_oos = [(tdist_mu_oos[i], tdist_sigma_oos[i], tdist_alpha_oos[i], tdist_nu_oos[i]) for i in range(T_oos)]
pits_skewed_t_oos = calculate_pits_oos(y_oos, skewed_t_params_oos)

# Set rvec for plotting
rvec = np.arange(0, 1.001, 0.001)

# Calculate empirical CDF for the PIT values
zST_ecdf = empirical_cdf(pits_skewed_t_oos, rvec)

# Determine critical values for the PIT test
kappa = 1.34  # Asymptotic critical value for h=1

# Plot the PITs empirical CDF vs. theoretical CDF
plt.figure(figsize=(12, 8))
plt.plot(rvec, zST_ecdf, '-b', label='GDP, SPF 5th Percentile Forecast and CISS')
plt.plot(rvec, rvec, 'k--', label='Theoretical 45-degree line')
plt.fill_between(rvec, rvec - (kappa / np.sqrt(len(pits_skewed_t_oos))), 
                 rvec + (kappa / np.sqrt(len(pits_skewed_t_oos))), color='b', alpha=0.2, label='95% Confidence Band')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('Out-of-Sample PIT Empirical CDF', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()


#%% Determine the starting index for out-of-sample prediction
jtFirstOOS = merged_df.index.get_loc(pd.Timestamp('1995-01-01'))

# Out-of-Sample Quantile Regression with CISS, GDP, and 25th percentile forecast
rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
T_oos = len(merged_df) - jtFirstOOS

quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))


# Perform quantile regression with a rolling window approach
for ti in range(jtFirstOOS, len(merged_df)):
    train_merged_df = merged_df.iloc[ti - rolling_window_size:ti].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_merged_df) < rolling_window_size:
        print(f"Skipping index {ti} due to insufficient data.")
        continue
    
    X_train = sm.add_constant(train_merged_df[['CISS', 'g_RGDP_original', 'SPF_25th_percentile']])
    y_train = train_merged_df[ycol]
    
    if X_train.isnull().values.any() or y_train.isnull().any():
        print(f"Skipping index {ti} due to NaN values in training data.")
        continue
    
    for i, q in enumerate(quantile_levels):
        try:
            quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
            quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict(
                [1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_25th_percentile'].iloc[ti]]
            )[0]
        except Exception as e:
            print(f"Quantile regression failed at index {ti}, quantile {q}: {e}")
            continue


# Fit Skewed t-Distribution for each out-of-sample time step
tdist_mu_oos = np.zeros(T_oos)
tdist_sigma_oos = np.zeros(T_oos)
tdist_alpha_oos = np.zeros(T_oos)
tdist_nu_oos = np.zeros(T_oos)

quantile_targets = [0.05, 0.25, 0.75, 0.95]


# Out-of-sample processing
for ti in range(T_oos):
    quantiles_to_fit = quantile_fitted_oos[ti, [0, 1, 3, 4]]
    mu, sigma, alpha, nu = quantiles_interpolation(quantiles_to_fit, quantile_targets)
    tdist_mu_oos[ti], tdist_sigma_oos[ti], tdist_alpha_oos[ti], tdist_nu_oos[ti] = mu, sigma, alpha, nu

# Calculate PIT values
def calculate_pits_oos(y_true, params):
    """
    Calculate the Probability Integral Transform (PIT) values for out-of-sample data.
    """
    pits = np.zeros(len(y_true))
    for i in range(len(y_true)):
        mu, sigma, alpha, nu = params[i]
        pits[i] = pskt(y_true[i], mu, sigma, alpha, nu)
    return pits


# Generate PITs for out-of-sample predictions
y_oos = merged_df[ycol].iloc[jtFirstOOS:].values
skewed_t_params_oos = [(tdist_mu_oos[i], tdist_sigma_oos[i], tdist_alpha_oos[i], tdist_nu_oos[i]) for i in range(T_oos)]
pits_skewed_t_oos = calculate_pits_oos(y_oos, skewed_t_params_oos)

# Set rvec for plotting
rvec = np.arange(0, 1.001, 0.001)

# Calculate empirical CDF for the PIT values
zST_ecdf = empirical_cdf(pits_skewed_t_oos, rvec)

# Determine critical values for the PIT test
kappa = 1.34  # Asymptotic critical value for h=1

# Plot the PITs empirical CDF vs. theoretical CDF
plt.figure(figsize=(12, 8))
plt.plot(rvec, zST_ecdf, '-b', label='GDP, SPF 25th Percentile Forecast and CISS')
plt.plot(rvec, rvec, 'k--', label='Theoretical 45-degree line')
plt.fill_between(rvec, rvec - (kappa / np.sqrt(len(pits_skewed_t_oos))), 
                 rvec + (kappa / np.sqrt(len(pits_skewed_t_oos))), color='b', alpha=0.2, label='95% Confidence Band')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('Out-of-Sample PIT Empirical CDF', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()



#%% Determine the starting index for out-of-sample prediction
jtFirstOOS = merged_df.index.get_loc(pd.Timestamp('1995-01-01'))

# Out-of-Sample Quantile Regression with CISS, GDP, and 75th percentile forecast
rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
T_oos = len(merged_df) - jtFirstOOS

quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))


# Perform quantile regression with a rolling window approach
for ti in range(jtFirstOOS, len(merged_df)):
    train_merged_df = merged_df.iloc[ti - rolling_window_size:ti].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_merged_df) < rolling_window_size:
        print(f"Skipping index {ti} due to insufficient data.")
        continue
    
    X_train = sm.add_constant(train_merged_df[['CISS', 'g_RGDP_original', 'SPF_75th_percentile']])
    y_train = train_merged_df[ycol]
    
    if X_train.isnull().values.any() or y_train.isnull().any():
        print(f"Skipping index {ti} due to NaN values in training data.")
        continue
    
    for i, q in enumerate(quantile_levels):
        try:
            quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
            quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict(
                [1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_75th_percentile'].iloc[ti]]
            )[0]
        except Exception as e:
            print(f"Quantile regression failed at index {ti}, quantile {q}: {e}")
            continue


# Fit Skewed t-Distribution for each out-of-sample time step
tdist_mu_oos = np.zeros(T_oos)
tdist_sigma_oos = np.zeros(T_oos)
tdist_alpha_oos = np.zeros(T_oos)
tdist_nu_oos = np.zeros(T_oos)

quantile_targets = [0.05, 0.25, 0.75, 0.95]


# Out-of-sample processing
for ti in range(T_oos):
    quantiles_to_fit = quantile_fitted_oos[ti, [0, 1, 3, 4]]
    mu, sigma, alpha, nu = quantiles_interpolation(quantiles_to_fit, quantile_targets)
    tdist_mu_oos[ti], tdist_sigma_oos[ti], tdist_alpha_oos[ti], tdist_nu_oos[ti] = mu, sigma, alpha, nu

# Calculate PIT values
def calculate_pits_oos(y_true, params):
    """
    Calculate the Probability Integral Transform (PIT) values for out-of-sample data.
    """
    pits = np.zeros(len(y_true))
    for i in range(len(y_true)):
        mu, sigma, alpha, nu = params[i]
        pits[i] = pskt(y_true[i], mu, sigma, alpha, nu)
    return pits


# Generate PITs for out-of-sample predictions
y_oos = merged_df[ycol].iloc[jtFirstOOS:].values
skewed_t_params_oos = [(tdist_mu_oos[i], tdist_sigma_oos[i], tdist_alpha_oos[i], tdist_nu_oos[i]) for i in range(T_oos)]
pits_skewed_t_oos = calculate_pits_oos(y_oos, skewed_t_params_oos)

# Set rvec for plotting
rvec = np.arange(0, 1.001, 0.001)

# Calculate empirical CDF for the PIT values
zST_ecdf = empirical_cdf(pits_skewed_t_oos, rvec)

# Determine critical values for the PIT test
kappa = 1.34  # Asymptotic critical value for h=1

# Plot the PITs empirical CDF vs. theoretical CDF
plt.figure(figsize=(12, 8))
plt.plot(rvec, zST_ecdf, '-b', label='GDP, SPF 75th Percentile Forecast and CISS')
plt.plot(rvec, rvec, 'k--', label='Theoretical 45-degree line')
plt.fill_between(rvec, rvec - (kappa / np.sqrt(len(pits_skewed_t_oos))), 
                 rvec + (kappa / np.sqrt(len(pits_skewed_t_oos))), color='b', alpha=0.2, label='95% Confidence Band')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('Out-of-Sample PIT Empirical CDF', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()



#%% Determine the starting index for out-of-sample prediction
jtFirstOOS = merged_df.index.get_loc(pd.Timestamp('1995-01-01'))

# Out-of-Sample Quantile Regression with CISS, GDP, and 90th percentile forecast
rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
T_oos = len(merged_df) - jtFirstOOS

quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))


# Perform quantile regression with a rolling window approach
for ti in range(jtFirstOOS, len(merged_df)):
    train_merged_df = merged_df.iloc[ti - rolling_window_size:ti].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_merged_df) < rolling_window_size:
        print(f"Skipping index {ti} due to insufficient data.")
        continue
    
    X_train = sm.add_constant(train_merged_df[['CISS', 'g_RGDP_original', 'SPF_90th_percentile']])
    y_train = train_merged_df[ycol]
    
    if X_train.isnull().values.any() or y_train.isnull().any():
        print(f"Skipping index {ti} due to NaN values in training data.")
        continue
    
    for i, q in enumerate(quantile_levels):
        try:
            quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
            quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict(
                [1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_90th_percentile'].iloc[ti]]
            )[0]
        except Exception as e:
            print(f"Quantile regression failed at index {ti}, quantile {q}: {e}")
            continue


# Fit Skewed t-Distribution for each out-of-sample time step
tdist_mu_oos = np.zeros(T_oos)
tdist_sigma_oos = np.zeros(T_oos)
tdist_alpha_oos = np.zeros(T_oos)
tdist_nu_oos = np.zeros(T_oos)

quantile_targets = [0.05, 0.25, 0.75, 0.95]


# Out-of-sample processing
for ti in range(T_oos):
    quantiles_to_fit = quantile_fitted_oos[ti, [0, 1, 3, 4]]
    mu, sigma, alpha, nu = quantiles_interpolation(quantiles_to_fit, quantile_targets)
    tdist_mu_oos[ti], tdist_sigma_oos[ti], tdist_alpha_oos[ti], tdist_nu_oos[ti] = mu, sigma, alpha, nu

# Calculate PIT values
def calculate_pits_oos(y_true, params):
    """
    Calculate the Probability Integral Transform (PIT) values for out-of-sample data.
    """
    pits = np.zeros(len(y_true))
    for i in range(len(y_true)):
        mu, sigma, alpha, nu = params[i]
        pits[i] = pskt(y_true[i], mu, sigma, alpha, nu)
    return pits


# Generate PITs for out-of-sample predictions
y_oos = merged_df[ycol].iloc[jtFirstOOS:].values
skewed_t_params_oos = [(tdist_mu_oos[i], tdist_sigma_oos[i], tdist_alpha_oos[i], tdist_nu_oos[i]) for i in range(T_oos)]
pits_skewed_t_oos = calculate_pits_oos(y_oos, skewed_t_params_oos)

# Set rvec for plotting
rvec = np.arange(0, 1.001, 0.001)

# Calculate empirical CDF for the PIT values
zST_ecdf = empirical_cdf(pits_skewed_t_oos, rvec)

# Determine critical values for the PIT test
kappa = 1.34  # Asymptotic critical value for h=1

# Plot the PITs empirical CDF vs. theoretical CDF
plt.figure(figsize=(12, 8))
plt.plot(rvec, zST_ecdf, '-b', label='GDP, SPF 90th Percentile Forecast and CISS')
plt.plot(rvec, rvec, 'k--', label='Theoretical 45-degree line')
plt.fill_between(rvec, rvec - (kappa / np.sqrt(len(pits_skewed_t_oos))), 
                 rvec + (kappa / np.sqrt(len(pits_skewed_t_oos))), color='b', alpha=0.2, label='95% Confidence Band')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('Out-of-Sample PIT Empirical CDF', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()
