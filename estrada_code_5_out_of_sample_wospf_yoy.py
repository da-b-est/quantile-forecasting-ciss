#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 00:50:51 2025

@author: dianniadreiestrada
"""

#Replication of: "Vulnerable Growth" Adrian et al. (2019)

#Out-of-Sample PIT Calculation and Specification Test

import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set the working directory
os.chdir('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise')

# Set a limit for minimization iterations
minimize_iter_max = 100

# Function to compute the quantile for the skewed t-distribution
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

# Load the data
df = pd.read_excel('data/Data_Adrian_2019.xlsx', sheet_name='EA_Quarterly')
df.index = pd.to_datetime(df.Date)
df.sort_index(inplace=True)
df['Intercept'] = 1

# Define forecasting horizon
h = 4  # Set to 1 for one-quarter ahead prediction
ycol = 'g_RGDP' if h == 1 else 'g_RGDPyoy'
df['g_RGDP_original'] = df[ycol]
df[ycol] = df[ycol].shift(-h)
df.dropna(subset=['CISS', ycol, 'g_RGDP_original'], inplace=True)

# Determine the starting index for out-of-sample prediction
jtFirstOOS = df.index.get_loc(pd.Timestamp('1995-01-01'))

# Out-of-Sample Quantile Regression with CISS and GDP
rolling_window_size = 40
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
T_oos = len(df) - jtFirstOOS

quantile_fitted_oos = np.zeros((T_oos, len(quantile_levels)))

# Perform quantile regression with a rolling window approach
for ti in range(jtFirstOOS, len(df)):
    # Use data up to the current time step `ti` (rolling window)
    train_df = df.iloc[ti - rolling_window_size:ti]
    X_train = sm.add_constant(train_df[['CISS', 'g_RGDP_original']])
    y_train = train_df[ycol]

    # Estimate quantiles for each quantile level
    for i, q in enumerate(quantile_levels):
        quantile_model = sm.QuantReg(y_train, X_train).fit(q=q)
        quantile_fitted_oos[ti - jtFirstOOS, i] = quantile_model.predict([1, df['CISS'].iloc[ti], df['g_RGDP_original'].iloc[ti]])[0]

# Fit Skewed t-Distribution for each out-of-sample time step
tdist_mu_oos = np.zeros(T_oos)
tdist_sigma_oos = np.zeros(T_oos)
tdist_alpha_oos = np.zeros(T_oos)
tdist_nu_oos = np.zeros(T_oos)

quantile_targets = [0.05, 0.25, 0.75, 0.95]

# Fit skewed-t distribution using the quantile estimates
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
y_oos = df[ycol].iloc[jtFirstOOS:].values
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
plt.plot(rvec, zST_ecdf, '-b', label='GDP and CISS')
plt.plot(rvec, rvec, 'k--', label='Theoretical 45-degree line')
plt.fill_between(rvec, rvec - (kappa / np.sqrt(len(pits_skewed_t_oos))), 
                 rvec + (kappa / np.sqrt(len(pits_skewed_t_oos))), color='b', alpha=0.2, label='95% Confidence Band')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Empirical CDF', fontsize=14)
plt.title('Out-of-Sample PIT Empirical CDF (YoY)', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()
