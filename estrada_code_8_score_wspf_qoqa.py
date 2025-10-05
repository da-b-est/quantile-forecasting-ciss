#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:26:43 2025

@author: dianniadreiestrada
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 08:29:30 2024

@author: dianniadreiestrada
"""

#Replication of: "Vulnerable Growth" Adrian et al. (2019)

#Part 2: fit distributions, plot 3D densities, and compute the score


# Importing necessary libraries
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import seaborn as sns
from scipy.stats import norm

minimize_iter_max = 50

# Set the working directory (useful to choose the folder where to export output files)
os.chdir('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise')

# Function to compute the quantile for the skewed t-distribution
def qskt(p, mu, sigma, alpha, nu):
    from scipy.stats import t
    quantile = mu + sigma * t.ppf(p, df=nu) * (1 + alpha * np.sign(p - 0.5))
    return quantile

# Function to perform quantile interpolation for skewed t-distribution
def quantiles_interpolation(quantiles, quantile_targets):
    def objective(params):
        mu, sigma, alpha, nu = params
        theoretical_quantiles = [qskt(q, mu, sigma, alpha, nu) for q in quantile_targets]
        error = np.sum((np.array(theoretical_quantiles) - np.array(quantiles)) ** 2)
        return error

    initial_guess = [0, 1, 0, 5]
    bounds = [(-np.inf, np.inf), (1e-6, np.inf), (-np.inf, np.inf), (2, np.inf)]
    result = minimize(objective, initial_guess, bounds=bounds,options={'maxiter': minimize_iter_max})
    mu, sigma, alpha, nu = result.x
    return mu, sigma, alpha, nu

# Function to calculate the density of the skewed t-distribution
def dskt(x, mu, sigma, alpha, nu):
    """
    Calculate the density of the skewed t-distribution.

    Parameters:
    - x: The points at which to evaluate the density.
    - mu: Location parameter.
    - sigma: Scale parameter.
    - alpha: Skewness parameter.
    - nu: Degrees of freedom.

    Returns:
    - Density values of the skewed t-distribution at the points x.
    """
    # Standardize the input
    z = (x - mu) / sigma
    
    # Compute the skewed t-density
    normalization = 2 / sigma * t.pdf(z, df=nu) * t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
    return normalization


# Load the data
spf_df = pd.read_csv('Data/spf_df.csv')

# Ensure numeric columns are properly recognized
numeric_columns = spf_df.columns.difference(['Date', 'FCT_SOURCE'])
spf_df[numeric_columns] = spf_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group data by Date and FCT_SOURCE to visualize distribution
def plot_distributions(spf_df, date_col, source_col, point_col):
    plt.figure(figsize=(14, 7))
    sns.boxplot(x=date_col, y=point_col, hue=source_col, data=spf_df, showfliers=False)
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
def calculate_statistics(spf_df, date_col):
    # Group by date
    stats = spf_df.groupby(date_col).agg(
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

merged_df.to_csv('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise/Data/merged_df.csv', index=False)

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

# Define which column is the dependent variable (based on h)
h = 1  # Change this to 4 for YoY

if h == 1:
    ycol = 'g_RGDP'
    label_text = "GDP Growth 1 Quarter Ahead (QoQA)"
elif h == 4:
    ycol = 'g_RGDPyoy'
    label_text = "GDP Growth 4 Quarters Ahead (YoY)"


#Prediction: shift GDP growth to see if we can predict it
# Replace the dependent variable with its shifted values
merged_df['g_RGDP_original'] = merged_df[ycol]
merged_df[ycol] = merged_df[ycol].shift(-h)

# Drop rows with NaN values in the 'CISS' or dependent variable
merged_df.dropna(subset=['CISS', ycol,'g_RGDP_original', 'SPF_mean_forecast', 'SPF_median_forecast'], inplace=True)

# Function to get recession dates from the dataset
def get_recession_dates(merged_df, recession_indicator_column):
    """
    Identify recession start and end dates based on the recession indicator column.
    """
    start_dates = []
    end_dates = []
    T = len(merged_df)

    for ti in range(T):
        date_aux = merged_df.index[ti]  # Assuming the DataFrame index is the date
        x_aux = merged_df[recession_indicator_column].iloc[ti]

        if x_aux == 1:
            # Determine the start and end of the recession
            year_aux = date_aux.year
            quarter_aux = (date_aux.month - 1) // 3 + 1
            start_date = pd.Timestamp(f"{year_aux}-{quarter_aux * 3 - 2}-01")
            end_date = pd.Timestamp(f"{year_aux}-{quarter_aux * 3}-01") + pd.offsets.QuarterEnd(0)

            # Add start and end dates to the list
            start_dates.append(start_date)
            end_dates.append(end_date)

    # Combine start and end dates into a list of tuples
    recession_dates = list(zip(start_dates, end_dates))
    return recession_dates

# Get the recession dates from the DataFrame
merged_df['Recession'] = ((merged_df[ycol] < 0) & (merged_df[ycol].shift(1) < 0)).astype(int)
recession_dates = get_recession_dates(merged_df, 'Recession')


# Function to add recession shading to plots
def add_recession_shading(ax, recession_dates, color='gray', alpha=0.3):
    """
    Add shaded areas to represent recession periods on a given axis.
    """
    for start_date, end_date in recession_dates:
        ax.axvspan(start_date, end_date, color=color, alpha=alpha)


##### SPF Forecast (Mean) #####
# Perform linear regression to get the model parameters (particularly for SPF_mean)
X = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'GDP': merged_df['g_RGDP_original'], 'SPF (Mean)': merged_df['SPF_mean_forecast']})   # Predictor variables (including intercept)
y = merged_df[ycol]  # Response variable
model = sm.OLS(y, X).fit()  # Fit the linear regression model
linreg_sigma = np.std(model.resid)  # Estimate residual standard deviation

# Perform quantile regression to estimate quantiles for each time step
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]  # Target quantiles
quantile_fitted_full = np.zeros((len(merged_df), len(quantile_levels)))

# Perform quantile regression for each quantile level
for i, q in enumerate(quantile_levels):
    quantile_model = sm.QuantReg(y, X).fit(q=q)
    quantile_fitted_full[:, i] = quantile_model.predict(X)

#%% Fit a Skewed t-distribution
# Define the target quantiles for interpolation
quantile_targets_t = [0.05, 0.25, 0.75, 0.95]

# Initialize arrays to store the results
T = len(merged_df)  # Assuming 'merged_df' has the time series data
tdist_mu = np.zeros(T)
tdist_sigma = np.zeros(T)
tdist_alpha = np.zeros(T)
tdist_nu = np.zeros(T)

# Loop through each time step to fit the skewed t-distribution
for t_aux in range(T):
    print(f"Iteration: {t_aux + 1}/{T}")
    quantiles_to_fit = quantile_fitted_full[t_aux, [0, 1, 3, 4]]  # Q05, Q25, Q75, Q95
    tdist_mu[t_aux], tdist_sigma[t_aux], tdist_alpha[t_aux], tdist_nu[t_aux] = quantiles_interpolation(quantiles_to_fit, quantile_targets_t)


#%% Replicating "Figure 1: Distribution of GDP Growth Over Time"  in Adrian et al. (2019)
#3D Densities Plot
support_aux = np.arange(-15, 15.1, 0.1)
# Limit Dates_forecast_num to the length of T (to avoid shape mismatch)
Dates_forecast_num = np.arange(merged_df.index[0].year + 0.25 * (merged_df.index[0].quarter - 1),
                               merged_df.index[T-1].year + 0.25 * (merged_df.index[T-1].quarter - 1) + 0.25, 0.25)

# Recreate the meshgrid based on the new Dates_forecast_num length
X_grid_aux, Y_grid_aux = np.meshgrid(Dates_forecast_num, support_aux)

# Reinitialize the linreg_Z and quantile_Z arrays to ensure correct shapes
linreg_Z = np.zeros((T, len(support_aux)))
quantile_Z = np.zeros((T, len(support_aux)))

# Loop through each time step to calculate the densities
for t_aux in range(T):
    # Compute the mean for the linear regression density using all three predictors
    mu_aux = np.dot([1, merged_df['CISS'].iloc[t_aux], merged_df['g_RGDP_original'].iloc[t_aux], merged_df['SPF_mean_forecast'].iloc[t_aux]], model.params)  # Using the linear model's parameters
    linreg_Z[t_aux, :] = norm.pdf(support_aux, loc=mu_aux, scale=linreg_sigma)
    
    # Compute the density for the quantile regression using the skewed t-distribution
    quantile_Z[t_aux, :] = dskt(support_aux, tdist_mu[t_aux], tdist_sigma[t_aux], tdist_alpha[t_aux], tdist_nu[t_aux])

# Adjust quantile_Z to match the dimensions of X_grid_aux and Y_grid_aux
quantile_Z = quantile_Z[:len(Dates_forecast_num), :]

# 3D plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid_aux, Y_grid_aux, quantile_Z.T, cmap='viridis', edgecolor='none', alpha=.8)
ax.view_init(20, -10)
ax.set_title('GDP Growth Probability Density with SPF Forecast (Mean)')
ax.set_xlabel('Time')
ax.set_ylabel(label_text)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_zlabel('Density')
plt.show()

#%% Skewed t-distribution parameters over time
# Plot Skewed t-Distribution Parameters over time
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Skewed t-Distribution Parameters Over Time', fontsize=14)

# Subplot 1: Location (mu)
axs[0, 0].plot(merged_df.index, tdist_mu, linewidth=1.25)
add_recession_shading(axs[0, 0], recession_dates)
axs[0, 0].set_title('Location (μ)')
axs[0, 0].set_xlabel('Time')

# Subplot 2: Scale (sigma)
axs[0, 1].plot(merged_df.index, tdist_sigma, linewidth=1.25)
add_recession_shading(axs[0, 1], recession_dates)
axs[0, 1].set_title('Scale (σ)')
axs[0, 1].set_xlabel('Time')

# Subplot 3: Shape (alpha)
axs[1, 0].plot(merged_df.index, tdist_alpha, linewidth=1.25)
add_recession_shading(axs[1, 0], recession_dates)
axs[1, 0].set_title('Shape (α)')
axs[1, 0].set_xlabel('Time')

# Subplot 4: Fatness (nu)
axs[1, 1].plot(merged_df.index, tdist_nu, linewidth=1.25)
add_recession_shading(axs[1, 1], recession_dates)
axs[1, 1].set_title('Fatness (ν)')
axs[1, 1].set_xlabel('Time')

# Adjust layout and display
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()



#%% Replicating "Figure 9: Growth Entropy and Expected Shortfall Over Time"  in Adrian et al. (2019)
#Expected Shortfall and Longrise (One-Quarter Ahead)
# Define quantile targets for Expected Shortfall and Longrise
quantile_targets_ES = [0.01, 0.02, 0.03, 0.04, 0.05]
quantile_targets_LR = [0.95, 0.96, 0.97, 0.98, 0.99]

# Perform quantile regression for the specified quantiles
def perform_quantile_regression(X, y, quantile_levels):
    quantile_fitted = np.zeros((len(X), len(quantile_levels)))
    for i, q in enumerate(quantile_levels):
        model = sm.QuantReg(y, X).fit(q=q)
        quantile_fitted[:, i] = model.predict(X)
    return quantile_fitted

# Add intercept to the predictor matrix
X_with_intercept = sm.add_constant(merged_df['SPF_mean_forecast'])

# Perform quantile regression for Expected Shortfall and Longrise
quantile_ES_aux = perform_quantile_regression(X_with_intercept, y, quantile_targets_ES)
quantile_LR_aux = perform_quantile_regression(X_with_intercept, y, quantile_targets_LR)

# Compute the fitted Expected Shortfall and Longrise
quantile_ES_fitted = np.mean(quantile_ES_aux, axis=1)
quantile_LR_fitted = np.mean(quantile_LR_aux, axis=1)

# Fitted Skewed t-Distribution
tdist_quantiles_ES = np.zeros((T, len(quantile_targets_ES)))
tdist_quantiles_LR = np.zeros((T, len(quantile_targets_LR)))

# Calculate the quantiles of the skewed t-distribution
for ti in range(T):
    for index, quantile_aux in enumerate(quantile_targets_ES):
        tdist_quantiles_ES[ti, index] = qskt(quantile_aux, tdist_mu[ti], tdist_sigma[ti], tdist_alpha[ti], tdist_nu[ti])
    for index, quantile_aux in enumerate(quantile_targets_LR):
        tdist_quantiles_LR[ti, index] = qskt(quantile_aux, tdist_mu[ti], tdist_sigma[ti], tdist_alpha[ti], tdist_nu[ti])

# Calculate the fitted Expected Shortfall and Longrise for the skewed t-distribution
tdist_ES_fitted = np.mean(tdist_quantiles_ES, axis=1)
tdist_LR_fitted = np.mean(tdist_quantiles_LR, axis=1)

# Plotting the Expected Shortfall and Longrise
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(merged_df.index, tdist_ES_fitted, label='Skewed t-Distribution ES', color='blue', linewidth=1.25)
ax.plot(merged_df.index, quantile_ES_fitted, linestyle='--', color='blue', linewidth=1.25, label='Quantile Regression ES')
ax.plot(merged_df.index, tdist_LR_fitted, label='Skewed t-Distribution LR', color='red', linewidth=1.25)
ax.plot(merged_df.index, quantile_LR_fitted, linestyle='--', color='red', linewidth=1.25, label='Quantile Regression LR')
# Add recession shading
add_recession_shading(ax, recession_dates)
# Adding labels, title, and legend
ax.set_title('Expected Shortfall and Longrise Over Time (using SPF Forecast Mean)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()

# Display the plot
plt.show()

#%% Replicating "Figure 8: Probability Densities"  in Adrian et al. (2019)
# Plot the densities for different dates
# Define the quarters to plot
quarters_to_plot = ['2006-04-01', '2008-10-01']

# Create an array for the support (x-axis) over which the density will be computed
support_aux = np.linspace(-15, 15, 300)  # Create a fine grid from -15 to 15

# Initialize a figure for plotting
plt.figure(figsize=(12, 8))

# Plot the densities for the specified quarters
for quarter in quarters_to_plot:
    # Find the index corresponding to the given date
    idx = merged_df.index.get_loc(quarter)
    
    # Get the parameters for the skewed t-distribution at the specified date
    mu = tdist_mu[idx]
    sigma = tdist_sigma[idx]
    alpha = tdist_alpha[idx]
    nu = tdist_nu[idx]
    
    # Compute the density using the dskt function
    density = dskt(support_aux, mu, sigma, alpha, nu)
    
    # Plot the density
    plt.plot(support_aux, density, label=f'Density for {quarter}', linewidth=1.5)

# Add labels and title
plt.title('Densities for Different Quarters (2008Q3, 2014Q3, 2020Q3)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Display the plot
plt.show()

# Define the quarters to plot
quarters_to_plot = ['2008-10-01', '2014-10-01', '2020-04-01']

# Create an array for the support (x-axis) over which the density will be computed
support_aux = np.linspace(-15, 15, 300)  # Create a fine grid from -15 to 15

# Initialize a figure for plotting
plt.figure(figsize=(12, 8))

# Plot the densities for the specified quarters
for quarter in quarters_to_plot:
    # Find the index corresponding to the given date
    idx = merged_df.index.get_loc(quarter)
    
    # Get the parameters for the skewed t-distribution at the specified date
    mu = tdist_mu[idx]
    sigma = tdist_sigma[idx]
    alpha = tdist_alpha[idx]
    nu = tdist_nu[idx]
    
    # Compute the density using the dskt function
    density = dskt(support_aux, mu, sigma, alpha, nu)
    
    # Plot the density
    plt.plot(support_aux, density, label=f'Density for {quarter}', linewidth=1.5)

# Add labels and title
plt.title('Densities for Different Quarters (2008Q3, 2014Q3, 2020Q3)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Display the plot
plt.show()


#%% Scores for the predictions (QoQA)
# Initialize arrays to store the scores
linreg_scores = np.zeros(T)
tdist_scores = np.zeros(T)

# Calculate the score for the linear regression model
for ti in range(T):
    # Compute the linear regression score using the normal distribution PDF
    mu_linreg = np.dot([1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_mean_forecast'].iloc[t_aux]], model.params)  # Predicted mean
    linreg_scores[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg, scale=linreg_sigma)

# Calculate the score for the skewed t-distribution
for ti in range(T):
    # Compute the skewed t-distribution score using the dskt function
    tdist_scores[ti] = dskt(y.iloc[ti], tdist_mu[ti], tdist_sigma[ti], tdist_alpha[ti], tdist_nu[ti])

#%% Plot the scores
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(merged_df.index, linreg_scores, label='Linear Regression', linewidth=1.25)
ax.plot(merged_df.index, tdist_scores, label='Skewed t-Distribution', linewidth=1.25)

# Add recession shading to the plot
add_recession_shading(ax, recession_dates)

# Setting plot title, labels, and legend
ax.set_title('Score of the Predictions Over Time - CISS + GDP + SPF Forecast (Mean) (QoQA) ')
ax.set_ylabel('Score f(y)')
ax.set_xlabel('Time')
ax.set_ylim([-0.05, 0.35])
ax.legend()

# Display the plot
plt.show()


#%% Forecast Comparisons

# Predictor sets
X_GDP = pd.DataFrame({'Intercept': 1, 'GDP': merged_df['g_RGDP_original']})
X_CISS = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS']})
X_CISS_GDP = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'GDP': merged_df['g_RGDP_original']})
X_GDP_SPF = pd.DataFrame({'Intercept': 1, 'GDP': merged_df['g_RGDP_original'], 'SPF_mean_forecast': merged_df['SPF_mean_forecast']})
X_CISS_SPF = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'SPF_mean_forecast': merged_df['SPF_mean_forecast']})
X_CISS_GDP_SPF = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'GDP': merged_df['g_RGDP_original'], 'SPF_mean_forecast': merged_df['SPF_mean_forecast']})

# Drop NaNs to align with y
X_GDP.dropna(inplace=True)
X_CISS.dropna(inplace=True)
X_CISS_GDP.dropna(inplace=True)
X_GDP_SPF.dropna(inplace=True)
X_CISS_SPF.dropna(inplace=True)
X_CISS_GDP_SPF.dropna(inplace=True)

y_GDP = y.loc[y.index.isin(X_GDP.index)]
y_CISS = y.loc[y.index.isin(X_CISS.index)]
y_CISS_GDP = y.loc[y.index.isin(X_CISS_GDP.index)]
y_GDP_SPF = y.loc[y.index.isin(X_GDP_SPF.index)]
y_CISS_SPF = y.loc[y.index.isin(X_CISS_SPF.index)]
y_CISS_GDP_SPF = y.loc[y.index.isin(X_CISS_GDP_SPF.index)]

# Perform linear regression

# Linear regression models
model_GDP = sm.OLS(y_GDP, X_GDP).fit()
linreg_sigma_GDP = np.std(model_GDP.resid)

model_CISS = sm.OLS(y_CISS, X_CISS).fit()
linreg_sigma_CISS = np.std(model_CISS.resid)

model_CISS_GDP = sm.OLS(y_CISS_GDP, X_CISS_GDP).fit()
linreg_sigma_CISS_GDP = np.std(model_CISS_GDP.resid)

model_GDP_SPF = sm.OLS(y_GDP_SPF, X_GDP_SPF).fit()
linreg_sigma_GDP_SPF = np.std(model_GDP_SPF.resid)

model_CISS_SPF = sm.OLS(y_CISS_SPF, X_CISS_SPF).fit()
linreg_sigma_CISS_SPF = np.std(model_CISS_SPF.resid)

model_CISS_GDP_SPF = sm.OLS(y_CISS_GDP_SPF, X_CISS_GDP_SPF).fit()
linreg_sigma_CISS_GDP_SPF = np.std(model_CISS_GDP_SPF.resid)

# Calculate the scores

# Initialize arrays for linear regression scores
T = len(merged_df)
linreg_scores_GDP = np.zeros(T)
linreg_scores_CISS = np.zeros(T)
linreg_scores_CISS_GDP = np.zeros(T)
linreg_scores_GDP_SPF = np.zeros(T)
linreg_scores_CISS_SPF = np.zeros(T)
linreg_scores_CISS_GDP_SPF = np.zeros(T)

# Calculate scores for each time point
for ti in range(T):
    if ti < len(X_GDP):
        predictors_GDP = [1, X_GDP['GDP'].iloc[ti]]
        mu_linreg_GDP = np.dot(predictors_GDP, model_GDP.params)
        linreg_scores_GDP[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_GDP, scale=linreg_sigma_GDP)

    if ti < len(X_CISS):
        predictors_CISS = [1, X_CISS['CISS'].iloc[ti]]
        mu_linreg_CISS = np.dot(predictors_CISS, model_CISS.params)
        linreg_scores_CISS[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS, scale=linreg_sigma_CISS)
    
    if ti < len(X_CISS_GDP):
        predictors_CISS_GDP = [1, X_CISS_GDP['CISS'].iloc[ti], X_CISS_GDP['GDP'].iloc[ti]]
        mu_linreg_CISS_GDP = np.dot(predictors_CISS_GDP, model_CISS_GDP.params)
        linreg_scores_CISS_GDP[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS_GDP, scale=linreg_sigma_CISS_GDP)
    
    if ti < len(X_GDP_SPF):
        predictors_GDP_SPF = [1, X_GDP_SPF['GDP'].iloc[ti], X_GDP_SPF['SPF_mean_forecast'].iloc[ti]]
        mu_linreg_GDP_SPF = np.dot(predictors_GDP_SPF, model_GDP_SPF.params)
        linreg_scores_GDP_SPF[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_GDP_SPF, scale=linreg_sigma_GDP_SPF)
    
    if ti < len(X_CISS_SPF):
        predictors_CISS_SPF = [1, X_CISS_SPF['CISS'].iloc[ti], X_CISS_SPF['SPF_mean_forecast'].iloc[ti]]
        mu_linreg_CISS_SPF = np.dot(predictors_CISS_SPF, model_CISS_SPF.params)
        linreg_scores_CISS_SPF[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS_SPF, scale=linreg_sigma_CISS_SPF)
    
    if ti < len(X_CISS_GDP_SPF):
        predictors_CISS_GDP_SPF = [1, X_CISS_GDP_SPF['CISS'].iloc[ti], X_CISS_GDP_SPF['GDP'].iloc[ti], X_CISS_GDP_SPF['SPF_mean_forecast'].iloc[ti]]
        mu_linreg_CISS_GDP_SPF = np.dot(predictors_CISS_GDP_SPF, model_CISS_GDP_SPF.params)
        linreg_scores_CISS_GDP_SPF[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS_GDP_SPF, scale=linreg_sigma_CISS_GDP_SPF)

#%% Plot the comparison of scores
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(merged_df.index, linreg_scores_GDP, label='GDP only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS, label='CISS only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP, label='CISS + GDP', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_GDP_SPF, label='GDP + SPF_mean_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_SPF, label='CISS + SPF_mean_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP_SPF, label='CISS + GDP + SPF_mean_forecast', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Linear Regression Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()




#%% Plot the comparison of scores (SPF compared to GDP and CISS only)
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(merged_df.index, linreg_scores_GDP, label='GDP only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS, label='CISS only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP, label='CISS + GDP', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP_SPF, label='CISS + GDP + SPF_mean_forecast', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Linear Regression Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()



#%% Plot the comparison of scores (SPF only)
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(merged_df.index, linreg_scores_GDP_SPF, label='GDP + SPF_mean_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_SPF, label='CISS + SPF_mean_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP_SPF, label='CISS + GDP + SPF_mean_forecast', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Linear Regression Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()



#%% Calculate the scores for the Skewed t-distribution with different predictors

# Define the target quantiles for interpolation
quantile_targets_t = [0.05, 0.25, 0.75, 0.95]

# Predictor sets
predictor_sets = {
    'GDP only': X_GDP,
    'CISS only': X_CISS,
    'CISS + GDP': X_CISS_GDP,
    'GDP + SPF_mean_forecast': X_GDP_SPF,
    'CISS + SPF_mean_forecast': X_CISS_SPF,
    'CISS + GDP + SPF_mean_forecast': X_CISS_GDP_SPF
}

# Initialize arrays to store results for each predictor set
T = len(merged_df)
tdist_params = {key: {'mu': np.zeros(T), 'sigma': np.zeros(T), 'alpha': np.zeros(T), 'nu': np.zeros(T)} for key in predictor_sets}
tdist_scores = {key: np.zeros(T) for key in predictor_sets}

# Loop through each predictor set
for key, X_set in predictor_sets.items():
    print(f"Fitting skewed t-distribution for predictor set: {key}")
    
    # Align y to X_set
    y_set = y.loc[y.index.isin(X_set.index)]
    
    # Quantile regression results (change accordingly for each set if needed)
    quantile_fitted_set = np.zeros((len(X_set), len(quantile_targets_t)))
    for i, q in enumerate(quantile_targets_t):
        quantile_model = sm.QuantReg(y_set, X_set).fit(q=q)
        quantile_fitted_set[:, i] = quantile_model.predict(X_set)
    
    # Fit skewed t-distribution parameters for each time step
    for t_aux in range(T):
        if t_aux < len(X_set):
            quantiles_to_fit = quantile_fitted_set[t_aux, [0, 1, 2, 3]]  # Q05, Q25, Q75, Q95
            tdist_params[key]['mu'][t_aux], tdist_params[key]['sigma'][t_aux], tdist_params[key]['alpha'][t_aux], tdist_params[key]['nu'][t_aux] = quantiles_interpolation(quantiles_to_fit, quantile_targets_t)
    
    # Calculate Skewed t-distribution scores for each time point
    for ti in range(T):
        if ti < len(X_set):
            mu = tdist_params[key]['mu'][ti]
            sigma = tdist_params[key]['sigma'][ti]
            alpha = tdist_params[key]['alpha'][ti]
            nu = tdist_params[key]['nu'][ti]
            tdist_scores[key][ti] = dskt(y.iloc[ti], mu, sigma, alpha, nu)

# Plot the comparison of Skewed t-distribution scores for different predictor sets
fig, ax = plt.subplots(figsize=(12, 8))

for key in predictor_sets.keys():
    ax.plot(merged_df.index, tdist_scores[key], label=f'Skewed t-Distribution with {key}', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Skewed t-Distribution Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()




#### SPF Forecast (Median) ####

#%% Perform linear regression to get the model parameters (particularly for SPF_mean)
X = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'GDP': merged_df['g_RGDP_original'], 'SPF (Median)': merged_df['SPF_median_forecast']})   # Predictor variables (including intercept)
y = merged_df[ycol]  # Response variable
model = sm.OLS(y, X).fit()  # Fit the linear regression model
linreg_sigma = np.std(model.resid)  # Estimate residual standard deviation

# Perform quantile regression to estimate quantiles for each time step
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]  # Target quantiles
quantile_fitted_full = np.zeros((len(merged_df), len(quantile_levels)))

# Perform quantile regression for each quantile level
for i, q in enumerate(quantile_levels):
    quantile_model = sm.QuantReg(y, X).fit(q=q)
    quantile_fitted_full[:, i] = quantile_model.predict(X)

#%% Fit a Skewed t-distribution
# Define the target quantiles for interpolation
quantile_targets_t = [0.05, 0.25, 0.75, 0.95]

# Initialize arrays to store the results
T = len(merged_df)  # Assuming 'merged_df' has the time series data
tdist_mu = np.zeros(T)
tdist_sigma = np.zeros(T)
tdist_alpha = np.zeros(T)
tdist_nu = np.zeros(T)

# Loop through each time step to fit the skewed t-distribution
for t_aux in range(T):
    print(f"Iteration: {t_aux + 1}/{T}")
    quantiles_to_fit = quantile_fitted_full[t_aux, [0, 1, 3, 4]]  # Q05, Q25, Q75, Q95
    tdist_mu[t_aux], tdist_sigma[t_aux], tdist_alpha[t_aux], tdist_nu[t_aux] = quantiles_interpolation(quantiles_to_fit, quantile_targets_t)


#%% Replicating "Figure 1: Distribution of GDP Growth Over Time"  in Adrian et al. (2019)
#3D Densities Plot
support_aux = np.arange(-15, 15.1, 0.1)
# Limit Dates_forecast_num to the length of T (to avoid shape mismatch)
Dates_forecast_num = np.arange(merged_df.index[0].year + 0.25 * (merged_df.index[0].quarter - 1),
                               merged_df.index[T-1].year + 0.25 * (merged_df.index[T-1].quarter - 1) + 0.25, 0.25)

# Recreate the meshgrid based on the new Dates_forecast_num length
X_grid_aux, Y_grid_aux = np.meshgrid(Dates_forecast_num, support_aux)

# Reinitialize the linreg_Z and quantile_Z arrays to ensure correct shapes
linreg_Z = np.zeros((T, len(support_aux)))
quantile_Z = np.zeros((T, len(support_aux)))

# Loop through each time step to calculate the densities
for t_aux in range(T):
    # Compute the mean for the linear regression density using all three predictors
    mu_aux = np.dot([1, merged_df['CISS'].iloc[t_aux], merged_df['g_RGDP_original'].iloc[t_aux], merged_df['SPF_median_forecast'].iloc[t_aux]], model.params)  # Using the linear model's parameters
    linreg_Z[t_aux, :] = norm.pdf(support_aux, loc=mu_aux, scale=linreg_sigma)
    
    # Compute the density for the quantile regression using the skewed t-distribution
    quantile_Z[t_aux, :] = dskt(support_aux, tdist_mu[t_aux], tdist_sigma[t_aux], tdist_alpha[t_aux], tdist_nu[t_aux])

# Adjust quantile_Z to match the dimensions of X_grid_aux and Y_grid_aux
quantile_Z = quantile_Z[:len(Dates_forecast_num), :]

# 3D plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid_aux, Y_grid_aux, quantile_Z.T, cmap='viridis', edgecolor='none', alpha=.8)
ax.view_init(20, -10)
ax.set_title('GDP Growth Probability Density with SPF Forecast (Median)')
ax.set_xlabel('Time')
ax.set_ylabel(label_text)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_zlabel('Density')
plt.show()

#%% Skewed t-distribution parameters over time
# Plot Skewed t-Distribution Parameters over time
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Skewed t-Distribution Parameters Over Time', fontsize=14)

# Subplot 1: Location (mu)
axs[0, 0].plot(merged_df.index, tdist_mu, linewidth=1.25)
add_recession_shading(axs[0, 0], recession_dates)
axs[0, 0].set_title('Location (μ)')
axs[0, 0].set_xlabel('Time')

# Subplot 2: Scale (sigma)
axs[0, 1].plot(merged_df.index, tdist_sigma, linewidth=1.25)
add_recession_shading(axs[0, 1], recession_dates)
axs[0, 1].set_title('Scale (σ)')
axs[0, 1].set_xlabel('Time')

# Subplot 3: Shape (alpha)
axs[1, 0].plot(merged_df.index, tdist_alpha, linewidth=1.25)
add_recession_shading(axs[1, 0], recession_dates)
axs[1, 0].set_title('Shape (α)')
axs[1, 0].set_xlabel('Time')

# Subplot 4: Fatness (nu)
axs[1, 1].plot(merged_df.index, tdist_nu, linewidth=1.25)
add_recession_shading(axs[1, 1], recession_dates)
axs[1, 1].set_title('Fatness (ν)')
axs[1, 1].set_xlabel('Time')

# Adjust layout and display
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()



#%% Replicating "Figure 9: Growth Entropy and Expected Shortfall Over Time"  in Adrian et al. (2019)
#Expected Shortfall and Longrise (One-Quarter Ahead)
# Define quantile targets for Expected Shortfall and Longrise
quantile_targets_ES = [0.01, 0.02, 0.03, 0.04, 0.05]
quantile_targets_LR = [0.95, 0.96, 0.97, 0.98, 0.99]

# Perform quantile regression for the specified quantiles
def perform_quantile_regression(X, y, quantile_levels):
    quantile_fitted = np.zeros((len(X), len(quantile_levels)))
    for i, q in enumerate(quantile_levels):
        model = sm.QuantReg(y, X).fit(q=q)
        quantile_fitted[:, i] = model.predict(X)
    return quantile_fitted

# Add intercept to the predictor matrix
X_with_intercept = sm.add_constant(merged_df['SPF_median_forecast'])

# Perform quantile regression for Expected Shortfall and Longrise
quantile_ES_aux = perform_quantile_regression(X_with_intercept, y, quantile_targets_ES)
quantile_LR_aux = perform_quantile_regression(X_with_intercept, y, quantile_targets_LR)

# Compute the fitted Expected Shortfall and Longrise
quantile_ES_fitted = np.mean(quantile_ES_aux, axis=1)
quantile_LR_fitted = np.mean(quantile_LR_aux, axis=1)

# Fitted Skewed t-Distribution
tdist_quantiles_ES = np.zeros((T, len(quantile_targets_ES)))
tdist_quantiles_LR = np.zeros((T, len(quantile_targets_LR)))

# Calculate the quantiles of the skewed t-distribution
for ti in range(T):
    for index, quantile_aux in enumerate(quantile_targets_ES):
        tdist_quantiles_ES[ti, index] = qskt(quantile_aux, tdist_mu[ti], tdist_sigma[ti], tdist_alpha[ti], tdist_nu[ti])
    for index, quantile_aux in enumerate(quantile_targets_LR):
        tdist_quantiles_LR[ti, index] = qskt(quantile_aux, tdist_mu[ti], tdist_sigma[ti], tdist_alpha[ti], tdist_nu[ti])

# Calculate the fitted Expected Shortfall and Longrise for the skewed t-distribution
tdist_ES_fitted = np.mean(tdist_quantiles_ES, axis=1)
tdist_LR_fitted = np.mean(tdist_quantiles_LR, axis=1)

# Plotting the Expected Shortfall and Longrise
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(merged_df.index, tdist_ES_fitted, label='Skewed t-Distribution ES', color='blue', linewidth=1.25)
ax.plot(merged_df.index, quantile_ES_fitted, linestyle='--', color='blue', linewidth=1.25, label='Quantile Regression ES')
ax.plot(merged_df.index, tdist_LR_fitted, label='Skewed t-Distribution LR', color='red', linewidth=1.25)
ax.plot(merged_df.index, quantile_LR_fitted, linestyle='--', color='red', linewidth=1.25, label='Quantile Regression LR')
# Add recession shading
add_recession_shading(ax, recession_dates)
# Adding labels, title, and legend
ax.set_title('Expected Shortfall and Longrise Over Time (using SPF Forecast Median)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()

# Display the plot
plt.show()

#%% Replicating "Figure 8: Probability Densities"  in Adrian et al. (2019)
# Plot the densities for different dates
# Define the quarters to plot
quarters_to_plot = ['2006-04-01', '2008-10-01']

# Create an array for the support (x-axis) over which the density will be computed
support_aux = np.linspace(-15, 15, 300)  # Create a fine grid from -15 to 15

# Initialize a figure for plotting
plt.figure(figsize=(12, 8))

# Plot the densities for the specified quarters
for quarter in quarters_to_plot:
    # Find the index corresponding to the given date
    idx = merged_df.index.get_loc(quarter)
    
    # Get the parameters for the skewed t-distribution at the specified date
    mu = tdist_mu[idx]
    sigma = tdist_sigma[idx]
    alpha = tdist_alpha[idx]
    nu = tdist_nu[idx]
    
    # Compute the density using the dskt function
    density = dskt(support_aux, mu, sigma, alpha, nu)
    
    # Plot the density
    plt.plot(support_aux, density, label=f'Density for {quarter}', linewidth=1.5)

# Add labels and title
plt.title('Densities for Different Quarters (2008Q3, 2014Q3, 2020Q3)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Display the plot
plt.show()

# Define the quarters to plot
quarters_to_plot = ['2008-10-01', '2014-10-01', '2020-04-01']

# Create an array for the support (x-axis) over which the density will be computed
support_aux = np.linspace(-15, 15, 300)  # Create a fine grid from -15 to 15

# Initialize a figure for plotting
plt.figure(figsize=(12, 8))

# Plot the densities for the specified quarters
for quarter in quarters_to_plot:
    # Find the index corresponding to the given date
    idx = merged_df.index.get_loc(quarter)
    
    # Get the parameters for the skewed t-distribution at the specified date
    mu = tdist_mu[idx]
    sigma = tdist_sigma[idx]
    alpha = tdist_alpha[idx]
    nu = tdist_nu[idx]
    
    # Compute the density using the dskt function
    density = dskt(support_aux, mu, sigma, alpha, nu)
    
    # Plot the density
    plt.plot(support_aux, density, label=f'Density for {quarter}', linewidth=1.5)

# Add labels and title
plt.title('Densities for Different Quarters (2008Q3, 2014Q3, 2020Q3)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Display the plot
plt.show()


#%% Scores for the predictions (QoQA)
# Initialize arrays to store the scores
linreg_scores = np.zeros(T)
tdist_scores = np.zeros(T)

# Calculate the score for the linear regression model
for ti in range(T):
    # Compute the linear regression score using the normal distribution PDF
    mu_linreg = np.dot([1, merged_df['CISS'].iloc[ti], merged_df['g_RGDP_original'].iloc[ti], merged_df['SPF_median_forecast'].iloc[t_aux]], model.params)  # Predicted mean
    linreg_scores[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg, scale=linreg_sigma)

# Calculate the score for the skewed t-distribution
for ti in range(T):
    # Compute the skewed t-distribution score using the dskt function
    tdist_scores[ti] = dskt(y.iloc[ti], tdist_mu[ti], tdist_sigma[ti], tdist_alpha[ti], tdist_nu[ti])

#%% Plot the scores
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(merged_df.index, linreg_scores, label='Linear Regression', linewidth=1.25)
ax.plot(merged_df.index, tdist_scores, label='Skewed t-Distribution', linewidth=1.25)

# Add recession shading to the plot
add_recession_shading(ax, recession_dates)

# Setting plot title, labels, and legend
ax.set_title('Score of the Predictions Over Time - CISS + GDP + SPF Forecast (Median) (QoQA) ')
ax.set_ylabel('Score f(y)')
ax.set_xlabel('Time')
ax.set_ylim([-0.05, 0.35])
ax.legend()

# Display the plot
plt.show()


#%% Forecast Comparisons

# Predictor sets
X_GDP = pd.DataFrame({'Intercept': 1, 'GDP': merged_df['g_RGDP_original']})
X_CISS = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS']})
X_CISS_GDP = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'GDP': merged_df['g_RGDP_original']})
X_GDP_SPF = pd.DataFrame({'Intercept': 1, 'GDP': merged_df['g_RGDP_original'], 'SPF_median_forecast': merged_df['SPF_median_forecast']})
X_CISS_SPF = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'SPF_median_forecast': merged_df['SPF_median_forecast']})
X_CISS_GDP_SPF = pd.DataFrame({'Intercept': 1, 'CISS': merged_df['CISS'], 'GDP': merged_df['g_RGDP_original'], 'SPF_median_forecast': merged_df['SPF_median_forecast']})

# Drop NaNs to align with y
X_GDP.dropna(inplace=True)
X_CISS.dropna(inplace=True)
X_CISS_GDP.dropna(inplace=True)
X_GDP_SPF.dropna(inplace=True)
X_CISS_SPF.dropna(inplace=True)
X_CISS_GDP_SPF.dropna(inplace=True)

y_GDP = y.loc[y.index.isin(X_GDP.index)]
y_CISS = y.loc[y.index.isin(X_CISS.index)]
y_CISS_GDP = y.loc[y.index.isin(X_CISS_GDP.index)]
y_GDP_SPF = y.loc[y.index.isin(X_GDP_SPF.index)]
y_CISS_SPF = y.loc[y.index.isin(X_CISS_SPF.index)]
y_CISS_GDP_SPF = y.loc[y.index.isin(X_CISS_GDP_SPF.index)]

# Perform linear regression

# Linear regression models
model_GDP = sm.OLS(y_GDP, X_GDP).fit()
linreg_sigma_GDP = np.std(model_GDP.resid)

model_CISS = sm.OLS(y_CISS, X_CISS).fit()
linreg_sigma_CISS = np.std(model_CISS.resid)

model_CISS_GDP = sm.OLS(y_CISS_GDP, X_CISS_GDP).fit()
linreg_sigma_CISS_GDP = np.std(model_CISS_GDP.resid)

model_GDP_SPF = sm.OLS(y_GDP_SPF, X_GDP_SPF).fit()
linreg_sigma_GDP_SPF = np.std(model_GDP_SPF.resid)

model_CISS_SPF = sm.OLS(y_CISS_SPF, X_CISS_SPF).fit()
linreg_sigma_CISS_SPF = np.std(model_CISS_SPF.resid)

model_CISS_GDP_SPF = sm.OLS(y_CISS_GDP_SPF, X_CISS_GDP_SPF).fit()
linreg_sigma_CISS_GDP_SPF = np.std(model_CISS_GDP_SPF.resid)

# Calculate the scores

# Initialize arrays for linear regression scores
T = len(merged_df)
linreg_scores_GDP = np.zeros(T)
linreg_scores_CISS = np.zeros(T)
linreg_scores_CISS_GDP = np.zeros(T)
linreg_scores_GDP_SPF = np.zeros(T)
linreg_scores_CISS_SPF = np.zeros(T)
linreg_scores_CISS_GDP_SPF = np.zeros(T)

# Calculate scores for each time point
for ti in range(T):
    if ti < len(X_GDP):
        predictors_GDP = [1, X_GDP['GDP'].iloc[ti]]
        mu_linreg_GDP = np.dot(predictors_GDP, model_GDP.params)
        linreg_scores_GDP[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_GDP, scale=linreg_sigma_GDP)

    if ti < len(X_CISS):
        predictors_CISS = [1, X_CISS['CISS'].iloc[ti]]
        mu_linreg_CISS = np.dot(predictors_CISS, model_CISS.params)
        linreg_scores_CISS[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS, scale=linreg_sigma_CISS)
    
    if ti < len(X_CISS_GDP):
        predictors_CISS_GDP = [1, X_CISS_GDP['CISS'].iloc[ti], X_CISS_GDP['GDP'].iloc[ti]]
        mu_linreg_CISS_GDP = np.dot(predictors_CISS_GDP, model_CISS_GDP.params)
        linreg_scores_CISS_GDP[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS_GDP, scale=linreg_sigma_CISS_GDP)
    
    if ti < len(X_GDP_SPF):
        predictors_GDP_SPF = [1, X_GDP_SPF['GDP'].iloc[ti], X_GDP_SPF['SPF_median_forecast'].iloc[ti]]
        mu_linreg_GDP_SPF = np.dot(predictors_GDP_SPF, model_GDP_SPF.params)
        linreg_scores_GDP_SPF[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_GDP_SPF, scale=linreg_sigma_GDP_SPF)
    
    if ti < len(X_CISS_SPF):
        predictors_CISS_SPF = [1, X_CISS_SPF['CISS'].iloc[ti], X_CISS_SPF['SPF_median_forecast'].iloc[ti]]
        mu_linreg_CISS_SPF = np.dot(predictors_CISS_SPF, model_CISS_SPF.params)
        linreg_scores_CISS_SPF[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS_SPF, scale=linreg_sigma_CISS_SPF)
    
    if ti < len(X_CISS_GDP_SPF):
        predictors_CISS_GDP_SPF = [1, X_CISS_GDP_SPF['CISS'].iloc[ti], X_CISS_GDP_SPF['GDP'].iloc[ti], X_CISS_GDP_SPF['SPF_median_forecast'].iloc[ti]]
        mu_linreg_CISS_GDP_SPF = np.dot(predictors_CISS_GDP_SPF, model_CISS_GDP_SPF.params)
        linreg_scores_CISS_GDP_SPF[ti] = norm.pdf(y.iloc[ti], loc=mu_linreg_CISS_GDP_SPF, scale=linreg_sigma_CISS_GDP_SPF)

#%% Plot the comparison of scores
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(merged_df.index, linreg_scores_GDP, label='GDP only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS, label='CISS only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP, label='CISS + GDP', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_GDP_SPF, label='GDP + SPF_median_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_SPF, label='CISS + SPF_median_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP_SPF, label='CISS + GDP + SPF_median_forecast', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Linear Regression Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()



#%% Plot the comparison of scores (SPF compared to GDP and CISS only)
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(merged_df.index, linreg_scores_GDP, label='GDP only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS, label='CISS only', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP, label='CISS + GDP', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP_SPF, label='CISS + GDP + SPF_median_forecast', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Linear Regression Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()



#%% Plot the comparison of scores (SPF only)
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(merged_df.index, linreg_scores_GDP_SPF, label='GDP + SPF_median_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_SPF, label='CISS + SPF_median_forecast', linewidth=1.25)
ax.plot(merged_df.index, linreg_scores_CISS_GDP_SPF, label='CISS + GDP + SPF_median_forecast', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Linear Regression Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()



#%% Calculate the scores for the Skewed t-distribution with different predictors

# Define the target quantiles for interpolation
quantile_targets_t = [0.05, 0.25, 0.75, 0.95]

# Predictor sets
predictor_sets = {
    'GDP only': X_GDP,
    'CISS only': X_CISS,
    'CISS + GDP': X_CISS_GDP,
    'GDP + SPF_median_forecast': X_GDP_SPF,
    'CISS + SPF_median_forecast': X_CISS_SPF,
    'CISS + GDP + SPF_median_forecast': X_CISS_GDP_SPF
}

# Initialize arrays to store results for each predictor set
T = len(merged_df)
tdist_params = {key: {'mu': np.zeros(T), 'sigma': np.zeros(T), 'alpha': np.zeros(T), 'nu': np.zeros(T)} for key in predictor_sets}
tdist_scores = {key: np.zeros(T) for key in predictor_sets}

# Loop through each predictor set
for key, X_set in predictor_sets.items():
    print(f"Fitting skewed t-distribution for predictor set: {key}")
    
    # Align y to X_set
    y_set = y.loc[y.index.isin(X_set.index)]
    
    # Quantile regression results (change accordingly for each set if needed)
    quantile_fitted_set = np.zeros((len(X_set), len(quantile_targets_t)))
    for i, q in enumerate(quantile_targets_t):
        quantile_model = sm.QuantReg(y_set, X_set).fit(q=q)
        quantile_fitted_set[:, i] = quantile_model.predict(X_set)
    
    # Fit skewed t-distribution parameters for each time step
    for t_aux in range(T):
        if t_aux < len(X_set):
            quantiles_to_fit = quantile_fitted_set[t_aux, [0, 1, 2, 3]]  # Q05, Q25, Q75, Q95
            tdist_params[key]['mu'][t_aux], tdist_params[key]['sigma'][t_aux], tdist_params[key]['alpha'][t_aux], tdist_params[key]['nu'][t_aux] = quantiles_interpolation(quantiles_to_fit, quantile_targets_t)
    
    # Calculate Skewed t-distribution scores for each time point
    for ti in range(T):
        if ti < len(X_set):
            mu = tdist_params[key]['mu'][ti]
            sigma = tdist_params[key]['sigma'][ti]
            alpha = tdist_params[key]['alpha'][ti]
            nu = tdist_params[key]['nu'][ti]
            tdist_scores[key][ti] = dskt(y.iloc[ti], mu, sigma, alpha, nu)

# Plot the comparison of Skewed t-distribution scores for different predictor sets
fig, ax = plt.subplots(figsize=(12, 8))

for key in predictor_sets.keys():
    ax.plot(merged_df.index, tdist_scores[key], label=f'Skewed t-Distribution with {key}', linewidth=1.25)

# Add recession shading
add_recession_shading(ax, recession_dates)

# Set title, labels, and legend
ax.set_title('Comparison of Skewed t-Distribution Scores for Different Predictor Variables')
ax.set_xlabel('Time')
ax.set_ylabel('Score f(y)')
ax.legend()

# Show the plot
plt.show()




