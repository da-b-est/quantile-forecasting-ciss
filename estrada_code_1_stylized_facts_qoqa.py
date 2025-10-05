#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:01:30 2024

@author: dianniadreiestrada
"""

#Replication of: "Vulnerable Growth" Adrian et al. (2019)

#Part 1: Stylized Facts

# Importing necessary libraries
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import DateFormatter
import os

#We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise')


# Load the data
df = pd.read_excel('Data/Data_Adrian_2019.xlsx', sheet_name='EA_Quarterly')
# import data from an Excel file
# refer to sheet named 'EA_Quarterly'

df.index = pd.to_datetime(df.Date)  
# Convert Date to DatetimeIndex for easy sorting and manipulation of time-series data

df.sort_index(inplace=True)
#Ensures chronological order of observations

df['Intercept'] = 1
# Add a constant (intercept) directly to df as a new column


df['current_gRGDP'] = df['g_RGDP']

# Define which column is the dependent variable (based on h)
h = 1  # GDP Growth 1 Quarter Ahead (to be changed h = 4 for YoY)
# h is time horizon for the GDP growth analysis.

if h == 1:
    ycol = 'g_RGDP'
    label_text = "GDP Growth 1 Quarter Ahead (QoQA)"
elif h == 4:
    ycol = 'g_RGDPyoy'
    label_text = "GDP Growth 4 Quarters Ahead (YoY)"
#sets the dependent variable (ycol) and its corresponding label (label_text) based on the chosen horizon


#Prediction: shift GDP growth to see if we can predict it
df[ycol] = df[ycol].shift(-h)
# Replace the dependent variable with its shifted values

df.dropna(subset=['CISS', ycol], inplace=True)
# Drop rows with NaN values in the 'CISS' or dependent variable
#Note: The CISS is indicator of contemporaneous stress in the financial system 
#which can be used as a measure for EU financial conditions(See Holló et al, 2012)


# Function to get recession dates from the dataset
def get_recession_dates(df, recession_indicator_column):
    """
    Identify recession start and end dates based on the recession indicator column.
    """
    start_dates = []
    end_dates = []
    T = len(df)

    for t in range(T):
        date_aux = df.index[t]  # Assuming the DataFrame index is the date
        x_aux = df[recession_indicator_column].iloc[t]

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


df['Recession'] = ((df[ycol] < 0) & (df[ycol].shift(1) < 0)).astype(int)
# Recession defined as two consecutive quarters of negative GDP growth

recession_dates = get_recession_dates(df, 'Recession')
# Get the recession dates from df based on the definition of recession using the function "get_recession_dates"


#%% Replicating 'Figure 2: Raw Data' in Adrian et al. (2019)
#Plot the GDP growth and the CISS
# Function to add recession shading to plots
def add_recession_shading(ax, recession_dates, color='gray', alpha=0.3):
    """
    Add shaded areas to represent recession periods on a given axis.
    """
    for start_date, end_date in recession_dates:
        ax.axvspan(start_date, end_date, color=color, alpha=alpha)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans serif',
})

plt.figure(figsize=(12, 5))
# Plotting GDP Growth (ycol) on the primary y-axis
ax = df[ycol].plot(label='GDP', linewidth=1, color='b')
ax.set_ylim([-40, 60])
ax.tick_params(axis='y', colors='b')  
ax.set_ylabel('Real GDP Growth (annualized)', color='b')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # Format y-axis as percentage
# Add recession shading
add_recession_shading(ax, recession_dates)
# Plotting CISS on the secondary y-axis
ax2 = ax.twinx()
df['CISS'].plot(ax=ax2, label='CISS', linewidth=1, color='r', linestyle='--')
ax2.set_ylabel('CISS (indicator)', color='r')  
ax2.tick_params(axis='y', colors='r')  
ax2.set_ylim([-1, 5])
# Setting up the legend and title
plt.title('GDP Growth vs Financial Conditions')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
#Display the plot
plt.show()


#%% Scatter plot setup
yaxis_bounds_aux = [-20, 20]  # Define y-axis bounds
plt.scatter(df.CISS, df[ycol], c='b', label='Data 1973-2024')
# Labels and title
plt.xlabel('Financial Conditions')
plt.ylabel(label_text)
plt.ylim(yaxis_bounds_aux)
plt.title('Data')
# Format y-axis as percentage (if needed)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
# Add legend
plt.legend(loc='best', fontsize=8)
# Display the plot
plt.show()

# Plot CISS and shifted GDP growth on two different y-axes
df.loc[:, ['CISS', ycol]].plot(secondary_y='CISS')
plt.title('CISS'' vs GDP Growth')
plt.savefig('Figures/Adrian_2019_Figure_2')
plt.show()


#%% Linear regression
# Prepare the matrix (CISS column for X, and ycol for y)
X = df['CISS']
y = df[ycol]

# Add a constant (intercept term) to the predictor matrix
X_with_intercept = sm.add_constant(X)

# Fit the linear regression model using statsmodels
model = sm.OLS(y, X_with_intercept).fit()

# Get the fitted values (predictions)
predictions = model.predict(X_with_intercept)

# Scatter plot setup
plt.figure(figsize=(10, 6))

# Scatter plot of CISS vs GDP Growth
plt.scatter(X, y, c='b', label='Data 1973-2024')
# Plot the regression line
plt.plot(X, predictions, 'r', label='Mean E[Y|X]', linewidth=2)
# Labels and title
plt.xlabel('Financial Conditions')
plt.ylabel(label_text)
plt.ylim([-10, 10])  # Replace with your specific y-axis bounds if needed
plt.title('Linear Regression: GDP Growth vs Financial Conditions')
# Add legend
plt.legend(loc='best', fontsize=8)
# Display the plot
plt.show()


# Time Series Plot with Confidence Bands
# Generate forecast dates
Dates_forecast_full = df.index
linreg_fitted_full = np.zeros((len(Dates_forecast_full), 5))

# Simulate confidence bands (Replace with actual prediction intervals if available)
linreg_fitted_full[:, 2] = predictions  # Median (Q50)
linreg_sigma = np.std(model.resid)  # Estimate residual standard deviation
linreg_fitted_full[:, 0] = predictions - 1.96 * linreg_sigma  # Q05
linreg_fitted_full[:, 4] = predictions + 1.96 * linreg_sigma  # Q95
linreg_fitted_full[:, 1] = predictions - 1.28 * linreg_sigma  # Q25
linreg_fitted_full[:, 3] = predictions + 1.28 * linreg_sigma  # Q75

# Assume y_full is the original target variable for plotting
y_full = df[ycol]

# Create the figure
plt.figure(figsize=(12, 6))

# Step 1: Plot the confidence bands
plt.fill_between(Dates_forecast_full, linreg_fitted_full[:, 0], linreg_fitted_full[:, 4], color='black', alpha=0.35, label='Q05-Q95')
plt.fill_between(Dates_forecast_full, linreg_fitted_full[:, 1], linreg_fitted_full[:, 3], color='black', alpha=0.4, label='Q25-Q75')

# Step 2: Plot the median (Q50) line
plt.plot(Dates_forecast_full, linreg_fitted_full[:, 2], 'k', linewidth=1, label='Median (Q50)')

# Step 3: Plot the actual GDP realization
plt.plot(Dates_forecast_full, y_full, 'b', linewidth=1, label='GDP Realization')

# Add recession shading
ax = plt.gca()
add_recession_shading(ax, recession_dates)

# Formatting
plt.title('Linear Regression with Confidence Bands')
plt.xlabel('Time')
plt.ylabel(label_text)
plt.ylim([-15, 20])  # Adjust y-axis bounds as needed
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())  # Format y-axis as percentage
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # Format x-axis as dates
plt.xticks(rotation=45)

# Add a legend
plt.legend(loc='best', fontsize=8)

# Show the plot
plt.show()

#%% Replicating "Figure 3: Quantile Regressions"  in Adrian et al. (2019)
#Quantile regression where x-axis: CISS
# Step 1: Perform Quantile Regression for different quantile levels
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]  # Quantiles to estimate
quantile_fitted = {}  # Dictionary to store fitted values for each quantile

for quantile in quantile_levels:
    # Fit quantile regression using statsmodels
    quantile_model = sm.QuantReg(y, X_with_intercept).fit(q=quantile)
    quantile_fitted[quantile] = quantile_model.predict(X_with_intercept)

# Step 2: Plotting the Data, Mean Regression, and Quantile Regression Lines
plt.figure(figsize=(12, 6))

# Scatter plot of the original data
plt.scatter(X, y, c='b', label='Data 1985-2024', alpha=0.6)

# Plot the mean linear regression line
plt.plot(X, predictions, 'r', linewidth=1.25, label='Mean E[Y|X]')

# Define colors for the quantile lines
color_map = {
    0.05: 'deepskyblue',  # Lower quantiles - blue shades
    0.25: 'dodgerblue',
    0.5: 'black',         # Median - black
    0.75: 'lightgreen',   # Upper quantiles - green shades
    0.95: 'forestgreen'
}

# Plot the quantile regression lines with different colors
for quantile in quantile_levels:
    plt.plot(X, quantile_fitted[quantile], '--', color=color_map[quantile], linewidth=1.25, label=f'Quantile {quantile*100:.0f}%')


# Formatting
plt.xlabel('CISS')
plt.ylabel(label_text)
plt.ylim([-10, 10])  # Adjust y-axis bounds if needed
plt.title('Panel C. One quarter ahead: CISS')
# Add a legend
plt.legend(loc='lower left', fontsize=8)
#plt.savefig('Figures/Adrian_2019_fig2.pdf')
# Show the plot
plt.show()

#%%  Replicating "Figure 5: Predicted Distributions"  in Adrian et al. (2019)
#Quantile Regression Time Series Plot
# Create a matrix for quantile regression fitted values (similar to linreg_fitted_full)
quantile_fitted_full = np.zeros((len(Dates_forecast_full), 5))

# Fill the matrix with quantile fitted values for each quantile
quantile_fitted_full[:, 0] = quantile_fitted[0.05]  # Q05
quantile_fitted_full[:, 1] = quantile_fitted[0.25]  # Q25
quantile_fitted_full[:, 2] = quantile_fitted[0.5]   # Q50 (Median)
quantile_fitted_full[:, 3] = quantile_fitted[0.75]  # Q75
quantile_fitted_full[:, 4] = quantile_fitted[0.95]  # Q95

# Create the figure for the quantile regression time series plot
plt.figure(figsize=(12, 6))

# Step 1: Plot the confidence bands
plt.fill_between(Dates_forecast_full, quantile_fitted_full[:, 0], quantile_fitted_full[:, 4], color='black', alpha=0.35, label='Q05-Q95')
plt.fill_between(Dates_forecast_full, quantile_fitted_full[:, 1], quantile_fitted_full[:, 3], color='black', alpha=0.4, label='Q25-Q75')

# Step 2: Plot the median (Q50) line
plt.plot(Dates_forecast_full, quantile_fitted_full[:, 2], 'k', linewidth=1, label='Median (Q50)')

# Step 3: Plot the actual GDP realization
plt.plot(Dates_forecast_full, y_full, 'b', linewidth=1, label='Realized')

# Add recession shading
ax = plt.gca()
add_recession_shading(ax, recession_dates)

# Formatting
plt.title('Panel A. One quarter ahead')
plt.xlabel('Time')
plt.ylabel(label_text)
plt.ylim([-15, 20])  # Adjust y-axis bounds as needed
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())  # Format y-axis as percentage
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # Format x-axis as dates
plt.xticks(rotation=45)

# Add a legend
plt.legend(loc='best', fontsize=8)

#plt.savefig('fig/Adrian_2019_fig3.pdf')

# Show the plot
plt.show()



df.dropna(subset=['current_gRGDP', ycol], inplace=True)

#%% Replicating 'Figure 2: Raw Data' in Adrian et al. (2019)
#Plot the GDP growth and the current_gRGDP
# Function to add recession shading to plots
def add_recession_shading(ax, recession_dates, color='gray', alpha=0.3):
    """
    Add shaded areas to represent recession periods on a given axis.
    """
    for start_date, end_date in recession_dates:
        ax.axvspan(start_date, end_date, color=color, alpha=alpha)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans serif',
})

plt.figure(figsize=(12, 5))
# Plotting GDP Growth (ycol) on the primary y-axis
ax = df[ycol].plot(label='GDP', linewidth=1, color='b')
ax.set_ylim([-40, 60])
ax.tick_params(axis='y', colors='b')  
ax.set_ylabel('Real GDP Growth (annualized)', color='b')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # Format y-axis as percentage
# Add recession shading
add_recession_shading(ax, recession_dates)
# Plotting current_gRGDP on the secondary y-axis
ax2 = ax.twinx()
df['current_gRGDP'].plot(ax=ax2, label='current quarter GDP growth', linewidth=1, color='r', linestyle='--')
ax2.set_ylabel('current quarter GDP growth', color='r')  
ax2.tick_params(axis='y', colors='r')  
ax2.set_ylim([-1, 5])
# Setting up the legend and title
plt.title('GDP Growth vs current quarter GDP growth')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
#Display the plot
plt.show()


#%% Scatter plot setup
yaxis_bounds_aux = [-20, 20]  # Define y-axis bounds
plt.scatter(df.current_gRGDP, df[ycol], c='b', label='Data 1973-2024')
# Labels and title
plt.xlabel('current quarter GDP growth')
plt.ylabel(label_text)
plt.ylim(yaxis_bounds_aux)
plt.title('Data')
# Format y-axis as percentage (if needed)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
# Add legend
plt.legend(loc='best', fontsize=8)
# Display the plot
plt.show()

# Plot current_gRGDP and shifted GDP growth on two different y-axes
df.loc[:, ['current_gRGDP', ycol]].plot(secondary_y='current_gRGDP')
plt.title('current_gRGDP'' vs GDP Growth')
plt.show()


#%% Linear regression
# Prepare the matrix (current_gRGDP column for X, and ycol for y)
X = df['current_gRGDP']
y = df[ycol]

# Add a constant (intercept term) to the predictor matrix
X_with_intercept = sm.add_constant(X)

# Fit the linear regression model using statsmodels
model = sm.OLS(y, X_with_intercept).fit()

# Get the fitted values (predictions)
predictions = model.predict(X_with_intercept)

# Scatter plot setup
plt.figure(figsize=(10, 6))

# Scatter plot of current_gRGDP vs GDP Growth
plt.scatter(X, y, c='b', label='Data 1973-2024')
# Plot the regression line
plt.plot(X, predictions, 'r', label='Mean E[Y|X]', linewidth=2)
# Labels and title
plt.xlabel('current quarter GDP growth')
plt.ylabel(label_text)
plt.ylim([-10, 10])  # Replace with your specific y-axis bounds if needed
plt.title('Linear Regression: GDP Growth vs current quarter GDP growth')
# Add legend
plt.legend(loc='best', fontsize=8)
# Display the plot
plt.show()


# Time Series Plot with Confidence Bands
# Generate forecast dates
Dates_forecast_full = df.index
linreg_fitted_full = np.zeros((len(Dates_forecast_full), 5))

# Simulate confidence bands (Replace with actual prediction intervals if available)
linreg_fitted_full[:, 2] = predictions  # Median (Q50)
linreg_sigma = np.std(model.resid)  # Estimate residual standard deviation
linreg_fitted_full[:, 0] = predictions - 1.96 * linreg_sigma  # Q05
linreg_fitted_full[:, 4] = predictions + 1.96 * linreg_sigma  # Q95
linreg_fitted_full[:, 1] = predictions - 1.28 * linreg_sigma  # Q25
linreg_fitted_full[:, 3] = predictions + 1.28 * linreg_sigma  # Q75

# Assume y_full is the original target variable for plotting
y_full = df[ycol]

# Create the figure
plt.figure(figsize=(12, 6))

# Step 1: Plot the confidence bands
plt.fill_between(Dates_forecast_full, linreg_fitted_full[:, 0], linreg_fitted_full[:, 4], color='black', alpha=0.35, label='Q05-Q95')
plt.fill_between(Dates_forecast_full, linreg_fitted_full[:, 1], linreg_fitted_full[:, 3], color='black', alpha=0.4, label='Q25-Q75')

# Step 2: Plot the median (Q50) line
plt.plot(Dates_forecast_full, linreg_fitted_full[:, 2], 'k', linewidth=1, label='Median (Q50)')

# Step 3: Plot the actual GDP realization
plt.plot(Dates_forecast_full, y_full, 'b', linewidth=1, label='GDP Realization')

# Add recession shading
ax = plt.gca()
add_recession_shading(ax, recession_dates)

# Formatting
plt.title('Linear Regression with Confidence Bands')
plt.xlabel('Time')
plt.ylabel(label_text)
plt.ylim([-15, 20])  # Adjust y-axis bounds as needed
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())  # Format y-axis as percentage
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # Format x-axis as dates
plt.xticks(rotation=45)

# Add a legend
plt.legend(loc='best', fontsize=8)

# Show the plot
plt.show()




#Replicating "Figure 3: Quantile Regressions"  in Adrian et al. (2019)
#Quantile regression where x-axis: Current quarter GDP growth
X = df['current_gRGDP']  # Replace with the correct data selection logic
y = df[ycol]  # Replace with the actual dependent variable

# Add a constant to the independent variable for the intercept term
X_with_intercept = sm.add_constant(X)

# Quantile regression
quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]  # Quantiles to estimate
quantile_fitted = {}  # Dictionary to store fitted values for each quantile

for quantile in quantile_levels:
    # Fit quantile regression using statsmodels
    quantile_model = sm.QuantReg(y, X_with_intercept).fit(q=quantile)
    quantile_fitted[quantile] = quantile_model.predict(X_with_intercept)

# Step 2: Plotting the Data, Mean Regression, and Quantile Regression Lines
plt.figure(figsize=(12, 6))

# Scatter plot of the original data
plt.scatter(X, y, c='b', label='Data 1985-2024', alpha=0.6)

# Fit and plot the mean linear regression line
mean_model = sm.OLS(y, X_with_intercept).fit()
predictions = mean_model.predict(X_with_intercept)
plt.plot(X, predictions, 'r', linewidth=1.25, label='Mean E[Y|X]')

# Define colors for the quantile lines
color_map = {
    0.05: 'deepskyblue',  # Lower quantiles - blue shades
    0.25: 'dodgerblue',
    0.5: 'black',         # Median - black
    0.75: 'lightgreen',   # Upper quantiles - green shades
    0.95: 'forestgreen'
}

# Plot the quantile regression lines with different colors
for quantile in quantile_levels:
    plt.plot(X, quantile_fitted[quantile], '--', color=color_map[quantile], linewidth=1.25, label=f'Quantile {quantile*100:.0f}%')

# Formatting
plt.xlabel('Current quarter GDP growth')
plt.ylabel(label_text)  # Replace with the actual label for y
plt.ylim([-10, 10])  # Adjust y-axis bounds if needed
plt.title('Panel A. One quarter ahead: GDP')
plt.legend(loc='lower left', fontsize=8)

# Show the plot
plt.show()
