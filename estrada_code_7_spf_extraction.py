#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:27:43 2024

@author: dianniadreiestrada
"""

import pandas as pd
import os

os.chdir('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise')
csv_files = os.listdir('data/SPF_individual_forecasts/')

def convert_to_date(file, df, year):

    # Extract the quarter based on the file naming logic
    quarter = int(file[5:6])
   
    # Update logic to extract the correct period
    if quarter == 1:
        # December of the current year
        df = df.loc[df['TARGET_PERIOD'] == f"{year}Q3", :]
        df['Date'] = pd.to_datetime(f"{year}-09-01")
    elif quarter == 2:
        # March of the next year
        df = df.loc[df['TARGET_PERIOD'] == f"{year}Q4", :]
        df['Date'] = pd.to_datetime(f"{year}-12-01")
    elif quarter == 3:
        # June of the next year
        df = df.loc[df['TARGET_PERIOD'] == f"{year+1}Q1", :]
        df['Date'] = pd.to_datetime(f"{year+1}-03-01")
    elif quarter == 4:
        # September of the next year
        df = df.loc[df['TARGET_PERIOD'] == f"{year+1}Q2", :]
        df['Date'] = pd.to_datetime(f"{year+1}-06-01")

    return df


# Define a function to capture lines of interest
def extract_growth_expectations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
   
    # Initialize variables for storing relevant lines
    extract_lines = False
    growth_data = []

    for line in lines:
        # Check for the target line to start extraction
        if "GROWTH EXPECTATIONS; YEAR-ON-YEAR CHANGE IN REAL GDP" in line:
            extract_lines = True
            continue
       
        # Stop extraction when an empty line is encountered
        if extract_lines:
            if line.strip() == "":
                break
            growth_data.append(line.strip())
   
    # Convert the list of extracted lines to a DataFrame
    if growth_data:
        growth_df = pd.DataFrame([row.split(",") for row in growth_data])
        # First line as column names
        # Set the first row as the header (column names)
        growth_df.columns = growth_df.iloc[0]
       
        # Drop the first row from the DataFrame
        growth_df = growth_df.drop(0).reset_index(drop=True)
        # convert to numeric
        if '' in growth_df.columns:
            del growth_df['']
        for coli in growth_df.columns:
            if coli != 'TARGET_PERIOD':
                growth_df[coli] = pd.to_numeric(growth_df[coli], errors='coerce')
        # Find the first empty row to determine where to stop
        # Identify the columns to check for null values, excluding 'TARGET_PERIOD'
        columns_to_check = growth_df.columns.difference(['TARGET_PERIOD'])
       
        # Find the first row where all values are NaN in the specified columns
        empty_line_index = growth_df[growth_df[columns_to_check].isna().all(axis=1)].index[0]

        # Keep all rows up until the empty line
        growth_df = growth_df.iloc[:empty_line_index, :]
        return growth_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data is found


# Initialize an empty DataFrame to store results across all files
df_panel = pd.DataFrame()

# Loop through each CSV file and process
for file in csv_files:
    file_path = f'data/SPF_individual_forecasts/{file}'
   
    # Extract growth expectations data from each file
    growth_df = extract_growth_expectations(file_path)
   
    # If growth data exists, process further
    if not growth_df.empty:
        year = int(file[:4])  # Assuming the year is at the start of the filename
        growth_df = convert_to_date(file, growth_df, year)  # Process with existing logic
        df_panel = pd.concat([df_panel, growth_df], ignore_index=True)


df_panel.dropna(inplace=True, how='all', axis=1)
df_panel.dropna(inplace=True, how='all', axis=0)


# Dictionary mapping columns to significance
column_significance_map = {
    'TN15_0':']-inf,-15]',
    'TN6_0':']-inf,-6]',
    'TN4_0': ']-inf, - 4]',
    'TN2_0': ']-inf, - 2]',
    'TN1_0': ']-inf, -1]',
    'T0_0': ']-inf, -0]',
    'FN15_0TN13_1':'[-15,-13]',
    'FN13_0TN11_1':'[-13,-11]',
    'FN11_0TN9_1':'[-11,-9]',
    'FN9_0TN7_1':'[-9,-7]',
    'FN7_0TN5_1':'[-7,-5]',
    'FN6_0TN5_6':'[-6,-5.5]',
    'FN5_5TN5_1':'[-5.5,-5]',
    'FN5_0TN4_6':'[-5,-4.5]',
    'FN4_5TN4_1':'[-4.5,-4]',
    'FN5_0TN3_1':'[-5,-3]',
    'FN3_0TN1_1':'[-3,-1]',
    'FN4_0TN3_6': '[-4,-3.5]',
    'FN3_5TN3_1': '[-3.5,-3.1]',
    'FN3_0TN2_6': '[-3,-2.5]',
    'FN2_5TN2_1': '[-2.5,-2.1]',
    'FN2_0TN1_6': '[-2,-1.5]',
    'FN1_5TN1_1': '[-1.5,-1.1]',
    'FN1_0TN0_6': '[-1,-0.5]',
    'FN0_5TN0_1': '[-0.5,-0.1]',
    'F0_0T0_4': '[0,0.5]',
    'F0_5T0_9': '[0.5,1]',
    'F1_0T1_4': '[1,1.5]',
    'F1_5T1_9': '[1.5,2]',
    'F2_0T2_4': '[2,2.5]',
    'F2_5T2_9': '[2.5,3]',
    'F3_0T3_4': '[3,3.5]',
    'F3_5T3_9': '[3.5,4]',
    'F4_0T4_4': '[4.0,4.5]',
    'F4_0T5_9':'[4,6]',
    'F6_0T7_9':'[6,8]',
    'F4_5T4_9': '[4.5,5]',
    'F8_0T9_9': '[8, 10]',
    'F10_0': '[10,+inf[',
    'F3_5': '[3.5,+inf[',
    'F4_0': '[4,+inf[',
    'F5_0': '[5,+inf['
}


# Rename the columns in the DataFrame based on the dictionary
df_panel.rename(columns=column_significance_map, inplace=True)
 
       
# order the columns
df_panel = df_panel.loc[:,['Date', 'FCT_SOURCE', 'POINT', ']-inf,-15]', ']-inf,-6]', ']-inf, -1]',
       ']-inf, -0]', '[-15,-13]', '[-13,-11]', '[-11,-9]', '[-9,-7]',
       '[-7,-5]', '[-6,-5.5]', '[-5.5,-5]', '[-5,-4.5]','[-5,-3]','[-4.5,-4]',
       '[-4,-3.5]', '[-3.5,-3.1]', '[-3,-2.5]','[-3,-1]', '[-2.5,-2.1]', '[-2,-1.5]',
       '[-1.5,-1.1]','[-1,-0.5]',  '[-0.5,-0.1]',  
       '[0,0.5]', '[0.5,1]', '[1,1.5]', '[1.5,2]', '[2,2.5]', '[2.5,3]',
       '[3,3.5]', '[3.5,4]', '[4.0,4.5]', '[4.5,5]','[4,6]', '[6,8]', '[8, 10]', '[5,+inf[',
       '[4,+inf[', '[10,+inf[' ]]

df_panel.sort_values(by='Date', inplace=True)

df_panel.to_csv('/Users/dianniadreiestrada/Desktop/QMF_Replication_Exercise/Data/spf_df.csv', index=False)
