""" Automatic Website Migration Tool V5, By @LeeFootSEO | https://LeeFoot.co.uk | 06/12/2023 

Uses a grid search to find the highest scoring match between columns specified in the matching_columns variable.
"""

import os
import pandas as pd
from polyfuzz import PolyFuzz
from tqdm import tqdm
import chardet

def file_exists(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return False
    return True

def read_csv_with_encoding(file_path, dtype):
    print(f"Detecting encoding for {file_path}...")
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']

    print(f"Reading {file_path} with encoding {encoding}...")
    try:
        return pd.read_csv(file_path, dtype=dtype, encoding=encoding, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# Check if files exist before loading
if file_exists('/python_scripts/migration_mapper/live.csv') and file_exists('/python_scripts/migration_mapper/staging.csv'):
    df_live = read_csv_with_encoding('/python_scripts/migration_mapper/live.csv', dtype="str")
    df_staging = read_csv_with_encoding('/python_scripts/migration_mapper/staging.csv', dtype="str")
else:
    raise FileNotFoundError("One or more input files are missing.")

# Check if DataFrames are empty
if df_live.empty or df_staging.empty:
    raise ValueError("One or more input DataFrames are empty after reading CSV files.")

# Convert to lowercase for case-insensitive matching
print("Preprocessing data...")
try:
    df_live = df_live.apply(lambda col: col.str.lower())
    df_staging = df_staging.apply(lambda col: col.str.lower())
except Exception as e:
    print(f"Error during preprocessing: {e}")

# Define the list of columns to match on including 'Address'
matching_columns = ['Address', 'H1-1', 'Title 1']  # Example columns

# Create a PolyFuzz model
print("Initializing PolyFuzz model...")
model = PolyFuzz("TF-IDF")

# Function to match and score each column
def match_and_score(col):
    # Handle NaN values by replacing them with an empty string
    live_list = df_live[col].fillna('').tolist()
    staging_list = df_staging[col].fillna('').tolist()

    # Perform matching only if both lists have content
    if live_list and staging_list:
        print(f"Matching {col}...")
        model.match(live_list, staging_list)
        return model.get_matches()
    else:
        return pd.DataFrame(columns=['From', 'To', 'Similarity'])

# Match each column and collect scores
print("Matching columns and collecting scores...")
matches_scores = {col: match_and_score(col) for col in tqdm(matching_columns, desc="Matching columns")}

# Function to find the overall best match for each row
def find_best_overall_match(row):
    best_match_info = {
        'Best Match on': None,
        'Highest Matching URL': None,
        'Highest Similarity Score': 0
    }

    for col in matching_columns:
        matches = matches_scores[col]
        if not matches.empty:
            match_row = matches.loc[matches['From'] == row[col]]
            if not match_row.empty and match_row.iloc[0]['Similarity'] > best_match_info['Highest Similarity Score']:
                best_match_info['Best Match on'] = col
                best_match_info['Highest Matching URL'] = df_staging.loc[
                    df_staging[col] == match_row.iloc[0]['To'], 'Address'
                ].values[0]
                best_match_info['Highest Similarity Score'] = match_row.iloc[0]['Similarity']

    return pd.Series(best_match_info)

# Apply the function to find the best overall match
print("Applying match function to each row...")
try:
    match_results = df_live.apply(find_best_overall_match, axis=1)
except Exception as e:
    print(f"Error during matching process: {e}")

# Concatenate the match results with the original dataframe
# Ensure to not include 'Address' from matching_columns in the final DataFrame as it is already present in df_live
print("Compiling final results...")
try:
    final_columns = ['Address'] + [col for col in matching_columns if col != 'Address']
    df_final = pd.concat([df_live[final_columns], match_results], axis=1)
except Exception as e:
    print(f"Error during DataFrame operations: {e}")

# Export the results
output_path = '/python_scripts/migration_mapper_output.csv'
print(f"Exporting results to {output_path}...")
try:
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
except Exception as e:
    print(f"Error exporting data: {e}")

print("All operations completed successfully.")
