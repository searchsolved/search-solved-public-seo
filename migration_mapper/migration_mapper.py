""" Automatic Website Migration Tool, By @LeeFootSEO - 03/04/2023"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a list of column names to match on
all_columns = ["Address", "H1-1", "Title 1"]

# Load the first CSV file into a pandas dataframe
df_live = pd.read_csv('/python_scripts/migration_mapper/live.csv', dtype="str")

# Load the second CSV file into a pandas dataframe
df_staging = pd.read_csv('/python_scripts/migration_mapper/staging.csv', dtype="str")

# Load a pre-trained BERT model from the sbert library
model = SentenceTransformer('all-distilroberta-v1')  # best

df_live[all_columns] = df_live[all_columns].apply(lambda x: x.str.lower())
df_staging[all_columns] = df_staging[all_columns].apply(lambda x: x.str.lower())

# Precompute the embeddings for each column and normalize them
encoded_cols_live = {}
encoded_cols_staging = {}

for col in tqdm(all_columns, desc='Precomputing embeddings'):
    encoded_cols_live[col] = model.encode(df_live[col].astype(str).tolist(), convert_to_tensor=True).cpu().numpy()
    encoded_cols_live[col] /= np.linalg.norm(encoded_cols_live[col], axis=1, keepdims=True)
    encoded_cols_staging[col] = model.encode(df_staging[col].astype(str).tolist(), convert_to_tensor=True).cpu().numpy()
    encoded_cols_staging[col] /= np.linalg.norm(encoded_cols_staging[col], axis=1, keepdims=True)

all_matches = []

for col in all_columns:
    # Drop NaN values for this column in both dataframes
    live_col = df_live[col].dropna()
    staging_col = df_staging[col].dropna()

    # Retrieve the precomputed embeddings for each matching column
    encoded_df_live = encoded_cols_live[col][live_col.index]
    encoded_df_staging = encoded_cols_staging[col][staging_col.index]

    # Convert NumPy arrays to PyTorch tensors
    encoded_df_live = torch.from_numpy(encoded_df_live).to(device)
    encoded_df_staging = torch.from_numpy(encoded_df_staging).to(device)

    # Compute the cosine similarity between the two matrices
    cosine_similarities = torch.matmul(encoded_df_live, encoded_df_staging.T).cpu().numpy()

    # Find the best match between the two dataframes
    matches = []

    # Loop through each row in the live dataframe
    desc = f'Matching {len(live_col)} rows in live dataframe with column {col}'
    for i, row in enumerate(tqdm(encoded_df_live, desc=desc)):
        best_score = np.max(cosine_similarities[i])
        best_match = np.argmax(cosine_similarities[i])
        if best_score > 0:
            matches.append({'Live Address': df_live.loc[live_col.index[i], 'Address'],
                            'Staging Address': df_staging.loc[staging_col.index[best_match], 'Address'],
                            'Matching Column': col, 'Highest Score': best_score})
        else:
            matches.append({'Live Address': df_live.loc[live_col.index[i], 'Address'], 'Staging Address': '',
                            'Matching Column': col, 'Highest Score': 0})

    # Append matches for this column to the overall list of matches
    all_matches.extend(matches)

# Convert the matches to a pandas dataframe
df = pd.DataFrame(all_matches)
df = df.sort_values(by="Highest Score", ascending=False)

# calculate median and number of columns matched on
median_scores = df.groupby('Live Address')['Highest Score'].median()
num_matched_cols = df.groupby('Live Address')['Matching Column'].count()

# create new columns in the original dataframe
df['Median Score'] = df['Live Address'].map(median_scores)
df['Number of Columns Matched'] = df['Live Address'].map(num_matched_cols)

# Drop duplicates based on the "Live Address" column
df.drop_duplicates(subset=['Live Address'], keep="first", inplace=True)

# Group the matches by the "Live Address" column and select the highest scoring match for each group
df_max = df.groupby('Live Address').apply(lambda x: x.loc[x['Highest Score'].idxmax()]).reset_index(drop=True)

# Output the final dataframe to a CSV file
df_max.to_csv("/python_scripts/migration_mapper_output.csv", index=False)

# Find the unmatched rows in the staging dataframe
staging_unmatched = df_staging[~df_staging['Address'].isin(df_max['Staging Address'])]

# Output the unmatched rows to a separate CSV file
staging_unmatched.to_csv('/python_scripts/staging_unmatched.csv', index=False)
