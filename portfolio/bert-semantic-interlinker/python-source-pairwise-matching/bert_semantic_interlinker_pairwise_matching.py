# BERT Semantic Interlinker by @LeeFootSEO - 11th December 2023
# Pair-wise similarity matches
# Need this as a managed service? Contact: hello@leefoot.co.uk
# Web: https://leefoot.co.uk

import time
import torch

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Global variables
INPUT_FILE_PATH = '/python_scripts/internal_html.csv'
OUTPUT_FILE_PATH = '/python_scripts/bert_clustered_results.csv'
TRANSFORMER_MODEL = 'multi-qa-mpnet-base-dot-v1'
MIN_SIMILARITY = 0.8
MAX_SUGGESTIONS_PER_PAGE = 10

# Automatically detect CUDA
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


def read_and_clean_data(filepath):
    """
    Read and clean the CSV data.

    This function reads a CSV file and cleans it by removing rows where the 'H1-1' column is NaN or starts with "All".

    Args:
    filepath (str): The file path to the CSV file to be read.

    Returns:
    pandas.DataFrame: A DataFrame with the cleaned data.
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    df = df[df["H1-1"].notna()]
    df = df[~df["H1-1"].str.contains("^All ", na=False, regex=True)]
    return df


def precompute_embeddings(df):
    """
    Precompute embeddings for the given DataFrame using a SentenceTransformer model.

    Args:
    df (pandas.DataFrame): A DataFrame containing the text to encode in the 'H1-1' column.

    Returns:
    tuple: A tuple containing the embeddings array and the list of original text strings.
    """
    embedding_model = SentenceTransformer(TRANSFORMER_MODEL, device=DEVICE)
    to_list = list(df['H1-1'])
    to_embeddings = embedding_model.encode(to_list)
    return to_embeddings, to_list


def find_matches(from_list, to_list, to_embeddings, embedding_model):
    """
    Find semantically similar matches between two lists of text strings using cosine similarity.

    Args:
    from_list (list of str): The list of text strings to find matches for.
    to_list (list of str): The list of text strings to match against.
    to_embeddings (numpy.ndarray): The embeddings corresponding to the 'to_list'.
    embedding_model (SentenceTransformer): The SentenceTransformer model to encode the 'from_list'.

    Returns:
    pandas.DataFrame: A DataFrame with columns 'From', 'To', and 'Similarity' for each match.
    """
    dfs = []
    with tqdm(total=len(from_list), desc="Finding Matches") as pbar:
        for i, kw in enumerate(from_list):
            kw_embedding = embedding_model.encode([kw])
            similarities = cosine_similarity(kw_embedding, to_embeddings)[0]
            matches = np.where(similarities >= MIN_SIMILARITY)[0]
            matches = matches[similarities[matches].argsort()[::-1]]
            if len(matches) > 0:
                match_indices = matches[:MAX_SUGGESTIONS_PER_PAGE]
                df = pd.DataFrame({
                    'From': [kw] * len(match_indices),
                    'To': [to_list[j] for j in match_indices],
                    'Similarity': [similarities[j] for j in match_indices]
                })
                dfs.append(df)
            pbar.update(1)
    return pd.concat(dfs)


def merge_url_data(df_final, df_h1_urls):
    """
    Merge URL data into the final DataFrame containing the match results.

    Args:
    df_final (pandas.DataFrame): The DataFrame with match results.
    df_h1_urls (pandas.DataFrame): The DataFrame containing the URLs.

    Returns:
    pandas.DataFrame: The merged DataFrame with 'Source URL' and 'Destination URL' added.
    """
    df_final = pd.merge(df_final, df_h1_urls, left_on="From", right_on="H1-1", how="left")
    df_final = df_final.rename(columns={"Address": "Source URL"})
    del df_final['H1-1']
    df_final = pd.merge(df_final, df_h1_urls, left_on="To", right_on="H1-1", how="left")
    df_final = df_final.rename(columns={"Address": "Destination URL"})
    del df_final['H1-1']
    return df_final


def process_final_df(df_final):
    """
    Process the final DataFrame by filtering duplicates, sorting, and applying thresholds.

    Args:
    df_final (pandas.DataFrame): The DataFrame to be processed.

    Returns:
    pandas.DataFrame: The processed DataFrame with duplicates removed and sorted by similarity score.
    """
    df_final.drop_duplicates(subset=["Source URL", "Destination URL"], keep="first", inplace=True)
    df_final = df_final.rename(columns={"From": "Source H1", "To": "Destination H1"})
    df_final = df_final[["Source H1", "Destination H1", "Similarity", "Source URL", "Destination URL"]]
    df_final.sort_values(["Source H1", "Similarity"], ascending=[True, False], inplace=True)
    df_final = df_final.groupby(['Source H1']).head(MAX_SUGGESTIONS_PER_PAGE)
    df_final = df_final[df_final.Similarity > MIN_SIMILARITY]
    df_final['Match'] = df_final['Source H1'] == df_final['Destination H1']
    df_final = df_final[df_final.Match == False]
    del df_final['Match']
    df_final['Similarity'] = df_final['Similarity'].round(2)
    return df_final


def save_results(df_final):
    """
    Save the processed DataFrame to a CSV file.

    Args:
    df_final (pandas.DataFrame): The DataFrame with the final results to be saved.
    """
    df_final.to_csv(OUTPUT_FILE_PATH, index=False)


def main():
    """
    Main function to run the clustering process.

    This function orchestrates the reading, cleaning, embedding, matching, and saving of clustering results.
    """
    startTime = time.time()
    df = read_and_clean_data(INPUT_FILE_PATH)
    df_h1_urls = df[['Address', 'H1-1']]
    from_list = list(df['H1-1'])
    to_embeddings, to_list = precompute_embeddings(df)
    embedding_model = SentenceTransformer(TRANSFORMER_MODEL, device=DEVICE)
    df_matches = find_matches(from_list, to_list, to_embeddings, embedding_model)
    df_final = merge_url_data(df_matches, df_h1_urls)
    df_final_processed = process_final_df(df_final)
    save_results(df_final_processed)
    print("The script took {0} seconds!".format(time.time() - startTime))


if __name__ == "__main__":
    main()