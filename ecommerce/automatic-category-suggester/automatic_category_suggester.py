# Automatic Category Suggester by LeeFoot
# Twitter: @leefootseo
# Web: https://LeeFoot.co.uk
# Contact me if you'd like this run as a managed service.

# Standard Library Imports
import re
import string
import collections

# Third-Party Library Imports
import pandas as pd
from nltk.util import ngrams
from tqdm import tqdm
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor


# Constants
# MODEL_TYPE = "paraphrase-MiniLM-L3-v2"  # fastest cosine
MODEL_TYPE = "multi-qa-mpnet-base-cos-v1"  # best semantic cosine
NUM_WORKERS = 8  # number of workers to use for multi-threading
MIN_MATCHING_PRODUCTS_EXACT = 0  # filter to a minimum numer of matching products in an exact match


# ---------------
# Data Loading and Cleaning
# ---------------

def load_csv(path, usecols=None, dtype="str"):
    """Load a CSV file."""
    return pd.read_csv(path, usecols=usecols, dtype=dtype)


def filter_status_code(df):
    """Filter DataFrame rows based on 'Status Code'."""
    return df[df["Status Code"].isin(["200"])] if "Status Code" in df.columns else df


def filter_type_hyperlink(df):
    """Filter DataFrame rows based on 'Type' being 'Hyperlink'."""
    return df[df["Type"].isin(["Hyperlink"])] if "Type" in df.columns else df


def filter_non_indexable(df):
    """Filter out 'Non-Indexable' rows from 'Indexability'."""
    return df[~df["Indexability"].isin(["Non-Indexable"])] if "Indexability" in df.columns else df


def process_headers_and_duplicates(df):
    """Process header fields and remove duplicates."""
    if "H1-1" in df.columns:
        df = df.assign(**{'H1-1': df['H1-1'].str.lower()})
        df = df.drop_duplicates(subset="H1-1")
        df = df[df["H1-1"].notna()]
    if "From" in df.columns and "To" in df.columns:
        df.drop_duplicates(subset=["From", "To"], keep="first", inplace=True)
    if "Page Type" in df.columns:
        df = df[df["Page Type"].notna()]
    return df


def clean_and_prepare_text(text):
    """Clean and prepare text for n-gram generation."""
    text = re.sub("<.*?>", "", text)  # Remove HTML tags
    text = "".join(c for c in text if c not in string.punctuation and not c.isdigit())
    text = text.lower()
    return text


# ---------------
# N-gram Generation
# ---------------

def generate_ngrams(text, min_length=2, max_length=7):
    """Generate and count n-grams for a given text."""
    tokenized = text.split()
    all_ngrams = sum([list(ngrams(tokenized, i)) for i in range(min_length, max_length)], [])
    ngrams_freq = collections.Counter(all_ngrams)
    return ngrams_freq.most_common(100)


# ---------------
# Match Calculation
# ---------------

def exact_match_worker(keyword, product_titles):
    return sum(keyword == title for title in product_titles)

def partial_match_worker(keyword, product_titles):
    keyword_words = set(keyword.split())
    return sum(bool(keyword_words.intersection(set(title.split()))) for title in product_titles)


def calculate_exact_matches(df_ngrams, column_name, product_titles):
    with ProcessPoolExecutor() as executor:
        # Map each keyword to the worker function in parallel and directly assign the results back to the DataFrame
        df_ngrams['matching_products_exact'] = list(tqdm(executor.map(exact_match_worker, df_ngrams[column_name], [product_titles]*len(df_ngrams)), total=len(df_ngrams), desc="Calculating Exact Matches"))
    return df_ngrams

def calculate_partial_matches(df_ngrams, column_name, product_titles):
    with ProcessPoolExecutor() as executor:
        # Map each keyword to the worker function in parallel and directly assign the results back to the DataFrame
        df_ngrams['matching_products_partial'] = list(tqdm(executor.map(partial_match_worker, df_ngrams[column_name], [product_titles]*len(df_ngrams)), total=len(df_ngrams), desc="Calculating Partial Matches"))
    return df_ngrams


def prepare_data_for_fuzzy_matching(df):
    """Prepare data by dropping NaN values and ensuring non-empty strings for 'Keyword' and 'H1-1'."""
    df = df.dropna(subset=['Keyword', 'H1-1'])  # Drop rows where either 'Keyword' or 'H1-1' is NaN.
    # Ensure both 'Keyword' and 'H1-1' are non-empty strings.
    df = df[(df['Keyword'].astype(str).str.strip() != '') & (df['H1-1'].astype(str).str.strip() != '')]
    return df


def fuzzy_match_single_row(row):
    # Using RapidFuzz's partial_ratio to calculate the similarity
    partial_ratio = fuzz.partial_ratio(row['Keyword'].lower(), row['H1-1'].lower())
    return partial_ratio


def fuzzy_match_keyword(df):
    df = prepare_data_for_fuzzy_matching(df)

    with tqdm(total=len(df), desc="Calculating Fuzzy Matches") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  # Adjust `max_workers` as needed
            futures = {executor.submit(fuzzy_match_single_row, row): index for index, row in df.iterrows()}

            for future in as_completed(futures):
                result = future.result()
                row_index = futures[future]
                df.at[row_index, 'Similarity'] = result
                pbar.update(1)  # Update progress after each task is completed

    return df

# ---------------
# Main Processing Flow
# ---------------

def process_product_and_category_data(internal_html, inlinks):
    """Separate product and category pages and merge with inlinks."""
    product = internal_html[internal_html['Page Type'].str.contains("Product Page", na=False)]
    category = internal_html[internal_html['Page Type'].str.contains("Category Page", na=False)]
    inlinks_renamed = inlinks.rename(columns={"From": "Parent URL", "To": "Product URL"})
    product = pd.merge(product, inlinks_renamed, left_on="Address", right_on="Product URL", how='left')
    product_pages = internal_html[internal_html['Page Type'] == 'Product Page']['Address']
    product = product[~product['Parent URL'].isin(product_pages)]
    return product, category


def generate_ngrams_for_products(product_df):
    """Generate n-grams for product titles."""
    ngrams_list = []
    for parent_url in tqdm(product_df['Parent URL'].unique(), desc="Processing Parent URLs"):
        if pd.isna(parent_url):
            continue
        df_parent = product_df[product_df["Parent URL"] == parent_url]
        text = " ".join(df_parent["H1-1"].dropna().astype(str).tolist()).lower()
        text = clean_and_prepare_text(text)
        ngrams_freq = generate_ngrams(text)
        for ngram, freq in ngrams_freq:
            ngrams_list.append({'Parent URL': parent_url, 'Keyword': ' '.join(ngram), 'Frequency': freq})
    return pd.DataFrame(ngrams_list)


def calculate_matches(df_ngrams, product_df):
    """Calculate exact and partial matches for n-grams and product titles."""
    product_titles = product_df['H1-1'].dropna().str.lower().unique().tolist()
    df_ngrams_exact = calculate_exact_matches(df_ngrams.copy(), 'Keyword', product_titles)
    df_ngrams_partial = calculate_partial_matches(df_ngrams_exact, 'Keyword', product_titles)
    return df_ngrams_partial[df_ngrams_partial['matching_products_exact'] >= MIN_MATCHING_PRODUCTS_EXACT]


def merge_keywords_with_categories(df_ngrams, category_df):
    """Merge keywords with category data."""
    merged_df = pd.merge(category_df, df_ngrams, left_on='Address', right_on='Parent URL', how='left')
    return merged_df.drop(columns=['Parent URL'])


def save_results_to_csv(df, path):
    """Save DataFrame to CSV file."""
    df.to_csv(path, index=False)


def main():
    df = load_csv('/python_scripts/cat_splitter/old_files/internal_html.csv',
                  usecols=["Address", "H1-1", "Title 1", "Page Type"], dtype="str")
    inlinks = load_csv('/python_scripts/cat_splitter//old_files/inlinks.csv', dtype="str")

    # clean the source dataframes
    df = filter_status_code(df)
    df = filter_type_hyperlink(df)
    df = filter_non_indexable(df)
    df = process_headers_and_duplicates(df)

    # Process product and category data
    product, category = process_product_and_category_data(df, inlinks)

    # Generate n-grams for products
    df_ngrams = generate_ngrams_for_products(product)

    # Calculate matches
    df_ngrams_with_matches = calculate_matches(df_ngrams, product)

    # Merge keywords with category data
    category_with_matched_keywords = merge_keywords_with_categories(df_ngrams_with_matches, category)

    # Optional: Apply additional filtering (e.g., fuzzy matching)
    category_with_matched_keywords_filtered = fuzzy_match_keyword(category_with_matched_keywords)

    # Save results
    save_results_to_csv(category_with_matched_keywords_filtered,
                        "/python_scripts/cat_splitter/final_results.csv")


if __name__ == "__main__":
    main()
