# Automatic Category Suggester by LeeFoot
# Twitter: @leefootseo
# Web: https://LeeFoot.co.uk
# Contact me if you'd like this run as a managed service.

# Automatic Category Suggester by LeeFoot
# Twitter: @leefootseo
# Web: https://LeeFoot.co.uk
# Contact me if you'd like this run as a managed service.

import pandas as pd
import re
import string
from nltk.util import ngrams
import collections
from tqdm import tqdm  # Import tqdm for progress tracking
from sentence_transformers import SentenceTransformer, util
import torch

MIN_MATCHING_PRODUCTS = 1  # the number of minimum products to match to. (kws found exactly in sequence in products).
TRANSFORMER_MODEL = "paraphrase-MiniLM-L3-v2"

# Assuming your CSV paths are correct
inlinks = pd.read_csv('/python_scripts/cat_splitter/old_files/inlinks.csv', dtype="str")
internal_html = pd.read_csv('/python_scripts/cat_splitter/old_files/internal_html.csv',
                            usecols=["Address", "H1-1", "Title 1", "Page Type"], dtype="str")


def clean_df(df):
    if "Status Code" in df.columns:
        df = df[df["Status Code"].isin(["200"])]
    if "Type" in df.columns:
        df = df[df["Type"].isin(["Hyperlink"])]
    if "Indexability" in df.columns:
        df = df[~df["Indexability"].isin(["Non-Indexable"])]
    if "H1-1" in df.columns:
        df = df.assign(**{'H1-1': df['H1-1'].str.lower()})
        df = df.drop_duplicates(subset="H1-1")
        df = df[df["H1-1"].notna()]
    if "From" in df.columns and "To" in df.columns:
        df.drop_duplicates(subset=["From", "To"], keep="first", inplace=True)
    if "Page Type" in df.columns:
        df = df[df["Page Type"].notna()]
    return df


# clean source dfs
inlinks = clean_df(inlinks)
internal_html = clean_df(internal_html)

# create separate product and category dataframes
product = internal_html[internal_html['Page Type'].str.contains("Product Page", na=False)]
category = internal_html[internal_html['Page Type'].str.contains("Category Page", na=False)]

# merge and rename product and inlinks dataframes
inlinks = inlinks[["From", "To"]]
product = pd.merge(product, inlinks, left_on="Address", right_on="To", how='left')
product.rename(columns={"From": "Parent URL", "To": "Product URL"}, inplace=True)

# only keep parent pages that are category pages
product_pages = internal_html[internal_html['Page Type'] == 'Product Page']['Address']
product = product[~product['Parent URL'].isin(product_pages)]


def filter_df_for_parent_url(product_df, parent_url):
    return product_df[product_df["Parent URL"] == parent_url]


def clean_and_prepare_text(df_parent):
    text = " ".join(df_parent["H1-1"].dropna().astype(str).tolist()).lower()
    text = "".join(c for c in text if not c.isdigit())
    text = re.sub("<.*?>", "", text)
    punctuation_no_full_stop = "[" + re.sub("\.", "", string.punctuation) + "]"
    text = re.sub(punctuation_no_full_stop, "", text)
    return text


def generate_ngrams_and_frequencies(text):
    tokenized = text.split()
    all_ngrams = [ngrams(tokenized, i) for i in range(2, 8)]
    all_ngrams_freq = [collections.Counter(ngram) for ngram in all_ngrams]
    ngrams_freq_tuples = sum([list(freq.items()) for freq in all_ngrams_freq], [])
    ngrams_combined_list = [(' '.join(gram), freq) for gram, freq in ngrams_freq_tuples]
    ngrams_combined_list.sort(key=lambda x: x[1], reverse=True)
    return ngrams_combined_list[:100]


def create_ngram_dataframe(ngrams_list, parent_url):
    df_ngrams = pd.DataFrame(ngrams_list, columns=["Keyword", "Frequency"])
    df_ngrams["Parent URL"] = parent_url
    return df_ngrams


def process_ngrams_for_products(product_df):
    appended_data = []
    for parent_url in tqdm(product_df['Parent URL'].unique(), desc="Generating Ngrams"):
        if pd.isna(parent_url):
            continue
        df_parent = filter_df_for_parent_url(product_df, parent_url)
        text = clean_and_prepare_text(df_parent)
        ngrams_list = generate_ngrams_and_frequencies(text)
        df_ngrams = create_ngram_dataframe(ngrams_list, parent_url)
        appended_data.append(df_ngrams)
    return pd.concat(appended_data).reset_index(drop=True)


df_ngrams = process_ngrams_for_products(product)


def calculate_exact_match(df_ngrams, product_df, min_products=MIN_MATCHING_PRODUCTS):
    product_h1_set = set(product_df['H1-1'].dropna().str.lower().unique())

    def count_exact_matches(keyword):
        return sum(keyword in title for title in product_h1_set)

    tqdm.pandas(desc="Calculating Exact Matches")
    df_ngrams['matching_products_exact'] = df_ngrams['Keyword'].progress_apply(count_exact_matches)
    df_filtered_ngrams = df_ngrams[df_ngrams['matching_products_exact'] >= min_products]

    return df_filtered_ngrams


df_ngrams_with_exact_match = calculate_exact_match(df_ngrams, product)


def merge_keywords_into_category(df_ngrams, category_df):
    merged_df = pd.merge(category_df, df_ngrams[['Parent URL', 'Keyword', 'matching_products_exact']],
                         left_on='Address', right_on='Parent URL', how='left')
    # Filter out rows with less than the minimum required matches
    merged_df = merged_df[merged_df['matching_products_exact'] >= MIN_MATCHING_PRODUCTS]
    merged_df.drop(columns=['Parent URL'], inplace=True)
    return merged_df


category_with_exact_match_keywords = merge_keywords_into_category(df_ngrams_with_exact_match, category)


def encode_texts_with_model(texts, model, batch_size=32, desc="Encoding texts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = []

    # Process texts in batches
    for batch_start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[batch_start:batch_start + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def calculate_semantic_similarity(df_ngrams, product_df, model_name=TRANSFORMER_MODEL, similarity_threshold=0.5,
                                  batch_size=32):
    # Load the model
    model = SentenceTransformer(model_name)

    # Prepare the texts
    keywords = df_ngrams['Keyword'].unique().tolist()
    product_titles = product_df['H1-1'].dropna().unique().tolist()

    # Encode the texts to get their embeddings with specific progress descriptions
    keyword_embeddings = encode_texts_with_model(keywords, model, batch_size, desc="Encoding Keywords")
    product_embeddings = encode_texts_with_model(product_titles, model, batch_size, desc="Encoding Product Titles")

    print("Product titles encoding complete. Proceeding with similarity calculations...")

    # Calculate cosine similarity
    cosine_similarities = util.pytorch_cos_sim(keyword_embeddings, product_embeddings)

    # Process the results to filter matches based on the similarity threshold
    for i, keyword in enumerate(keywords):
        similarities = cosine_similarities[i].cpu().numpy()
        matched_indices = [j for j, similarity in enumerate(similarities) if similarity >= similarity_threshold]
        matched_products_count = len(matched_indices)
        df_ngrams.loc[df_ngrams['Keyword'] == keyword, 'matching_products_semantic'] = matched_products_count

    df_filtered_ngrams = df_ngrams[df_ngrams['matching_products_semantic'] >= MIN_MATCHING_PRODUCTS]

    return df_filtered_ngrams


# Apply the semantic similarity calculation
df_ngrams_with_semantic_similarity = calculate_semantic_similarity(df_ngrams, product)

# Merge keywords into category
category_with_semantic_match_keywords = merge_keywords_into_category(df_ngrams_with_semantic_similarity, category)

# Save the result
category_with_semantic_match_keywords.to_csv("/python_scripts/category_with_semantic_match_keywords.csv")
