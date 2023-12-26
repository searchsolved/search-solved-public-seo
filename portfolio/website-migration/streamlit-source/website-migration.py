import re
import io

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import chardet
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

MAX_ROWS = 15_000  # beta limit max rows

# LeeFootSEO | https://leefoot.co.uk |26th December 2023

# -------------
# Streamlit UI and Setup Functions
# -------------


if 'df_clustered' not in st.session_state:
    st.session_state['df_clustered'] = None


def configure_streamlit_page():
    st.set_page_config(page_title="✨ Semantic Keyword Clustering Tool | LeeFoot.co.uk", layout="wide")
    st.title("✨ Semantic Keyword Clustering Tool | Dec 23")
    st.markdown("### Cluster your data with Sentence Transformers. (Max 10,000 Rows)")

    st.markdown(
        """
        <p>
            Created by <a href="https://twitter.com/LeeFootSEO" target="_blank">LeeFootSEO</a> |
            <a href="https://leefoot.co.uk" target="_blank">More Apps & Scripts on my Website</a>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Create an expander for instructions
    with st.expander("Instructions"):
        st.markdown("""
            Follow these steps to use the tool:
            1) **Upload a keyword file** in CSV or Excel format.
            2) **Select a keyword column**. Common column names are automatically mapped.
            3) **Choose a sentence transformer**. The default option is the fastest transformer.
            4) **Adjust chart settings** as needed for better display of your data.
            5) **Download your file** once the clustering is completed.
            6) **Output** Clusters are appended to your original data.
        """)


# -------------
# Configuration Functions
# -------------

def configure_sidebar_for_clustering():
    min_cluster_size = st.sidebar.slider("Minimum Cluster Size", min_value=2, max_value=10, value=2,
                                         help="Select the minimum number of elements in a cluster.",
                                         key="min_cluster_size_slider")
    return min_cluster_size


def configure_transformer_model():
    st.sidebar.header("Transformer Model Selection")
    transformer_info = {
        "paraphrase-MiniLM-L3-v2": "Semantic Score: 39.19, Speed: 19,000",
        "multi-qa-MiniLM-L6-cos-v1": "Semantic Score: 51.83, Speed: 14,200",
        "paraphrase-multilingual-MiniLM-L12-v2": "Semantic Score: 39.19, Speed: 7,500",
        "all-distilroberta-v1": "Semantic Score: 50.94, Speed: 4,000",
        "all-mpnet-base-v2": "Semantic Score: 57.01, Speed: 2,800"
    }

    model = st.sidebar.radio(
        "Select Transformer Model",
        options=list(transformer_info.keys()),
        help="Select the transformer model for clustering."
    )
    st.sidebar.caption(f"Details - {transformer_info[model]}")
    return model


def configure_cluster_accuracy():
    accuracy = st.sidebar.slider("Cluster accuracy (0-100)", min_value=0, max_value=100, value=70,
                                 help="Adjust the accuracy of the clusters.")
    return accuracy / 100


def configure_minimum_cluster_size():
    min_cluster_size = st.sidebar.slider("Minimum Cluster Size", min_value=2, max_value=10, value=2,
                                         help="Select the minimum number of elements in a cluster.")
    return min_cluster_size


def configure_duplicate_removal_and_tagging():
    remove_duplicates = st.sidebar.checkbox("Remove duplicate keywords?", value=True)
    tag_questions = st.sidebar.checkbox("Tag question keywords?", value=True)
    return remove_duplicates, tag_questions


def configure_visualisation_options():
    st.sidebar.header("Chart Options")
    chart_type = st.sidebar.radio(
        "Choose chart type for cluster distribution",
        ["Bar Chart", "Pie Chart"],
        help="Select the type of chart for visualizing cluster distribution."
    )
    top_clusters = st.sidebar.slider("Number of top clusters to display", 5, 100, 25)
    exclude_no_cluster = st.sidebar.checkbox("Exclude Unclustered Results from Chart", True)
    return chart_type, top_clusters, exclude_no_cluster


# -------------
# File Upload and Preprocessing
# -------------

def upload_file():
    return st.file_uploader("Upload file", type=["csv", "xls", "xlsx"])


def handle_file_upload():
    uploaded_file = upload_file()
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = read_csv_with_detected_encoding(uploaded_file, detect_csv_file_encoding(uploaded_file), MAX_ROWS)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = read_excel_file(uploaded_file, MAX_ROWS)
        return df
    return None


def detect_csv_file_encoding(uploaded_file):
    result = chardet.detect(uploaded_file.getvalue())
    return result['encoding']


def read_csv_with_detected_encoding(uploaded_file, encoding_value, max_rows):
    df = pd.read_csv(
        uploaded_file,
        encoding=encoding_value,
        on_bad_lines='skip',
        nrows=max_rows + 1  # Read one extra row to check if it exceeds MAX_ROWS
    )
    if len(df) > max_rows:
        st.toast(f"File truncated to the first {MAX_ROWS} rows.", icon="⚠️")
        return df.head(max_rows)
    return df


def read_excel_file(uploaded_file, max_rows):
    df = pd.read_excel(uploaded_file, nrows=max_rows + 1)  # Read one extra row to check if it exceeds MAX_ROWS
    if len(df) > max_rows:
        st.toast(f"File truncated to the first {MAX_ROWS} rows.", icon="⚠️")
        return df.head(max_rows)
    return df


def limit_dataframe_rows(df, max_rows):
    if len(df) > max_rows:
        return df[:max_rows]
    return df


def select_column_for_clustering(df):
    likely_keyword_columns = ['Keyword', 'keyword', 'keywords', 'Keywords', 'query', 'queries', 'Query', 'Queries']
    default_column = next((col for col in likely_keyword_columns if col in df.columns), df.columns[0])
    return st.selectbox("Select the keyword column", df.columns, index=df.columns.get_loc(default_column))


# -------------
# Clustering and Data Transformation Functions
# -------------

@st.cache_resource()
def load_sentence_transformer_model(model_name):
    model = SentenceTransformer(model_name)
    return model


def load_model(model_name):
    return load_sentence_transformer_model(model_name)


def encode_sentences(model, sentences):
    corpus_embeddings = model.encode(sentences, batch_size=256, show_progress_bar=True)
    return torch.tensor(corpus_embeddings)


def detect_communities(embeddings, threshold):
    return util.community_detection(embeddings, threshold=threshold)


def map_clusters_to_sentences(sentences, clusters):
    cluster_mapping = {}
    for i, cluster in enumerate(clusters):
        for sentence_id in cluster:
            cluster_mapping[sentences[sentence_id]] = f"Cluster {i + 1}"
    return cluster_mapping


def assign_clusters_to_dataframe(df, column, cluster_mapping):
    df_clustered = df.copy()
    df_clustered['cluster'] = df_clustered[column].map(cluster_mapping)
    return df_clustered


def remove_duplicate_rows(df, column):
    return df.drop_duplicates(subset=column, inplace=False)


def cluster_keywords(df, selected_column, model_name, min_similarity, min_cluster_size, remove_duplicates_flag=True):
    if not is_valid_dataframe(df):
        return None

    with st.spinner('Clustering in progress...'):
        if remove_duplicates_flag:
            df = remove_duplicate_rows(df, selected_column)

        model = load_model(model_name)
        sentences = df[selected_column].tolist()
        embeddings = encode_sentences(model, sentences)

        # Adjust community_detection to use min_cluster_size
        clusters = util.community_detection(embeddings, min_community_size=min_cluster_size, threshold=min_similarity)

        cluster_mapping = map_clusters_to_sentences(sentences, clusters)
        return assign_clusters_to_dataframe(df, selected_column, cluster_mapping)


def initiate_clustering(df, model_radio_button, min_similarity, min_cluster_size, remove_duplicates, tag_questions):
    if df is not None:
        selected_column = select_column_for_clustering(df)
        if st.button("Cluster Keywords") or 'df_clustered' not in st.session_state:
            return process_clustering(df, selected_column, model_radio_button, min_similarity, min_cluster_size,
                                      remove_duplicates, tag_questions)
    return None


def process_clustering(df, selected_column, model_radio_button, min_similarity, min_cluster_size, remove_duplicates,
                       tag_questions):
    clustered_df = cluster_keywords(df, selected_column, model_radio_button, min_similarity, min_cluster_size,
                                    remove_duplicates)
    if clustered_df is not None:
        clustered_df = rename_clusters_to_shortest_keywords(clustered_df, selected_column)
        if tag_questions:
            clustered_df = tag_question_keywords(clustered_df, selected_column)
        st.session_state['df_clustered'] = finalise_clustered_dataframe(clustered_df)
    return clustered_df


def is_valid_dataframe(df):
    return df is not None and not df.empty


def rename_clusters_to_shortest_keywords(df, column_name):
    df['keyword_len'] = df[column_name].str.len()
    df = df.sort_values(by="keyword_len", ascending=True)
    df['cluster'] = df.groupby('cluster')[column_name].transform('first')
    return df


# -------------
# Question Extraction
# -------------

# Regex for detecting common question words
question_regex = re.compile(r'^(who|what|where|when|why|how)\b', re.IGNORECASE)


def tag_question_keywords(df, column_name='keyword'):
    """
    Tags each keyword in the DataFrame as a question or not based on a regex match.
    Enforces the column data to be string type.
    :param df: DataFrame containing the keywords.
    :param column_name: Name of the column in the DataFrame to apply the regex on.
    :return: DataFrame with an additional 'is_question' column.
    """
    df[column_name] = df[column_name].astype(str)  # Enforce string type
    df['is_question'] = df[column_name].apply(lambda x: bool(question_regex.match(x)))
    return df


# -------------
# Finalisation and Save Files
# -------------

def finalise_clustered_dataframe(df):
    cluster_col = df.pop('cluster')
    cluster_col.fillna('no cluster', inplace=True)  # Fill NaN values with 'No Cluster'
    df.insert(0, 'cluster', cluster_col)

    if 'keyword_len' in df.columns:
        df = df.drop(columns=['keyword_len'])

    df = df.sort_values(by='cluster', ascending=True)

    return df


def convert_dataframe_to_excel(df):
    """
    Converts a DataFrame into an Excel format with two worksheets:
    one for the entire DataFrame and another for rows where 'is_question' is True.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Export the entire DataFrame to one sheet
        df.to_excel(writer, index=False, sheet_name='Clustered Keywords', startrow=1, header=False)
        worksheet = writer.sheets['Clustered Keywords']
        # Freeze the top row
        worksheet.freeze_panes(1, 0)

        # Add a table
        (max_row, max_col) = df.shape
        column_settings = [{'header': column} for column in df.columns]
        worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

        # Filter the DataFrame for rows where 'is_question' is True and export to another sheet
        questions_df = df[df['is_question'] == True]
        questions_df.to_excel(writer, index=False, sheet_name='Questions', startrow=1, header=False)
        questions_worksheet = writer.sheets['Questions']
        # Freeze the top row
        questions_worksheet.freeze_panes(1, 0)

        # Add a table
        (max_row, max_col) = questions_df.shape
        column_settings = [{'header': column} for column in questions_df.columns]
        questions_worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

    return output.getvalue()


def export_clustered_keywords_to_excel(df, key):
    """
    Provides a button in the Streamlit app to download the given DataFrame as an Excel file
    with an additional worksheet for questions.
    :param df: DataFrame to be exported.
    :param key: Unique key for the download button.
    """
    if df is not None and not df.empty:
        st.success("All keywords clustered successfully.")
        st.markdown("### **🎈 Download an Excel Export with Questions Worksheet!**")

        excel_data = convert_dataframe_to_excel(df)
        st.download_button(
            label="📥 Download your report!",
            data=excel_data,
            file_name="clustered_keywords_with_questions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key  # Unique key for each button
        )
    else:
        st.error("No data available to export.")


# -------------
# Visualisation
# -------------

def plot_cluster_distribution(df_clustered, chart_type, top_clusters, tag_questions, exclude_no_cluster):
    df_filtered = df_clustered[df_clustered['cluster'] != 'no cluster'] if exclude_no_cluster else df_clustered
    cluster_counts = df_filtered['cluster'].value_counts().head(top_clusters)

    if chart_type == "Bar Chart":
        ax = cluster_counts.plot(kind='bar', figsize=(10, 6), color='blue', label='Total Keywords')

        if tag_questions and 'is_question' in df_clustered.columns:
            # Count of question keywords per cluster
            question_counts = df_filtered[df_filtered['is_question']]['cluster'].value_counts()
            if not question_counts.empty:
                question_counts.reindex(cluster_counts.index).plot(kind='bar', ax=ax, color='orange', alpha=0.6,
                                                                   label='Question Keywords')

        plt.title('Cluster Size Distribution')
        plt.ylabel('Number of Keywords')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

    elif chart_type == "Pie Chart":
        plt.figure(figsize=(8, 8))
        cluster_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140,
                            colors=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
        plt.title('Overall Cluster Distribution')
        plt.ylabel('')
        plt.tight_layout()
        st.pyplot(plt)


def plot_question_cluster_distribution(df_clustered, chart_type, top_clusters):
    if 'is_question' not in df_clustered.columns:
        st.error("Question keyword tagging is not enabled.")
        return

    # Filter out rows where the cluster value is 'no cluster' or equivalent
    question_clusters = df_clustered[(df_clustered['is_question']) & (df_clustered['cluster'] != 'no cluster')]
    cluster_counts = question_clusters['cluster'].value_counts().head(top_clusters)

    if cluster_counts.empty:
        st.caption("No question keywords found in clusters.")
        return

    if chart_type == "Bar Chart":
        plt.figure(figsize=(10, 6))
        cluster_counts.plot(kind='bar')
    elif chart_type == "Pie Chart":
        plt.figure(figsize=(8, 8))
        cluster_counts.plot(kind='pie', autopct='%1.1f%%')

    plt.title('Question Keyword Cluster Distribution')
    plt.ylabel('Number of Question Keywords')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)


def plot_question_proportion_chart(df_clustered):
    if df_clustered is not None and 'is_question' in df_clustered.columns:
        # Calculate the count of question and non-question keywords per cluster
        cluster_question_counts = df_clustered.groupby(['cluster', 'is_question']).size().unstack(fill_value=0)

        # Normalize the counts to get proportions
        cluster_proportions = cluster_question_counts.div(cluster_question_counts.sum(axis=1), axis=0) * 100

        # Plotting
        cluster_proportions.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Proportion of Question Keywords in Each Cluster')
        plt.ylabel('Percentage (%)')
        plt.xlabel('Cluster')
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)


def display_charts(chart_type, top_clusters, tag_questions, exclude_no_cluster):
    if 'df_clustered' in st.session_state and st.session_state['df_clustered'] is not None:
        export_clustered_keywords_to_excel(st.session_state['df_clustered'], key="download_button")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Overall Cluster Distribution")
            plot_cluster_distribution(st.session_state['df_clustered'], chart_type, top_clusters, tag_questions,
                                      exclude_no_cluster)

        with col2:
            st.markdown("### Question Cluster Distribution")
            if tag_questions:
                plot_question_cluster_distribution(st.session_state['df_clustered'], chart_type, top_clusters)
            else:
                st.error("Question keyword tagging is not enabled.")


# -------------
# Main Function
# -------------

def main():
    configure_streamlit_page()
    model_radio_button = configure_transformer_model()
    min_similarity = configure_cluster_accuracy()
    remove_duplicates, tag_questions = configure_duplicate_removal_and_tagging()
    chart_type, top_clusters, exclude_no_cluster = configure_visualisation_options()

    df = handle_file_upload()
    if df is not None:
        min_cluster_size = configure_minimum_cluster_size()
        initiate_clustering(df, model_radio_button, min_similarity, min_cluster_size, remove_duplicates, tag_questions)

    display_charts(chart_type, top_clusters, tag_questions, exclude_no_cluster)


if __name__ == "__main__":
    main()
