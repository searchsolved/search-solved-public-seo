import streamlit as st
import pandas as pd
import numpy as np
import chardet
from polyfuzz import PolyFuzz
from io import BytesIO
import base64


def read_csv_with_encoding(file, dtype):
    # Detecting encoding
    result = chardet.detect(file.getvalue())
    encoding = result['encoding']

    # Reading CSV with detected encoding
    return pd.read_csv(file, dtype=dtype, encoding=encoding, on_bad_lines='skip')


def get_table_download_link(df, filename):
    """
    Generates a link allowing the processed dataframe to be downloaded.
    """
    towrite = BytesIO()
    df.to_csv(towrite, index=False, encoding='utf-8-sig')
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download the Migration File</a>'
    return href


def process_files(df_live, df_staging, matching_columns, progress_bar, message_placeholder, selected_additional_columns):
    # Convert to lowercase for case-insensitive matching
    df_live = df_live.apply(lambda col: col.str.lower())
    df_staging = df_staging.apply(lambda col: col.str.lower())

    # Create a PolyFuzz model
    model = PolyFuzz("TF-IDF")

    # Dictionary to store match and score data for each column
    matches_scores = {}

    # Function to match and score each column
    def match_and_score(col):
        live_list = df_live[col].fillna('').tolist()
        staging_list = df_staging[col].fillna('').tolist()

        if live_list and staging_list:
            model.match(live_list, staging_list)
            matches = model.get_matches()
            matches_scores[col] = matches
            return matches
        else:
            return pd.DataFrame(columns=['From', 'To', 'Similarity'])

    # Match each column and collect scores
    for index, col in enumerate(matching_columns):
        match_and_score(col)
        progress = (index + 1) / len(matching_columns)
        progress_bar.progress(progress)
        message_placeholder.info('Finalising the processing. Please Wait!')

    # Function to find the overall best match for each row and calculate row-wise median match score
    def find_best_overall_match_and_median(row):
        similarities = []
        best_match_info = {
            'Best Match on': None,
            'Highest Matching URL': None,
            'Highest Similarity Score': 0,
            'Best Match Content': None
        }

        for col in matching_columns:
            matches = matches_scores.get(col, pd.DataFrame())
            if not matches.empty:
                match_row = matches.loc[matches['From'] == row[col]]
                if not match_row.empty:
                    similarity_score = match_row.iloc[0]['Similarity']
                    similarities.append(similarity_score)
                    if similarity_score > best_match_info['Highest Similarity Score']:
                        best_match_info.update({
                            'Best Match on': col,
                            'Highest Matching URL': df_staging.loc[df_staging[col] == match_row.iloc[0]['To'], 'Address'].values[0],
                            'Highest Similarity Score': similarity_score,
                            'Best Match Content': match_row.iloc[0]['To']
                        })

        # Adding user-selected additional columns from staging dataframe
        for additional_col in selected_additional_columns:
            if additional_col in df_staging.columns:
                staging_value = df_staging.loc[df_staging['Address'] == best_match_info['Highest Matching URL'], additional_col].values
                best_match_info[f'Staging {additional_col}'] = staging_value[0] if staging_value.size > 0 else None

        # Calculate the median similarity score for the row
        best_match_info['Median Match Score'] = np.median(similarities) if similarities else None

        return pd.Series(best_match_info)

    # Apply the function to find the best overall match and calculate row-wise median match score
    match_results = df_live.apply(find_best_overall_match_and_median, axis=1)

    # Concatenate the match results with the original dataframe
    final_columns = ['Address'] + [col for col in matching_columns if col != 'Address']
    df_final = pd.concat([df_live[final_columns], match_results], axis=1)

    # Generate and display the download link for the final DataFrame
    download_link = get_table_download_link(df_final, 'migration_mapping_data.csv')
    st.markdown(download_link, unsafe_allow_html=True)
    st.balloons()

    return df_final


def main():
    st.set_page_config(page_title="Automatic Website Migration Tool | LeeFoot.co.uk", layout="wide")

    st.title("Automatic Website Migration Tool")
    st.markdown("### Effortlessly migrate your website data")

    # Instructions Expander
    with st.expander("How to Use This Tool"):
        st.write("""
            - Crawl both the staging and live Websites using Screaming Frog SEO Spider.
            - Export the HTML as CSV Files.
            - Upload your 'Live' and 'Staging' CSV files using the file uploaders below.
            - By Default the app looks for columns named 'Address' 'H1-1' and 'Title 1' but they can be manually mapped if not found.
            - Select up to 3 columns that you want to match.
            - Click the 'Process Files' button to start the matching process.
            - Once processed, a download link for the output file will be provided.
        """)

    col1, col2 = st.columns(2)

    with col1:
        file_live = st.file_uploader("Upload Live CSV", type=['csv'])

    with col2:
        file_staging = st.file_uploader("Upload Staging CSV", type=['csv'])

    # Check for identical file uploads
    if file_live and file_staging:
        if file_live.getvalue() == file_staging.getvalue():
            st.warning(
                "Warning: The same file has been uploaded for both live and staging. Please upload different files.")
        else:
            df_live = read_csv_with_encoding(file_live, "str")
            df_staging = read_csv_with_encoding(file_staging, "str")

            # Check for empty dataframes
            if df_live.empty or df_staging.empty:
                st.warning("Warning: One or both of the uploaded files are empty.")
            else:
                common_columns = list(set(df_live.columns) & set(df_staging.columns))

                # Check for 'Address', 'URL', or 'url' in the common columns for default address column
                address_defaults = ['Address', 'URL', 'url']
                default_address_column = next((col for col in address_defaults if col in common_columns),
                                              common_columns[0])

                st.write("Select the column to use as 'Address':")
                address_column = st.selectbox("Address Column", common_columns,
                                              index=common_columns.index(default_address_column))

                # Filter out the selected 'Address' column from additional matching columns
                additional_columns = [col for col in common_columns if col != address_column]
                default_additional_columns = ['H1-1', 'Title 1']
                default_selection = [col for col in default_additional_columns if col in additional_columns]

                st.write("Select additional columns to match (optional, max 2):")
                max_additional_columns = min(2, len(additional_columns))
                selected_additional_columns = st.multiselect("Additional Columns", additional_columns,
                                                             default=default_selection[:max_additional_columns],
                                                             max_selections=max_additional_columns)

                if st.button("Process Files"):
                    # Use st.empty to hold the place of the message
                    message_placeholder = st.empty()
                    message_placeholder.info('Matching Columns, Please Wait!')

                    # Rename the user-selected 'Address' column to 'Address'
                    df_live.rename(columns={address_column: 'Address'}, inplace=True)
                    df_staging.rename(columns={address_column: 'Address'}, inplace=True)

                    all_selected_columns = ['Address'] + selected_additional_columns
                    progress_bar = st.progress(0)

                    # Pass message_placeholder to process_files
                    df_final = process_files(df_live, df_staging, all_selected_columns, progress_bar, message_placeholder, selected_additional_columns)

    # Enhanced Footer
    st.markdown("""
        <hr style="height:2px;border-width:0;color:gray;background-color:gray">
        <p style="font-style: italic;">Created by <a href="https://twitter.com/LeeFootSEO" target="_blank">LeeFootSEO</a> | <a href="https://leefoot.co.uk" target="_blank">Website</a></p>
        <p style="font-style: italic;">Need an app? <a href="mailto:hello@leefoot.co.uk">Hire Me!</a></p>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
