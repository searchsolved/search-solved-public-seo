import streamlit as st
import pandas as pd
import chardet
from polyfuzz import PolyFuzz
from io import BytesIO
import base64
import plotly.graph_objects as go
import urllib.parse


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


def clean_name(url_part):
    return url_part.replace('/', ' ').replace('-', ' ').strip()


def extract_hierarchy_levels(url):
    parsed_url = urllib.parse.urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")

    # Exclude files (e.g., .html pages) from the path parts
    if path_parts:
        if '.' in path_parts[-1]:  # Checks if the last part looks like a file
            path_parts = path_parts[:-1]  # Remove the last part if it's a file

    if not path_parts:  # If the URL path is empty or only had a file, it's the homepage
        return ['No Path']

    # Create a hierarchy list with cleaned folder names
    hierarchy_levels = [clean_name(part) for part in path_parts]
    return hierarchy_levels


def prepare_sankey_data(df, top_x=20):
    df['Source Hierarchy'] = df['Address'].apply(extract_hierarchy_levels)
    df['Target Hierarchy'] = df['Highest Matching URL'].apply(extract_hierarchy_levels)

    # Flatten the hierarchy to create source-target pairs for all levels
    rows = []
    for _, row in df.iterrows():
        source_hierarchy = row['Source Hierarchy']
        target_hierarchy = row['Target Hierarchy']

        # Connect every source level with all subsequent levels
        for i in range(len(source_hierarchy)):
            for j in range(i + 1, len(source_hierarchy)):
                rows.append({
                    'Source Level': source_hierarchy[i],
                    'Target Level': source_hierarchy[j]
                })

        # Apply the same logic for the target hierarchy
        for i in range(len(target_hierarchy)):
            for j in range(i + 1, len(target_hierarchy)):
                rows.append({
                    'Source Level': target_hierarchy[i],
                    'Target Level': target_hierarchy[j]
                })
    sankey_data = pd.DataFrame(rows)

    # Aggregate and count the source-target pairs
    sankey_data = sankey_data.groupby(['Source Level', 'Target Level']).size().reset_index(name='Count')

    # Get top mappings by count for Sankey Chart
    top_mappings = sankey_data.nlargest(top_x, 'Count')
    return top_mappings


def create_sankey_chart(sankey_data):
    # Generate labels for each unique level in the hierarchy
    labels = sorted(set(
        label for pair in zip(sankey_data['Source Level'], sankey_data['Target Level']) for label in pair
    ), key=lambda x: ('No Path' not in x, x))
    level_index = {level: idx for idx, level in enumerate(labels)}

    # Map levels to indices using the cleaned names
    source_indices = [level_index[level] for level in sankey_data['Source Level']]
    target_indices = [level_index[level] for level in sankey_data['Target Level']]
    weights = sankey_data['Count']

    # Create Sankey diagram with hierarchical data
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,  # Use the cleaned labels for node names
            color="blue"
        ),
        link=dict(
            source=target_indices,
            target=source_indices,
            value=weights
        ),
        # Set the layout for node alignment
        arrangement='snap'
    )])

    fig.update_layout(title_text='Top 20 Folder Mappings')
    return fig


def process_files(df_live, df_staging, matching_columns, progress_bar, message_placeholder):
    # Convert to lowercase for case-insensitive matching
    df_live = df_live.apply(lambda col: col.str.lower())
    df_staging = df_staging.apply(lambda col: col.str.lower())

    # Create a PolyFuzz model
    model = PolyFuzz("TF-IDF")

    # Function to match and score each column
    def match_and_score(col):
        live_list = df_live[col].fillna('').tolist()
        staging_list = df_staging[col].fillna('').tolist()

        if live_list and staging_list:
            model.match(live_list, staging_list)
            return model.get_matches()
        else:
            return pd.DataFrame(columns=['From', 'To', 'Similarity'])

    # Match each column and collect scores
    matches_scores = {}
    total_columns = len(matching_columns)
    for index, col in enumerate(matching_columns):
        matches_scores[col] = match_and_score(col)
        progress = (index + 1) / total_columns
        progress_bar.progress(progress)
        message_placeholder.info('Finalising the processing. Please Wait!')

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
                if not match_row.empty and match_row.iloc[0]['Similarity'] > best_match_info[
                    'Highest Similarity Score']:
                    best_match_info['Best Match on'] = col
                    best_match_info['Highest Matching URL'] = df_staging.loc[
                        df_staging[col] == match_row.iloc[0]['To'], 'Address'
                    ].values[0]
                    best_match_info['Highest Similarity Score'] = match_row.iloc[0]['Similarity']

        return pd.Series(best_match_info)

    # Apply the function to find the best overall match
    match_results = df_live.apply(find_best_overall_match, axis=1)

    # Concatenate the match results with the original dataframe
    final_columns = ['Address'] + [col for col in matching_columns if col != 'Address']
    df_final = pd.concat([df_live[final_columns], match_results], axis=1)

    # Drop 'Source Hierarchy' and 'Target Hierarchy' columns if they exist
    df_final.drop(columns=['Source Hierarchy', 'Target Hierarchy'], errors='ignore', inplace=True)

    # Generate and display the download link before creating the Sankey chart
    download_link = get_table_download_link(df_final, 'migration_mapping_data.csv')
    st.markdown(download_link, unsafe_allow_html=True)

    # Prepare and create the Sankey chart
    sankey_data = prepare_sankey_data(df_final)
    sankey_chart = create_sankey_chart(sankey_data)
    message_placeholder.empty()
    st.plotly_chart(sankey_chart, use_container_width=True)

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
                    df_final = process_files(df_live, df_staging, all_selected_columns, progress_bar,
                                             message_placeholder)

                    # After the sankey chart is displayed, clear the message
                    message_placeholder.empty()
                    st.balloons()

    # Enhanced Footer
    st.markdown("""
        <hr style="height:2px;border-width:0;color:gray;background-color:gray">
        <p style="font-style: italic;">Created by <a href="https://twitter.com/LeeFootSEO" target="_blank">LeeFootSEO</a> | <a href="https://leefoot.co.uk" target="_blank">Website</a></p>
        <p style="font-style: italic;">Need an app? <a href="mailto:hello@leefoot.co.uk">Hire Me!</a></p>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
