import streamlit as st
import pandas as pd
import chardet
from polyfuzz import PolyFuzz
from io import BytesIO
import base64
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from polyfuzz.models import TFIDF, EditDistance
import xlsxwriter



# Function to dynamically import and create model
def create_polyfuzz_model(selected_model="TF-IDF"):
    if selected_model == "Edit Distance":
        from polyfuzz.models import EditDistance
        model = EditDistance()
    elif selected_model == "RapidFuzz":
        from polyfuzz.models import RapidFuzz
        model = RapidFuzz()
    else:  # Default to TF-IDF
        from polyfuzz.models import TFIDF
        model = TFIDF(min_similarity=0)
    return model


def read_excel(file, dtype):
    return pd.read_excel(file, dtype=dtype)


# Function Definitions
def read_csv_with_encoding(file, dtype):
    result = chardet.detect(file.getvalue())
    encoding = result['encoding']
    return pd.read_csv(file, dtype=dtype, encoding=encoding, on_bad_lines='skip')


def prepare_score_distribution_data(df_final):
    # Assuming 'Median Match Score' is already calculated and needs to be scaled to a 0-100 scale.
    df_final['Median Match Score Scaled'] = df_final['Median Match Score'] * 100
    bins = range(0, 110, 10)
    labels = [f'{i}-{i + 10}' for i in range(0, 100, 10)]
    df_final['Score Bracket'] = pd.cut(df_final['Median Match Score Scaled'], bins=bins, labels=labels, include_lowest=True)
    score_brackets = df_final['Score Bracket'].value_counts().sort_index().reindex(labels, fill_value=0)

    score_data = pd.DataFrame({
        'Score Bracket': score_brackets.index,
        'URL Count': score_brackets.values
    })
    return score_data


def create_and_write_excel(df, score_data, filename):
    excel_writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(excel_writer, sheet_name='Sheet1', index=False)
    score_data.to_excel(excel_writer, sheet_name='Median Score Distribution', index=False)
    return excel_writer


def format_excel_sheets(excel_writer, df):
    workbook = excel_writer.book
    worksheet1 = excel_writer.sheets['Sheet1']
    worksheet2 = excel_writer.sheets['Median Score Distribution']

    left_align_format = workbook.add_format({'align': 'left'})
    percentage_format = workbook.add_format({'num_format': '0.00%', 'align': 'left'})

    num_rows = len(df)
    num_cols = len(df.columns)
    worksheet1.add_table(0, 0, num_rows, num_cols - 1, {'columns': [{'header': col} for col in df.columns]})
    worksheet1.freeze_panes(1, 0)

    max_col_width = 80
    for i, col in enumerate(df.columns):
        col_width = max(len(col), max(df[col].astype(str).apply(len).max(), 10)) + 2
        col_width = min(col_width, max_col_width)
        if i == 4 or i == 7:
            worksheet1.set_column(i, i, col_width, percentage_format)
            worksheet1.conditional_format(1, i, num_rows, i, {'type': '3_color_scale'})
        else:
            worksheet1.set_column(i, i, col_width, left_align_format)


def insert_chart_into_excel(excel_writer, score_data):
    workbook = excel_writer.book
    worksheet2 = excel_writer.sheets['Median Score Distribution']
    chart = workbook.add_chart({'type': 'column'})
    max_row = len(score_data) + 1

    chart.add_series({
        'name': '=Median Score Distribution!$B$1',
        'categories': f'=Median Score Distribution!$A$2:$A${max_row}',
        'values': f'=Median Score Distribution!$B$2:$B${max_row}',
    })

    chart.set_title({'name': 'Distribution of Median Match Scores'})
    chart.set_x_axis({'name': 'Median Match Score Brackets'})
    chart.set_y_axis({'name': 'URL Count'})
    worksheet2.insert_chart('D2', chart)


def generate_download_link(filename):
    with open(filename, 'rb') as file:
        b64 = base64.b64encode(file.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Click here to download {filename}</a>'



def get_table_download_link(df, filename, score_data):
    excel_writer = create_and_write_excel(df, score_data, filename)
    format_excel_sheets(excel_writer, df)
    insert_chart_into_excel(excel_writer, score_data)
    excel_writer.close()

    download_link = generate_download_link(filename)
    st.markdown(download_link, unsafe_allow_html=True)


def lowercase_dataframe(df):
    return df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)


def match_and_score_columns(selected_model, df_live, df_staging, matching_columns):
    matches_scores = {}

    # Initialize the model based on the selected model
    if selected_model == "Edit Distance":
        model = PolyFuzz(EditDistance())
    elif selected_model == "RapidFuzz":
        model = PolyFuzz(RapidFuzz())
    else:  # Default to TF-IDF
        model = PolyFuzz(TFIDF())

    for col in matching_columns:
        live_list = df_live[col].fillna('').tolist()
        staging_list = df_staging[col].fillna('').tolist()
        if live_list and staging_list:
            model.match(live_list, staging_list)
            matches = model.get_matches()
            matches_scores[col] = matches

    return matches_scores


def find_best_match_and_median(df_live, df_staging, matches_scores, matching_columns, selected_additional_columns):
    def find_best_overall_match_and_median(row):
        similarities = []
        individual_match_scores = []  # Store All Column Match Scores
        best_match_info = {'Best Match on': None, 'Highest Matching URL': None, 'Highest Similarity Score': 0,
                           'Best Match Content': None}
        for col in matching_columns:
            matches = matches_scores.get(col, pd.DataFrame())
            if not matches.empty:
                match_row = matches.loc[matches['From'] == row[col]]
                if not match_row.empty:
                    similarity_score = match_row.iloc[0]['Similarity']
                    similarities.append(similarity_score)
                    individual_match_scores.append((col, round(similarity_score, 2)))  # Add individual match score
                    if similarity_score > best_match_info['Highest Similarity Score']:
                        best_match_info.update({
                            'Best Match on': col,
                            'Highest Matching URL':
                                df_staging.loc[df_staging[col] == match_row.iloc[0]['To'], 'Address'].values[0],
                            'Highest Similarity Score': similarity_score,
                            'Best Match Content': match_row.iloc[0]['To']
                        })

        for additional_col in selected_additional_columns:
            if additional_col in df_staging.columns:
                staging_value = df_staging.loc[
                    df_staging['Address'] == best_match_info['Highest Matching URL'], additional_col].values
                best_match_info[f'Staging {additional_col}'] = staging_value[0] if staging_value.size > 0 else None

        best_match_info['Median Match Score'] = np.median(similarities) if similarities else None
        best_match_info['All Column Match Scores'] = individual_match_scores  # Store the All Column Match Scores
        return pd.Series(best_match_info)

    return df_live.apply(find_best_overall_match_and_median, axis=1)



def prepare_final_dataframe(df_live, match_results, matching_columns):
    final_columns = ['Address'] + [col for col in matching_columns if col != 'Address']
    final_df = pd.concat([df_live[final_columns], match_results], axis=1)

    # Convert the list of tuples into a string representation
    final_df['All Column Match Scores'] = final_df['All Column Match Scores'].apply(lambda x: str(x) if x is not None else None)

    return final_df


def display_download_link(df_final, filename):
    # Prepare score data without the unwanted columns
    # Assuming that 'Median Match Score' column is still required
    df_for_score_data = df_final.drop(['Median Match Score Scaled', 'Score Bracket'], axis=1, inplace=False, errors='ignore')
    score_data = prepare_score_distribution_data(df_for_score_data)
    get_table_download_link(df_final, 'migration_mapping_data.xlsx', score_data)


def process_files(df_live, df_staging, matching_columns, progress_bar, message_placeholder,
                  selected_additional_columns):
    df_live = lowercase_dataframe(df_live)
    df_staging = lowercase_dataframe(df_staging)

    model = create_polyfuzz_model()
    matches_scores = match_and_score_columns(model, df_live, df_staging, matching_columns)

    for index, _ in enumerate(matching_columns):
        progress = (index + 1) / len(matching_columns)
        progress_bar.progress(progress)

    message_placeholder.info('Finalising the processing. Please Wait!')
    match_results = find_best_match_and_median(df_live, df_staging, matches_scores, matching_columns,
                                               selected_additional_columns)

    df_final = prepare_final_dataframe(df_live, match_results, matching_columns)
    display_download_link(df_final, 'migration_mapping_data')
    plot_median_score_brackets(df_final)
    st.balloons()

    # Update the message after all processing is complete
    message_placeholder.success('Complete!')

    return df_final


def upload_file(column, file_types):
    file_type_label = "/".join(file_types).upper()  # Creating a string like "CSV/XLSX/XLS"
    return st.file_uploader(f"Upload {column} {file_type_label}", type=file_types)


def select_columns(title, options, default_value, max_selections):
    st.write(title)
    return st.multiselect(title, options, default=default_value, max_selections=max_selections)


def display_warning(message):
    st.warning(message)


def rename_column(df, old_name, new_name):
    df.rename(columns={old_name: new_name}, inplace=True)


def display_instructions():
    with st.expander("How to Use This Tool"):
        st.write("""
            - Crawl both the staging and live Websites using Screaming Frog SEO Spider.
            - Export the HTML as CSV Files.
            - Upload your 'Live' and 'Staging' CSV files using the file uploaders below.
            - By Default the app looks for columns named 'Address' 'H1-1' and 'Title 1' but they can be manually mapped if not found.
            - Select up to 3 columns that you want to match.
            - Click the 'Process Files' button to start the matching process.
            - Once processed, a download link for the output file will be provided.
            - Statistic such as median match score and a total mediam similarity score will be shown. Run the script with a different combination of columns to get the best score!
        """)


def create_footer():
    st.markdown("""
        <hr style="height:2px;border-width:0;color:gray;background-color:gray">
        <p style="font-style: italic;">Need an app? Need this run as a managed service? <a href="mailto:hello@leefoot.co.uk">Hire Me!</a></p>
        """, unsafe_allow_html=True)


def initialize_interface():
    st.set_page_config(page_title="Automatic Website Migration Tool | LeeFoot.co.uk", layout="wide")
    st.title("Automatic Website Migration Tool")
    st.markdown("### Effortlessly migrate your website data")

    # Add the creator information here
    st.markdown("""
        <p style="font-style: italic;">Created by <a href="https://twitter.com/LeeFootSEO" target="_blank">LeeFootSEO</a> | <a href="https://leefoot.co.uk" target="_blank">Website</a></p>
        """, unsafe_allow_html=True)

    display_instructions()


def validate_uploads(file1, file2):
    if not file1 or not file2 or file1.getvalue() == file2.getvalue():
        display_warning(
            "Warning: The same file has been uploaded for both live and staging. Please upload different files.")
        return False
    return True


def upload_files():
    col1, col2 = st.columns(2)
    with col1:
        file_live = upload_file("Live", ['csv', 'xlsx', 'xls'])
    with col2:
        file_staging = upload_file("Staging", ['csv', 'xlsx', 'xls'])
    return file_live, file_staging


def process_and_validate_uploads(file_live, file_staging):
    if validate_uploads(file_live, file_staging):
        # Determine file type and read accordingly
        if file_live.name.endswith('.csv'):
            df_live = read_csv_with_encoding(file_live, "str")
        else:  # Excel file
            df_live = read_excel(file_live, "str")

        if file_staging.name.endswith('.csv'):
            df_staging = read_csv_with_encoding(file_staging, "str")
        else:  # Excel file
            df_staging = read_excel(file_staging, "str")

        # Check if dataframes are empty
        if df_live.empty or df_staging.empty:
            display_warning("Warning: One or both of the uploaded files are empty.")
            return None, None
        else:
            return df_live, df_staging
    return None, None


def select_columns_for_matching(df_live, df_staging):
    common_columns = list(set(df_live.columns) & set(df_staging.columns))
    address_defaults = ['Address', 'URL', 'url', 'Adresse', 'Dirección', 'Indirizzo']
    default_address_column = next((col for col in address_defaults if col in common_columns), common_columns[0])

    st.write("Select the column to use as 'Address':")
    address_column = st.selectbox("Address Column", common_columns, index=common_columns.index(default_address_column))

    additional_columns = [col for col in common_columns if col != address_column]
    default_additional_columns = ['H1-1', 'Title 1', 'Titel 1', 'Título 1', 'Titolo 1']
    default_selection = [col for col in default_additional_columns if col in additional_columns]

    st.write("Select additional columns to match (optional, max 3):")
    max_additional_columns = min(3, len(additional_columns))
    # Ensure default selections do not exceed the maximum allowed
    default_selection = default_selection[:max_additional_columns]
    selected_additional_columns = st.multiselect("Additional Columns", additional_columns,
                                                 default=default_selection,
                                                 max_selections=max_additional_columns)
    return address_column, selected_additional_columns


def handle_file_processing(df_live, df_staging, address_column, selected_additional_columns):
    message_placeholder = st.empty()
    message_placeholder.info('Matching Columns, Please Wait!')

    rename_column(df_live, address_column, 'Address')
    rename_column(df_staging, address_column, 'Address')

    all_selected_columns = ['Address'] + selected_additional_columns
    progress_bar = st.progress(0)
    df_final = process_files(df_live, df_staging, all_selected_columns, progress_bar, message_placeholder,
                             selected_additional_columns)
    return df_final


def plot_median_score_brackets(df_final):
    # Scale 'Median Match Score' values to 0-100
    df_final['Median Match Score'] *= 100

    # Define the bins and labels for the histogram
    bins = range(0, 110, 10)
    labels = [f'{i}-{i + 10}' for i in range(0, 100, 10)]

    # Group the scores into brackets
    df_final['Score Bracket'] = pd.cut(df_final['Median Match Score'], bins=bins, labels=labels, include_lowest=True)
    bracket_counts = df_final['Score Bracket'].value_counts().sort_index()

    # Create two columns for plots
    col1, col2 = st.columns(2)

    # First plot in the first column
    with col1:
        plt.figure(figsize=(5, 3))
        ax = bracket_counts.plot(kind='bar', width=0.9)
        ax.set_title('Distribution of Median Match Scores', fontsize=10)
        ax.set_xlabel('Median Match Score Brackets', fontsize=8)
        ax.set_ylabel('URL Count', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
        plt.tight_layout()
        st.pyplot(plt)

    # Second plot (Plotly indicator) in the second column
    with col2:
        plot_indicator_chart(df_final)


def plot_indicator_chart(df_final):
    # Calculate the median of the 'Highest Similarity Score'
    median_similarity_score = df_final['Highest Similarity Score'].median()

    # Check if there's a previous score stored in session state
    if 'previous_score' in st.session_state:
        reference_value = st.session_state['previous_score']
    else:
        # If not, use the current score as reference (delta will be zero)
        reference_value = median_similarity_score

    # Create the Plotly figure
    fig = go.Figure()

    # Add the indicator trace using the median_similarity_score
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=median_similarity_score,
        delta={'reference': reference_value, 'relative': False, 'valueformat': '.2%'},
        number={'valueformat': '.2%', 'font': {'color': 'black'}},
        title={'text': "Total Median Similarity Score for Selected Columns", 'font': {'color': 'black'}},
        domain={'row': 0, 'column': 0}))

    # Update the layout of the figure
    fig.update_layout(
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"}
    )

    st.plotly_chart(fig)

    # Update the session state with the new score
    st.session_state['previous_score'] = median_similarity_score


def main():
    initialize_interface()

    # Advanced settings expander for model selection
    with st.expander("Advanced Settings"):
        model_options = ['TF-IDF', 'Edit Distance', 'RapidFuzz']
        selected_model = st.selectbox("Select Matching Model", model_options)

        if selected_model == "TF-IDF":
            st.write("Use TF-IDF for comprehensive text analysis, suitable for most use cases.")
        elif selected_model == "Edit Distance":
            st.write(
                "Edit Distance is useful for matching based on character-level differences, such as small text variations.")
        elif selected_model == "RapidFuzz":
            st.write("RapidFuzz is efficient for large datasets, offering fast and approximate string matching.")

    file_live, file_staging = upload_files()
    if file_live and file_staging:
        df_live, df_staging = process_and_validate_uploads(file_live, file_staging)
        if df_live is not None and df_staging is not None:
            address_column, selected_additional_columns = select_columns_for_matching(df_live, df_staging)
            if st.button("Process Files"):
                df_final = handle_file_processing(df_live, df_staging, address_column, selected_additional_columns)
                score_data = prepare_score_distribution_data(df_final)
                get_table_download_link(df_final, 'migration_mapping_data.xlsx', score_data)

    create_footer()


if __name__ == "__main__":
    main()
