import base64
import chardet
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF, EditDistance, RapidFuzz
import plotly.graph_objects as go
import xlsxwriter

# LeeFootSEO | https://leefoot.co.uk | 10th December 2023


# Streamlit Interface Setup and Utilities ------------------------------------------------------------------------------

def setup_streamlit_interface():
    """
    Sets up the Streamlit interface for the Automatic Website Migration Tool.
    Configures the page layout, title, and adds creator information and instructions.
    """
    st.set_page_config(page_title="Automatic Website Migration Tool | LeeFoot.co.uk", layout="wide")
    st.title("Automatic Website Migration Tool")
    st.markdown("### Effortlessly migrate your website data")

    st.markdown(
        """
        <p style="font-style: italic;">
            Created by <a href="https://twitter.com/LeeFootSEO" target="_blank">LeeFootSEO</a> |
            <a href="https://leefoot.co.uk" target="_blank">Website</a>
        </p>
        """,
        unsafe_allow_html=True
    )

    show_instructions_expander()


def create_file_uploader_widget(column, file_types):
    """
    Creates a file uploader widget in Streamlit.

    Args:
    column (str): A label indicating the type of file to be uploaded (e.g., "Live", "Staging").
    file_types (list): A list of acceptable file types for upload (e.g., ['csv', 'xlsx', 'xls']).

    Returns:
    streamlit.file_uploader: The file uploader widget.
    """
    file_type_label = "/".join(file_types).upper()  # Creating a string like "CSV/XLSX/XLS"
    return st.file_uploader(f"Upload {column} {file_type_label}", type=file_types)


def select_columns_for_data_matching(title, options, default_value, max_selections):
    """
    Creates a multi-select widget in Streamlit for selecting columns for data matching.

    Args:
    title (str): The title of the widget.
    options (list): A list of options (columns) to choose from.
    default_value (list): Default selected values.
    max_selections (int): Maximum number of selections allowed.

    Returns:
    list: A list of selected options.
    """
    st.write(title)
    return st.multiselect(title, options, default=default_value, max_selections=max_selections)


def show_warning_message(message):
    """
    Displays a warning message in the Streamlit interface.

    Args:
    message (str): The warning message to display.
    """
    st.warning(message)


def show_instructions_expander():
    """
    Creates an expander in the Streamlit interface to display instructions on how to use the tool.
    """
    instructions = (
        "- Crawl both the staging and live Websites using Screaming Frog SEO Spider.\n"
        "- Export the HTML as CSV Files.\n"
        "- Upload your 'Live' and 'Staging' CSV files using the file uploaders below.\n"
        "- By Default the app looks for columns named 'Address' 'H1-1' and 'Title 1' "
        "but they can be manually mapped if not found.\n"
        "- Select up to 3 columns that you want to match.\n"
        "- Click the 'Process Files' button to start the matching process.\n"
        "- Once processed, a download link for the output file will be provided.\n"
        "- Statistic such as median match score and a total mediam similarity score "
        "will be shown. Run the script with a different combination of columns to "
        "get the best score!"
    )
    with st.expander("How to Use This Tool"):
        st.write(instructions)


def create_page_footer_with_contact_info():
    """
    Adds a footer with contact information to the Streamlit page.
    """
    footer_html = (
        "<hr style='height:2px;border-width:0;color:gray;background-color:gray'>"
        "<p style='font-style: italic;'>Need an app? Need this run as a managed service? "
        "<a href='mailto:hello@leefoot.co.uk'>Hire Me!</a></p>"
    )
    st.markdown(footer_html, unsafe_allow_html=True)


def validate_uploaded_files(file1, file2):
    """
    Validates the uploaded files to ensure they are different.

    Args:
    file1 (UploadedFile): The first uploaded file.
    file2 (UploadedFile): The second uploaded file.

    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not file1 or not file2 or file1.getvalue() == file2.getvalue():
        show_warning_message(
            "Warning: The same file has been uploaded for both live and staging. Please upload different files.")
        return False
    return True


def create_file_uploader_widgets():
    """
    Creates file uploader widgets for live and staging files in the Streamlit interface.

    Returns:
    tuple: A tuple containing the live file uploader widget and the staging file uploader widget.
    """
    col1, col2 = st.columns(2)
    with col1:
        file_live = create_file_uploader_widget("Live", ['csv', 'xlsx', 'xls'])
    with col2:
        file_staging = create_file_uploader_widget("Staging", ['csv', 'xlsx', 'xls'])
    return file_live, file_staging


def handle_data_matching_and_processing(df_live, df_staging, address_column, selected_additional_columns,
                                        selected_model):
    """
    Handles the process of data matching and processing between live and staging dataframes.

    Args:
    df_live (pd.DataFrame): The dataframe for the live data.
    df_staging (pd.DataFrame): The dataframe for the staging data.
    address_column (str): The name of the address column to use for matching.
    selected_additional_columns (list): Additional columns selected for matching.
    selected_model (str): The name of the matching model to use.

    Returns:
    pd.DataFrame: The final processed dataframe after matching.
    """
    message_placeholder = st.empty()
    message_placeholder.info('Matching Columns, Please Wait!')

    rename_dataframe_column(df_live, address_column, 'Address')
    rename_dataframe_column(df_staging, address_column, 'Address')

    all_selected_columns = ['Address'] + selected_additional_columns
    progress_bar = st.progress(0)
    df_final = process_uploaded_files_and_match_data(df_live, df_staging, all_selected_columns, progress_bar,
                                                     message_placeholder,
                                                     selected_additional_columns, selected_model)
    return df_final


# File Reading and Data Preparation ------------------------------------------------------------------------------------

def read_excel_file(file, dtype):
    """
    Reads an Excel file into a Pandas DataFrame.

    Args:
    file (UploadedFile): The Excel file to read.
    dtype (str): Data type to use for the DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing the data from the Excel file.
    """
    return pd.read_excel_file(file, dtype=dtype)


def read_csv_file_with_detected_encoding(file, dtype):
    """
    Reads a CSV file with automatically detected encoding into a Pandas DataFrame.

    Args:
    file (UploadedFile): The CSV file to read.
    dtype (str): Data type to use for the DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    result = chardet.detect(file.getvalue())
    encoding = result['encoding']
    return pd.read_csv(file, dtype=dtype, encoding=encoding, on_bad_lines='skip')


def convert_dataframe_to_lowercase(df):
    """
    Converts all string columns in a DataFrame to lowercase.

    Args:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: DataFrame with all string columns in lowercase.
    """
    return df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)


def rename_dataframe_column(df, old_name, new_name):
    """
    Renames a column in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the column to rename.
    old_name (str): The current name of the column.
    new_name (str): The new name for the column.

    Returns:
    None: The DataFrame is modified in place.
    """
    df.rename(columns={old_name: new_name}, inplace=True)


def process_and_validate_uploaded_files(file_live, file_staging):
    """
    Processes and validates the uploaded live and staging files.

    Args:
    file_live (UploadedFile): The live file uploaded by the user.
    file_staging (UploadedFile): The staging file uploaded by the user.

    Returns:
    tuple: A tuple containing the DataFrame for the live file and the DataFrame for the staging file.
    """
    if validate_uploaded_files(file_live, file_staging):
        # Determine file type and read accordingly
        if file_live.name.endswith('.csv'):
            df_live = read_csv_file_with_detected_encoding(file_live, "str")
        else:  # Excel file
            df_live = read_excel_file(file_live, "str")

        if file_staging.name.endswith('.csv'):
            df_staging = read_csv_file_with_detected_encoding(file_staging, "str")
        else:  # Excel file
            df_staging = read_excel_file(file_staging, "str")

        # Check if dataframes are empty
        if df_live.empty or df_staging.empty:
            show_warning_message("Warning: One or both of the uploaded files are empty.")
            return None, None
        else:
            return df_live, df_staging
    return None, None


# Data Matching and Analysis -------------------------------------------------------------------------------------------

def initialise_matching_model(selected_model="TF-IDF"):
    """
    Initializes the matching model based on the selected option.

    Args:
    selected_model (str, optional): The name of the model to use for matching. Defaults to "TF-IDF".

    Returns:
    PolyFuzz model: An instance of the selected PolyFuzz model.
    """
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


def setup_matching_model(selected_model):
    """
    Sets up the PolyFuzz matching model based on the selected model type.

    Args:
    selected_model (str): The name of the model to use for matching.

    Returns:
    PolyFuzz model: An instance of the selected PolyFuzz model.
    """
    if selected_model == "Edit Distance":
        model = PolyFuzz(EditDistance())
    elif selected_model == "RapidFuzz":
        model = PolyFuzz(RapidFuzz())
    else:  # Default to TF-IDF
        model = PolyFuzz(TFIDF())
    return model


def match_columns_and_compute_scores(model, df_live, df_staging, matching_columns):
    """
    Matches columns between two DataFrames (df_live and df_staging) and computes similarity scores.

    Args:
        model: The matching model to use for matching (e.g., PolyFuzz).
        df_live (pd.DataFrame): The DataFrame containing live data.
        df_staging (pd.DataFrame): The DataFrame containing staging data.
        matching_columns (list): List of column names to match between DataFrames.

    Returns:
        dict: A dictionary containing match scores for each column.
    """
    matches_scores = {}
    for col in matching_columns:
        # Check if the column exists in both dataframes
        if col in df_live.columns and col in df_staging.columns:
            # Ensure the data type is appropriate (i.e., Pandas Series)
            if isinstance(df_live[col], pd.Series) and isinstance(df_staging[col], pd.Series):
                live_list = df_live[col].fillna('').tolist()
                staging_list = df_staging[col].fillna('').tolist()

                # Here's the matching logic:
                model.match(live_list, staging_list)
                matches = model.get_matches()
                matches_scores[col] = matches

            else:
                st.warning(f"The column '{col}' in either the live or staging data is not a valid series.")
        else:
            st.warning(f"The column '{col}' does not exist in both the live and staging data.")

    return matches_scores


def identify_best_matching_url(row, matches_scores, matching_columns, df_staging):
    """
    Identifies the best matching URL for a given row in the DataFrame.

    Args:
    row (pd.Series): A row from the DataFrame.
    matches_scores (dict): Dictionary containing match scores for columns.
    matching_columns (list): List of column names used for matching.
    df_staging (pd.DataFrame): The DataFrame containing staging data.

    Returns:
    tuple: A tuple containing best match information and similarity scores.
    """
    best_match_info = {'Best Match on': None, 'Highest Matching URL': None,
                       'Highest Similarity Score': 0, 'Best Match Content': None}
    similarities = []

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
                        'Highest Matching URL':
                            df_staging.loc[df_staging[col] == match_row.iloc[0]['To'], 'Address'].values[0],
                        'Highest Similarity Score': similarity_score,
                        'Best Match Content': match_row.iloc[0]['To']
                    })

    best_match_info['Median Match Score'] = np.median(similarities) if similarities else None
    return best_match_info, similarities


def add_additional_info_to_match_results(best_match_info, df_staging, selected_additional_columns):
    """
    Adds additional information to the best match results.

    Args:
    best_match_info (dict): Dictionary containing information about the best match.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    selected_additional_columns (list): List of additional columns selected for matching.

    Returns:
    dict: Updated best match information with additional details.
    """
    for additional_col in selected_additional_columns:
        if additional_col in df_staging.columns:
            staging_value = df_staging.loc[
                df_staging['Address'] == best_match_info['Highest Matching URL'], additional_col].values
            best_match_info[f'Staging {additional_col}'] = staging_value[0] if staging_value.size > 0 else None
    return best_match_info


def identify_best_matching_url_and_median(df_live, df_staging, matches_scores, matching_columns,
                                          selected_additional_columns):
    """
    Identifies the best matching URLs and computes median match scores for the entire DataFrame.

    Args:
    df_live (pd.DataFrame): The DataFrame containing live data.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    matches_scores (dict): Dictionary containing match scores for columns.
    matching_columns (list): List of column names used for matching.
    selected_additional_columns (list): List of additional columns selected for matching.

    Returns:
    pd.DataFrame: DataFrame with best match URLs and median scores.
    """

    def process_row(row):
        best_match_info, similarities = identify_best_matching_url(row, matches_scores, matching_columns, df_staging)
        best_match_info = add_additional_info_to_match_results(best_match_info, df_staging, selected_additional_columns)
        # Convert scores to percentage format with '%' sign
        best_match_info['All Column Match Scores'] = [
            (col, f"{round(score * 100)}%" if not pd.isna(score) else "NaN%")
            for col, score in zip(matching_columns, similarities)
        ]
        return pd.Series(best_match_info)

    return df_live.apply(process_row, axis=1)


def finalise_match_results_processing(df_live, df_staging, matches_scores, matching_columns,
                                      selected_additional_columns):
    """
    Finalizes the match result processing by combining live and matched data.

    Args:
    df_live (pd.DataFrame): The DataFrame containing live data.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    matches_scores (dict): Dictionary containing match scores for columns.
    matching_columns (list): List of column names used for matching.
    selected_additional_columns (list): List of additional columns selected for matching.

    Returns:
    pd.DataFrame: The final DataFrame after processing match results.
    """
    match_results = identify_best_matching_url_and_median(df_live, df_staging, matches_scores, matching_columns,
                                                          selected_additional_columns)
    df_final = prepare_concatenated_dataframe_for_display(df_live, match_results, matching_columns)
    return df_final


def process_uploaded_files_and_match_data(df_live, df_staging, matching_columns, progress_bar, message_placeholder,
                                          selected_additional_columns,
                                          selected_model):
    """
    Processes the uploaded files and performs data matching using the specified model.

    Args:
    df_live (pd.DataFrame): The DataFrame containing live data.
    df_staging (pd.DataFrame): The DataFrame containing staging data.
    matching_columns (list): List of column names to match between DataFrames.
    progress_bar (streamlit.progress_bar): Streamlit progress bar object.
    message_placeholder (streamlit.empty): Streamlit placeholder for messages.
    selected_additional_columns (list): Additional columns selected for matching.
    selected_model (str): The name of the matching model to use.

    Returns:
    pd.DataFrame: The final DataFrame after processing and matching data.
    """
    df_live = convert_dataframe_to_lowercase(df_live)
    df_staging = convert_dataframe_to_lowercase(df_staging)

    model = setup_matching_model(selected_model)
    matches_scores = process_column_matches_and_scores(model, df_live, df_staging, matching_columns)

    for index, _ in enumerate(matching_columns):
        progress = (index + 1) / len(matching_columns)
        progress_bar.progress(progress)

    message_placeholder.info('Finalising the processing. Please Wait!')
    df_final = finalise_match_results_processing(df_live, df_staging, matches_scores, matching_columns,
                                                 selected_additional_columns)

    display_final_results_and_download_link(df_final, 'migration_mapping_data.xlsx')
    message_placeholder.success('Complete!')

    return df_final


def scale_median_match_scores_to_percentage(df_final):
    """
    Scales the median match scores in the DataFrame to percentage values.

    Args:
    df_final (pd.DataFrame): The final DataFrame containing match results.

    Returns:
    pd.DataFrame: The DataFrame with scaled median match scores.
    """
    df_final['Median Match Score Scaled'] = df_final['Median Match Score'] * 100
    return df_final


def group_median_scores_into_brackets(df_final):
    """
    Groups the median match scores in the DataFrame into predefined score brackets.

    Args:
    df_final (pd.DataFrame): The final DataFrame containing match results.

    Returns:
    pd.DataFrame: The DataFrame with median scores grouped into brackets.
    """
    bins = range(0, 110, 10)
    labels = [f'{i}-{i + 10}' for i in range(0, 100, 10)]
    df_final['Score Bracket'] = pd.cut(df_final['Median Match Score Scaled'], bins=bins, labels=labels,
                                       include_lowest=True)
    return df_final


def generate_score_distribution_dataframe(df_final):
    """
    Generates a DataFrame representing the distribution of median match scores.

    Args:
    df_final (pd.DataFrame): The final DataFrame containing match results.

    Returns:
    pd.DataFrame: A DataFrame with score distribution data.
    """
    df_final['Median Match Score Scaled'] = df_final['Median Match Score'] * 100
    bins = range(0, 110, 10)
    labels = [f'{i}-{i + 10}' for i in range(0, 100, 10)]
    df_final['Score Bracket'] = pd.cut(df_final['Median Match Score Scaled'], bins=bins, labels=labels,
                                       include_lowest=True)
    score_brackets = df_final['Score Bracket'].value_counts().sort_index().reindex(labels, fill_value=0)

    score_data = pd.DataFrame({
        'Score Bracket': score_brackets.index,
        'URL Count': score_brackets.values
    })
    return score_data


def select_columns_for_matching(df_live, df_staging):
    """
    Selects columns for data matching from live and staging DataFrames.

    Args:
    df_live (pd.DataFrame): The DataFrame containing live data.
    df_staging (pd.DataFrame): The DataFrame containing staging data.

    Returns:
    tuple: A tuple containing the selected address column and additional columns for matching.
    """
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


def process_column_matches_and_scores(model, df_live, df_staging, matching_columns):
    """
    Processes and computes the scores for column matches between live and staging dataframes.

    Args:
    model (PolyFuzz model): The matching model to use.
    df_live (pd.DataFrame): The live dataframe.
    df_staging (pd.DataFrame): The staging dataframe.
    matching_columns (list): A list of columns to match between the dataframes.

    Returns:
    dict: A dictionary containing the match scores for each column.
    """
    return match_columns_and_compute_scores(model, df_live, df_staging, matching_columns)


# Data Visualization and Reporting -------------------------------------------------------------------------------------

def plot_median_score_histogram(df_final, col):
    """
    Plots a histogram of median match scores.

    Args:
    df_final (pd.DataFrame): The final dataframe containing the match scores.
    col (streamlit column): The Streamlit column where the plot will be displayed.
    """
    bracket_counts = df_final['Score Bracket'].value_counts().sort_index()

    with col:
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


def display_final_results_and_download_link(df_final, filename):
    """
    Displays the final results and a download link for the output Excel file.

    Args:
    df_final (pd.DataFrame): The final processed dataframe.
    filename (str): The name of the file to be downloaded.
    """
    show_download_link_for_final_excel(df_final, filename)
    display_median_score_brackets_chart(df_final)
    st.balloons()


def display_median_similarity_indicator_chart(df_final, col):
    """
    Displays a chart showing the median similarity indicator.

    Args:
    df_final (pd.DataFrame): The final dataframe containing similarity scores.
    col (streamlit column): The Streamlit column where the chart will be displayed.
    """
    median_similarity_score = df_final['Highest Similarity Score'].median()

    if 'previous_score' in st.session_state:
        reference_value = st.session_state['previous_score']
    else:
        reference_value = median_similarity_score

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=median_similarity_score,
        delta={'reference': reference_value, 'relative': False, 'valueformat': '.2%'},
        number={'valueformat': '.2%', 'font': {'color': 'black'}},
        title={'text': "Highest Matching Column Median Similarity Score", 'font': {'color': 'black'}},
        domain={'row': 0, 'column': 0}))

    fig.update_layout(
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"}
    )

    with col:
        st.plotly_chart(fig)

    st.session_state['previous_score'] = median_similarity_score


def display_median_score_brackets_chart(df_final):
    """
    Displays a chart of the median score brackets.

    Args:
    df_final (pd.DataFrame): The final dataframe containing the score brackets.
    """
    df_scaled = scale_median_match_scores_to_percentage(df_final)
    df_bracketed = group_median_scores_into_brackets(df_scaled)

    col1, col2 = st.columns(2)
    plot_median_score_histogram(df_bracketed, col1)
    display_median_similarity_indicator_chart(df_final, col2)


# Excel File Operations ------------------------------------------------------------------------------------------------

def create_excel_with_dataframes(df, score_data, filename):
    """
    Creates an Excel file with the provided dataframes.

    Args:
    df (pd.DataFrame): The main dataframe to be included in the Excel file.
    score_data (pd.DataFrame): The dataframe containing score distribution data.
    filename (str): The name of the output Excel file.

    Returns:
    pd.ExcelWriter: Excel writer object used to write the dataframes to an Excel file.
    """
    excel_writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(excel_writer, sheet_name='Mapped URLs', index=False)
    score_data.to_excel(excel_writer, sheet_name='Median Score Distribution', index=False)
    return excel_writer


def apply_formatting_to_excel_sheets(excel_writer, df):
    """
    Applies formatting to the Excel sheets created by the Excel writer.

    Args:
    excel_writer (pd.ExcelWriter): The Excel writer object.
    df (pd.DataFrame): The dataframe used for setting column widths.
    """
    workbook = excel_writer.book
    worksheet1 = excel_writer.sheets['Mapped URLs']

    # Formats
    left_align_format = workbook.add_format({'align': 'left'})
    percentage_format = workbook.add_format({'num_format': '0.00%', 'align': 'center'})

    num_rows = len(df)
    num_cols = len(df.columns)
    worksheet1.add_table(0, 0, num_rows, num_cols - 1, {'columns': [{'header': col} for col in df.columns]})
    worksheet1.freeze_panes(1, 0)

    max_col_width = 80
    for i, col in enumerate(df.columns):
        col_width = max(len(col), max(df[col].astype(str).apply(len).max(), 10)) + 2
        col_width = min(col_width, max_col_width)

        # Apply specific formatting for columns 'E', 'F', and 'H' (indices 4, 5, and 7)
        if i in [3, 5, 7]:  # Adjusting the indices for columns E, F, and H
            worksheet1.set_column(i, i, col_width, percentage_format)
            # Apply 3-color scale formatting with specified colors
            worksheet1.conditional_format(1, i, num_rows, i, {
                'type': '3_color_scale',
                'min_color': "#f8696b",  # Custom red for lowest values
                'mid_color': "#ffeb84",  # Custom yellow for middle values
                'max_color': "#63be7b"  # Custom green for highest values
            })
        else:
            worksheet1.set_column(i, i, col_width, left_align_format)

    return workbook


def add_chart_to_excel_sheet(excel_writer, score_data):
    """
    Adds a chart to the Excel sheet.

    Args:
    excel_writer (pd.ExcelWriter): The Excel writer object.
    score_data (pd.DataFrame): The dataframe containing score distribution data to be plotted.
    """
    workbook = excel_writer.book
    worksheet2 = excel_writer.sheets['Median Score Distribution']
    chart = workbook.add_chart({'type': 'column'})
    max_row = len(score_data) + 1

    chart.add_series({
        'name': "='Median Score Distribution'!$B$1",
        'categories': "='Median Score Distribution'!$A$2:$A$" + str(max_row),
        'values': "='Median Score Distribution'!$B$2:$B$" + str(max_row),
    })

    chart.set_title({'name': 'Distribution of Median Match Scores'})
    chart.set_x_axis({'name': 'Median Match Score Brackets'})
    chart.set_y_axis({'name': 'URL Count'})
    worksheet2.insert_chart('D2', chart)


def create_excel_download_link(filename):
    """
    Creates a download link for an Excel file.

    Args:
    filename (str): The name of the file for which the download link is created.

    Returns:
    str: A HTML hyperlink for downloading the Excel file.
    """
    with open(filename, 'rb') as file:
        b64 = base64.b64encode(file.read()).decode()
    download_link = (
        f'<a href="data:application/vnd.openxmlformats-officedocument.'
        f'spreadsheetml.sheet;base64,{b64}" download="{filename}">'
        f'Click here to download {filename}</a>'
    )
    return download_link


def generate_excel_download_and_display_link(df, filename, score_data):
    """
    Generates an Excel file and creates a download link for it.

    Args:
    df (pd.DataFrame): The dataframe to be included in the Excel file.
    filename (str): The name of the output Excel file.
    score_data (pd.DataFrame): Additional score data to be included in the Excel file.
    """
    excel_writer = create_excel_with_dataframes(df, score_data, filename)
    apply_formatting_to_excel_sheets(excel_writer, df)
    add_chart_to_excel_sheet(excel_writer, score_data)
    excel_writer.close()

    download_link = create_excel_download_link(filename)
    st.markdown(download_link, unsafe_allow_html=True)


def show_download_link_for_final_excel(df_final, filename):
    """
    Displays the download link for the final Excel file.

    Args:
    df_final (pd.DataFrame): The final dataframe to be included in the Excel file.
    filename (str): The name of the output Excel file.
    """
    df_for_score_data = df_final.drop(['Median Match Score Scaled', 'Score Bracket'], axis=1, inplace=False,
                                      errors='ignore')
    score_data = generate_score_distribution_dataframe(df_for_score_data)
    generate_excel_download_and_display_link(df_final, 'migration_mapping_data.xlsx', score_data)


# Main Function and Additional Utilities -------------------------------------------------------------------------------

def format_match_scores_as_strings(df):
    """
    Formats the match scores in the dataframe as strings.

    Args:
    df (pd.DataFrame): The dataframe containing match scores.

    Returns:
    pd.DataFrame: The updated dataframe with formatted match scores.
    """
    df['All Column Match Scores'] = df['All Column Match Scores'].apply(lambda x: str(x) if x is not None else None)
    return df


def merge_live_and_matched_dataframes(df_live, match_results, matching_columns):
    """
    Merges the live dataframe with the matched results dataframe.

    Args:
    df_live (pd.DataFrame): The live dataframe.
    match_results (pd.DataFrame): The dataframe containing matched results.
    matching_columns (list): List of columns used for matching.

    Returns:
    pd.DataFrame: The merged dataframe.
    """
    final_columns = ['Address'] + [col for col in matching_columns if col != 'Address']
    return pd.concat([df_live[final_columns], match_results], axis=1)


def prepare_concatenated_dataframe_for_display(df_live, match_results, matching_columns):
    """
    Prepares the concatenated dataframe for display by merging and formatting.

    Args:
    df_live (pd.DataFrame): The live dataframe.
    match_results (pd.DataFrame): The matched results dataframe.
    matching_columns (list): The columns used for matching.

    Returns:
    pd.DataFrame: The final concatenated and formatted dataframe.
    """
    final_df = merge_live_and_matched_dataframes(df_live, match_results, matching_columns)
    final_df = format_match_scores_as_strings(final_df)
    return final_df


def main():
    """
    The main function to run the Streamlit application. Sets up the interface, handles file uploads,
    processes data matching, and displays results.
    """
    setup_streamlit_interface()

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

    file_live, file_staging = create_file_uploader_widgets()
    if file_live and file_staging:
        df_live, df_staging = process_and_validate_uploaded_files(file_live, file_staging)
        if df_live is not None and df_staging is not None:
            address_column, selected_additional_columns = select_columns_for_matching(df_live,
                                                                                      df_staging)
            if st.button("Process Files"):
                df_final = handle_data_matching_and_processing(df_live, df_staging, address_column,
                                                               selected_additional_columns,
                                                               selected_model)

    create_page_footer_with_contact_info()


if __name__ == "__main__":
    main()
