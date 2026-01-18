import pandas as pd
import streamlit as st

st.set_page_config(page_title="Search Term Report MPN & Key Phrase Extractor", page_icon="✂️",
                   layout="wide")
import chardet
from io import BytesIO
from rakun2 import RakunKeyphraseDetector

submitted = False

# set the variable slides
min_len = st.sidebar.slider("Set Minimum MPN Length", value=4)
detect_sizes = st.sidebar.selectbox('Remove Sizes, mm, cm etc?', (True, False))
num_keywords = st.sidebar.slider("num_keywords", value=10)
merge_threshold = st.sidebar.slider("merge_threshold", value=1.1)
alpha = st.sidebar.slider("alpha", value=0.3)
token_prune_len = st.sidebar.slider("token_prune_len", value=3)

# initialise rakun2 settings
hyperparameters = {"num_keywords": num_keywords,
                   "merge_threshold": merge_threshold,
                   "alpha": alpha,
                   "token_prune_len": token_prune_len}

keyword_detector = RakunKeyphraseDetector(hyperparameters)

st.title("Search Term Report MPN & Key Phrase Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")


with st.expander("How to use this tool"):
    st.markdown("""
    1. Export your Search Terms Report from Google Ads
    2. Upload the CSV file below
    3. Select the column containing search terms
    4. Download the extracted MPNs, words and keyphrases
    """)

uploaded_file = st.file_uploader(
    "Upload your Search Terms Report",
    help="""Upload a Search Terms Report Exported from Google Ads""")

if uploaded_file is not None:

    result = chardet.detect(uploaded_file.getvalue())
    encoding_value = result["encoding"]

    if encoding_value == "UTF-16":
        white_space = True
    else:
        white_space = False

    df = pd.read_csv(
        uploaded_file,
        encoding=encoding_value,
        delim_whitespace=white_space,
        encoding_errors='ignore',
        on_bad_lines='skip',
        skiprows=2,
        dtype='str',
    )

    if len(df) == 0:
        st.caption("Your sheet seems empty!")

    with st.expander("↕️ View raw data", expanded=False):
        st.write(df)
else:
    st.stop()


with st.form(key='columns_in_form_2'):
    st.subheader("Please Select the Search Term Column")
    kw_col = st.selectbox('Select the keyword column:', df.columns)
    submitted = st.form_submit_button('Submit')

if submitted:
    st.info("Hold Tight Processing Data!")
    df[kw_col] = df[kw_col].str.encode('ascii', 'ignore').str.decode('ascii')
    df.drop_duplicates(subset=kw_col, inplace=True)

    # keep only rows which are not NaN
    df = df[df[kw_col].notna()]
    df.drop_duplicates(subset=kw_col, inplace=True)  # Drop Duplicate Rows

    # use rakun 2 to extract keyphrases
    txt = ' '.join(df[kw_col])
    keyphrases = keyword_detector.find_keywords(txt, input_type="string")
    df_phrases = pd.DataFrame(keyphrases)
    df_phrases.rename(columns={df_phrases.columns[0]: "Key Phrase", df_phrases.columns[1]: "Score"}, inplace=True)
    df_phrases['Score'] = df_phrases['Score'].round(2)

    # extract words and models from the dataset (used to extract brands and MPNs)
    df[['words', 'model_number']] = (df[kw_col]
     .str.extractall(r'(\S*\d\S*)|([^\s\d]+)') # order is important
     .groupby(level=0).agg(lambda s: ' '.join(s.dropna()))
     .loc[:, ::-1] # invert 2 columns
    )

    if detect_sizes:
        size_patterns = ['0mm', '1mm', '2mm', '3mm', '4mm', '5mm', '6mm', '7mm', '8mm', '9mm',
                        '0cm', '1cm', '2cm', '3cm', '4cm', '5cm', '6cm', '7cm', '8cm', '9cm',
                        '0ft', '1ft', '2ft', '3ft', '4ft', '5ft', '6ft', '7ft', '8ft', '9ft']
        for pattern in size_patterns:
            df = df[~df["model_number"].str.contains(pattern, na=False)]

    # make mpn dataframe
    df_model = df.copy()
    df_model = df_model[['model_number']]

    # make word dataframe
    df_words = df.copy()
    df_words['words'] = df_words['words'].astype(str)
    df_words = df_words[['words']]

    # find all unique mpns in a dataframe column
    df_model['model_number'] = df_model['model_number'].astype(str)
    my_mpns = df_model['model_number'].str.lower().str.findall("\w+")
    unique_mpns = set()

    for x in my_mpns:
        unique_mpns.update(x)

    mpns = (list(unique_mpns))
    mpns = sorted(mpns)

    # find all unique words in a dataframe column
    words = df_words['words'].str.lower().str.findall("\w+")
    unique_words = set()

    for x in words:
        unique_words.update(x)

    words = (list(unique_words))
    words = sorted(words)

    words = [x for x in words if len(x) > min_len]
    mpns = [x for x in mpns if len(x) > min_len]

    # make dataframes from words and mpn lists

    df_words_export = pd.DataFrame(words, columns=["Unique Words"])
    df_mpns_export = pd.DataFrame(mpns, columns=["MPNs"])

    # save to Excel sheet
    dfs = [df_mpns_export, df_words_export, df_phrases]  # make a list of the dataframe to use as a sheet

    def dfs_tabs(df_list, sheet_list, file_name):  # function to save all dataframes to one single excel doc

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        count = 0
        for dataframe, sheet in zip(df_list, sheet_list):
            dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0, index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheet]

            # Get the dimensions of the dataframe.
            (max_row, max_col) = dfs[count].shape

            # Create a list of column headers, to use in add_table().
            column_settings = [{'header': column} for column in dfs[count].columns]

            # Add the Excel table structure. Pandas will add the data.
            worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

            # Make the columns wider for clarity.
            worksheet.set_column(0, max_col - 1, 25)

            count = count + 1  # increase the count

        workbook.close()
        processed_data = output.getvalue()
        return processed_data

    sheets = ['MPNS', 'Unique Words', 'Key Phrases']  # set list of worksheet names

    df_xlsx = dfs_tabs(dfs, sheets, 'mpn-extraction.xlsx')
    st.download_button(label='📥 Download Extracted Data', data=df_xlsx, file_name='extracted_mpns.xlsx')

    with st.expander("View MPNs"):
        st.write(mpns)

    with st.expander("View Key Phrases"):
        st.write(list(df_phrases['Key Phrase']))

    with st.expander("View Unique Words"):
        st.write(words)
