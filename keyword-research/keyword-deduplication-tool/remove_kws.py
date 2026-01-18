import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
import chardet
from stqdm import stqdm  # Import stqdm
import io


def read_file(file_buffer, file_type, encoding):
    if file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file_buffer, engine='openpyxl')
    elif file_type == "text/csv":
        df = pd.read_csv(file_buffer, encoding=encoding)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df


# Function to detect encoding
def detect_encoding(file_buffer):
    file_buffer.seek(0)  # Ensure you're at the start of the file
    result = chardet.detect(file_buffer.read(100000))  # Read first 100000 bytes to determine encoding
    file_buffer.seek(0)  # Reset file pointer after reading
    return result['encoding']



# Function to find close variations of a keyword
def find_close_variations(keyword, keywords_list, threshold=99):
    # Get the best match with similarity score >= threshold
    match = process.extractOne(keyword, keywords_list, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    # Return the match if found
    if match:
        return match[0]
    else:
        return None


def main():
    st.set_page_config(page_title="Keyword Deduplication Tool", page_icon=":mag:")
    st.title("Keyword Deduplication Tool")
    st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Removes duplicate keywords from lists
    - Handles variations and near-duplicates
    - Cleans keyword data for analysis

    **How to use:**
    1. Upload keyword CSV
    2. Select keyword column
    3. Configure deduplication settings
    4. Download cleaned list

    **Best for:**
    - Keyword list cleaning
    - Data preparation for tools
    - Removing research redundancy
    """)
    st.markdown(
        '<p class="small-font">Created by <a href="https://LeeFoot.com">LeeFoot</a> on 08th March 2024</p>',
        unsafe_allow_html=True)

    with st.expander("How to Use This App"):
        st.markdown("""
        **Keyword Deduplication Tool**

        This application deduplicates similar keywords, keeping the keyword with the highest search volume.

        **Functionality:**

        - Upload a file in either Excel or CSV format containing a list of keywords along with their corresponding search volumes.
        - Select the relevant columns for keywords and search volumes.
        - Start the deduplication process by clicking the 'Dedupe' button.
        - Similar keywords are automatically removed. They are also saved in a separate worksheet to be reviewed.

        """)

    message_placeholder = st.empty()  # Placeholder for messages

    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if file is not None:
        encoding = detect_encoding(file)
        st.write("Detected Encoding:", encoding)

        df = read_file(file, file.type, encoding)

        if df is not None:
            message_placeholder.success("File successfully loaded!")
            st.subheader("Column Mapping")

            default_keyword_cols = ["KW", "kw", "Keyword", "keyword", "keywords", "Keywords"]
            keyword_col = st.selectbox("Select Keyword Column",
                                       [col for col in df.columns if col in default_keyword_cols])

            default_volume_cols = ["vol", "Vol", "Volume", "volume", "search volume", "Search Volume"]
            volume_col = st.selectbox("Select Volume Column", [col for col in df.columns if col in default_volume_cols])

            if keyword_col == volume_col:
                st.error("Keyword column and Volume column must be different.")
                return

            if st.button("Dedupe"):
                df_copy = df.copy()

                # Process deduplication
                with stqdm(df_copy[keyword_col], desc="Progress") as progress_bar:
                    matches = []
                    for keyword in progress_bar:
                        match = find_close_variations(keyword, df_copy[keyword_col].tolist())
                        matches.append(match)
                df_copy['Parent_Keyword'] = matches

                df_copy['Equal_Keyword'] = df_copy[keyword_col] == df_copy['Parent_Keyword']

                # Create a DataFrame for dropped rows
                dropped_df = df_copy[~df_copy['Equal_Keyword']].copy()

                # Keep rows where Equal_Keyword is True
                df_copy = df_copy[df_copy['Equal_Keyword']].copy()

                # Remove 'Equal_Keyword' column since it's no longer needed
                df_copy.drop(columns=['Equal_Keyword', 'Parent_Keyword'], inplace=True)
                dropped_df.drop(columns=['Equal_Keyword', 'Parent_Keyword'], inplace=True)

                # Use Pandas ExcelWriter to write to multiple sheets
                excel_bytes_io = io.BytesIO()
                with pd.ExcelWriter(excel_bytes_io, engine='xlsxwriter') as writer:
                    df_copy.to_excel(writer, sheet_name='Processed Keywords', index=False)
                    dropped_df.to_excel(writer, sheet_name='Dropped Keywords', index=False)

                # Streamlit download button for Excel file
                st.download_button(
                    label="Download Excel file with processed and dropped data",
                    data=excel_bytes_io.getvalue(),
                    file_name="keywords_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Update the message placeholder after processing is complete
                message_placeholder.success("Processing complete!")


# The __name__ == "__main__" condition should be outside and after the main function
if __name__ == "__main__":
    main()
