####################################################################################
#                                                                                  #
#  GSC Folder Analyzer                                                             #
#                                                                                  #
#  Groups Search Console data into a folder/path view for quick analysis.          #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""
GSC Folder Analyzer - Streamlit App

Aggregates Google Search Console data by URL folder/path to analyze
site section performance. Upload a GSC export and see clicks, impressions,
and top keywords grouped by folder depth.

Requirements:
    pip install streamlit pandas
"""

import streamlit as st
import pandas as pd
from io import BytesIO

# App Configuration
st.set_page_config(
    page_title="GSC Folder Analyzer",
    page_icon="📁",
    layout="wide"
)

st.title("📁 GSC Folder Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Analyzes GSC performance by URL folder
    - Groups metrics by site section
    - Identifies top/bottom performing areas

    **How to use:**
    1. Upload GSC data with URLs
    2. Configure folder depth
    3. Analyze folder performance
    4. Export section analysis

    **Best for:**
    - Site section analysis
    - Content area prioritization
    - Resource allocation decisions
    """)
st.markdown("""
Analyze your Google Search Console data by URL folder structure.
Upload a GSC export to see performance metrics aggregated by path/folder.
""")

# Sidebar configuration
st.sidebar.header("Settings")
domain = st.sidebar.text_input(
    "Domain (with trailing slash)",
    value="https://example.com/",
    help="Enter your domain with protocol and trailing slash, e.g., https://example.com/"
)

max_folder_depth = st.sidebar.slider(
    "Maximum Folder Depth",
    min_value=1,
    max_value=10,
    value=5,
    help="How deep to analyze the folder structure"
)

# File uploader
st.header("Upload GSC Data")
uploaded_file = st.file_uploader(
    "Upload your Search Console export (CSV)",
    type=["csv"],
    help="Export from GSC should contain: query, page, clicks, impressions, position columns"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, dtype=str)

        # Standardize column names (handle various GSC export formats)
        column_mapping = {
            'Top queries': 'query',
            'Query': 'query',
            'Queries': 'query',
            'Top pages': 'page',
            'Page': 'page',
            'Pages': 'page',
            'URL': 'page',
            'Clicks': 'clicks',
            'Impressions': 'impressions',
            'CTR': 'ctr',
            'Position': 'position',
            'Average position': 'position'
        }

        df.rename(columns=column_mapping, inplace=True)
        df.columns = df.columns.str.lower()

        # Check required columns
        required_cols = ['query', 'page', 'clicks', 'impressions', 'position']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.write("Available columns:", list(df.columns))
            st.stop()

        # Convert numeric columns
        df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
        df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
        df['position'] = pd.to_numeric(df['position'], errors='coerce').fillna(0)

        st.success(f"Loaded {len(df):,} rows of GSC data")

        # Show raw data preview
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head(20))

        # Process button
        if st.button("Analyze Folder Structure", type="primary"):
            with st.spinner("Processing folder analysis..."):
                # Make a copy for processing
                df_work = df.copy()

                # Find top keyword by clicks for each page
                df_work['clicks_max'] = df_work.groupby('page')['clicks'].transform('max')
                df_work.sort_values(['page', 'clicks_max'], ascending=[True, False], inplace=True)
                df_work['exact_clicks_match'] = df_work['clicks_max'] == df_work['clicks']

                df_work.loc[df_work['exact_clicks_match'] == True, 'Top Keyword'] = df_work['query']
                df_work.loc[df_work['exact_clicks_match'] == True, 'Volume'] = df_work['impressions']
                df_work.loc[df_work['exact_clicks_match'] == True, 'Top Traffic'] = df_work['clicks']
                df_work.loc[df_work['exact_clicks_match'] == True, 'Top Position'] = df_work['position']

                # Forward fill the top keyword data
                df_work = df_work.sort_values('page')
                df_work['Top Keyword'] = df_work.groupby('page')['Top Keyword'].ffill()
                df_work['Volume'] = df_work.groupby('page')['Volume'].ffill()
                df_work['Top Traffic'] = df_work.groupby('page')['Top Traffic'].ffill()
                df_work['Top Position'] = df_work.groupby('page')['Top Position'].ffill()

                # Clean page URLs - remove domain
                if domain:
                    df_work['page'] = df_work['page'].str.replace(domain, "", regex=False)

                # Remove parameters and anchors
                df_work['page'] = df_work['page'].str.split("?").str[0]
                df_work['page'] = df_work['page'].str.split("#").str[0]

                # Handle homepage
                df_work.loc[df_work['page'] == "/", "page"] = domain
                df_work.loc[df_work['page'] == "", "page"] = domain
                df_work['page'] = df_work['page'].str.rstrip("/")

                # Calculate folder depth
                df_work["folder_depth"] = df_work["page"].str.count("/")

                # Limit to max folder depth
                actual_max_depth = min(df_work["folder_depth"].max() + 1, max_folder_depth)
                cols = list(range(0, actual_max_depth))

                # Split path into columns
                df_work[cols] = df_work['page'].str.split('/', expand=True).iloc[:, :actual_max_depth]

                # Build cumulative paths
                for column in cols:
                    n1 = column + 1
                    if n1 in cols:
                        try:
                            df_work[n1] = df_work[column].astype(str) + "/" + df_work[n1].astype(str)
                        except (ValueError, KeyError):
                            pass

                # Make a copy for page counting
                df_raw_data = df_work.drop_duplicates(subset=["page"]).copy()

                # Aggregate by folder
                df_list = []
                df_work.sort_values(["clicks", "Volume"], ascending=[True, False], inplace=True)

                for i in cols:
                    if i in df_work.columns:
                        df_loop = df_work.groupby(i).agg({
                            "clicks": "sum",
                            "query": "count",
                            "impressions": "sum",
                            "Top Keyword": "first",
                            "Volume": "first",
                            "Top Position": "first"
                        })
                        df_list.append(df_loop)

                if not df_list:
                    st.error("No folder data could be extracted. Check your domain setting.")
                    st.stop()

                df_final = pd.concat(df_list).reset_index()
                df_final.rename(columns={
                    "index": "Path",
                    "clicks": "Traffic",
                    "query": "Keywords"
                }, inplace=True)

                # Add domain prefix back to paths
                df_final['Path'] = domain.rstrip('/') + "/" + df_final['Path'].astype(str)

                # Count pages in each path
                count_list = []
                for path in df_final['Path']:
                    try:
                        search_path = path.replace(domain, "")
                        temp = df_raw_data[df_raw_data["page"].str.contains(search_path, na=False, regex=False)]
                        count_list.append(len(temp))
                    except Exception:
                        count_list.append(0)

                df_final['Pages'] = count_list

                # Format output
                output_cols = ["Traffic", "Keywords", "Pages", "Path", "Top Keyword", "Volume", "Top Position"]
                df_final = df_final.reindex(columns=[c for c in output_cols if c in df_final.columns])

                if 'Top Position' in df_final.columns:
                    df_final['Top Position'] = df_final['Top Position'].round(2)

                # Sort by traffic
                df_final = df_final.sort_values("Traffic", ascending=False)

                # Display results
                st.header("Folder Analysis Results")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Traffic", f"{df_final['Traffic'].sum():,}")
                with col2:
                    st.metric("Total Keywords", f"{df_final['Keywords'].sum():,}")
                with col3:
                    st.metric("Unique Paths", f"{len(df_final):,}")
                with col4:
                    st.metric("Total Pages", f"{df_final['Pages'].sum():,}")

                # Results table
                st.dataframe(
                    df_final,
                    use_container_width=True,
                    hide_index=True
                )

                # Download button
                output = BytesIO()
                df_final.to_csv(output, index=False)
                output.seek(0)

                st.download_button(
                    label="📥 Download CSV",
                    data=output,
                    file_name="gsc_folder_analysis.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload a CSV file from Google Search Console to get started.")

    st.markdown("""
    ### How to export data from Google Search Console:

    1. Go to [Google Search Console](https://search.google.com/search-console)
    2. Select your property
    3. Go to **Performance** report
    4. Click **Export** > **Download CSV**
    5. Upload the file here

    ### What this tool does:

    - Groups your GSC data by URL folder/path structure
    - Shows traffic, keywords, and pages per folder
    - Identifies the top performing keyword per folder
    - Helps identify which site sections perform best
    """)
