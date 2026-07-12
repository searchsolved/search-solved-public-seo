# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  Navigation Label Mismatch Detector                                              #
#                                                                                  #
#  Find navigation links whose anchor text does not match the destination         #
#  page's H1 or primary title keyword, using Screaming Frog exports.               #
#                                                                                  #
####################################################################################
# Author: Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Navigation Label Mismatch Detector - Streamlit App

Takes a Screaming Frog 'All Inlinks' export and an 'Internal HTML' export,
identifies navigation links (links where the alt text matches the anchor text),
then compares each navigation label against the destination page's H1 and the
primary keyword from its page title. Mismatches are flagged for review.

Requirements:
    pip install streamlit pandas
"""

from io import BytesIO

import pandas as pd
import streamlit as st

# App Configuration
st.set_page_config(
    page_title="Navigation Label Mismatch Detector",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Navigation Label Mismatch Detector")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Identifies navigation links from a Screaming Frog inlinks export
    - Compares each navigation label to the destination page's H1
    - Compares each navigation label to the page title's primary keyword
    - Flags labels that do not match, so you can align them

    **How to use:**
    1. Upload your Screaming Frog **All Inlinks** export (Bulk Export > Links > All Inlinks)
    2. Upload your Screaming Frog **Internal HTML** export (Internal tab > filter HTML > Export)
    3. Check the column mapping (defaults match standard Screaming Frog exports)
    4. Click "Find Mismatches" and review the results

    **Best for:**
    - Ecommerce navigation audits
    - Keeping menu labels, H1s and titles consistent
    - Spotting renamed categories that were never updated in the navigation
    """)

st.markdown("""
Compare navigation anchor text against destination page H1s and title keywords.
Navigation links are identified as links where the alt text matches the anchor text,
which is typical of image-based navigation and menu templates.
""")

# Sidebar configuration
st.sidebar.header("Settings")

title_separator = st.sidebar.text_input(
    "Title Separator",
    value="|",
    help="Character used to separate the primary keyword from the brand in page titles, e.g. 'Widgets | Example Store' on example.com"
)

# File uploaders
st.header("Upload Crawl Data")

col_upload_1, col_upload_2 = st.columns(2)

with col_upload_1:
    inlinks_file = st.file_uploader(
        "All Inlinks export (CSV)",
        type=["csv"],
        help="Screaming Frog: Bulk Export > Links > All Inlinks. Needs Source, Destination, Alt Text and Anchor columns."
    )

with col_upload_2:
    internal_file = st.file_uploader(
        "Internal HTML export (CSV)",
        type=["csv"],
        help="Screaming Frog: Internal tab, filter HTML, then Export. Needs Address, H1-1 and Title 1 columns."
    )

if inlinks_file is not None and internal_file is not None:
    try:
        df_inlinks = pd.read_csv(inlinks_file, dtype=str)
        df_internal = pd.read_csv(internal_file, dtype=str)

        st.success(f"Loaded {len(df_inlinks):,} inlink rows and {len(df_internal):,} internal HTML rows")

        # Column mapping
        st.subheader("Map Columns")

        inlink_cols = df_inlinks.columns.tolist()
        internal_cols = df_internal.columns.tolist()

        def default_index(columns, name):
            return columns.index(name) if name in columns else 0

        st.markdown("**All Inlinks columns**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            source_col = st.selectbox(
                "Source Column",
                options=inlink_cols,
                index=default_index(inlink_cols, "Source")
            )

        with col2:
            destination_col = st.selectbox(
                "Destination Column",
                options=inlink_cols,
                index=default_index(inlink_cols, "Destination")
            )

        with col3:
            alt_col = st.selectbox(
                "Alt Text Column",
                options=inlink_cols,
                index=default_index(inlink_cols, "Alt Text")
            )

        with col4:
            anchor_col = st.selectbox(
                "Anchor Column",
                options=inlink_cols,
                index=default_index(inlink_cols, "Anchor")
            )

        st.markdown("**Internal HTML columns**")
        col5, col6, col7 = st.columns(3)

        with col5:
            address_col = st.selectbox(
                "Address Column",
                options=internal_cols,
                index=default_index(internal_cols, "Address")
            )

        with col6:
            h1_col = st.selectbox(
                "H1 Column",
                options=internal_cols,
                index=default_index(internal_cols, "H1-1")
            )

        with col7:
            title_col = st.selectbox(
                "Title Column",
                options=internal_cols,
                index=default_index(internal_cols, "Title 1")
            )

        # Show raw data previews
        with st.expander("Preview Raw Data"):
            st.markdown("**All Inlinks**")
            st.dataframe(df_inlinks[[source_col, destination_col, alt_col, anchor_col]].head(20))
            st.markdown("**Internal HTML**")
            st.dataframe(df_internal[[address_col, h1_col, title_col]].head(20))

        # Process button
        if st.button("Find Mismatches", type="primary"):
            with st.spinner("Processing..."):

                # Identify navigation links: alt text matches anchor text
                df_nav = df_inlinks[[source_col, destination_col, alt_col, anchor_col]].copy()
                df_nav.columns = ["Source", "Destination", "Alt Text", "Anchor"]
                df_nav = df_nav[df_nav["Alt Text"] == df_nav["Anchor"]]
                df_nav = df_nav[["Source", "Destination", "Anchor"]]

                if len(df_nav) == 0:
                    st.error("No navigation links found. Navigation links are identified as rows where the alt text matches the anchor text. Check your column mapping.")
                    st.stop()

                st.info(f"Found {len(df_nav):,} navigation links (alt text matches anchor text)")

                # Merge with internal HTML data
                df_pages = df_internal[[address_col, h1_col, title_col]].copy()
                df_pages.columns = ["Address", "H1-1", "Title 1"]

                df_nav = pd.merge(df_nav, df_pages, left_on="Destination", right_on="Address", how="left")
                df_nav = df_nav[df_nav["Address"].notna()]
                del df_nav["Address"]

                if len(df_nav) == 0:
                    st.error("No navigation links matched a page in the Internal HTML export. Check both exports come from the same crawl.")
                    st.stop()

                # Deduplicate on anchor, H1 and title combination
                df_nav.drop_duplicates(subset=["Anchor", "H1-1", "Title 1"], keep="first", inplace=True)

                # Extract the primary keyword from the page title
                df_nav["Page Title Primary KW"] = df_nav["Title 1"].str.split(title_separator, regex=False).str[0]
                del df_nav["Title 1"]
                df_nav["Page Title Primary KW"] = df_nav["Page Title Primary KW"].str.rstrip()

                # Compare navigation label against H1 and title primary keyword
                def normalise(series):
                    return series.fillna("").str.strip().str.casefold()

                anchor_norm = normalise(df_nav["Anchor"])
                df_nav["Anchor Matches H1"] = anchor_norm == normalise(df_nav["H1-1"])
                df_nav["Anchor Matches Title KW"] = anchor_norm == normalise(df_nav["Page Title Primary KW"])
                df_nav["Mismatch"] = ~(df_nav["Anchor Matches H1"] & df_nav["Anchor Matches Title KW"])

                # Sort mismatches to the top
                df_nav = df_nav.sort_values(["Mismatch", "Anchor"], ascending=[False, True])

                # Display results
                st.header("Navigation Label Results")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Navigation Labels Analysed", f"{len(df_nav):,}")
                with col2:
                    mismatch_count = int(df_nav["Mismatch"].sum())
                    st.metric("Mismatched Labels", f"{mismatch_count:,}")
                with col3:
                    match_rate = 1 - (mismatch_count / len(df_nav)) if len(df_nav) else 0
                    st.metric("Match Rate", f"{match_rate:.1%}")

                # Filter options
                show_only_mismatches = st.checkbox("Show only mismatches", value=True)

                df_display = df_nav.copy()
                if show_only_mismatches:
                    df_display = df_display[df_display["Mismatch"]]

                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(f"Showing {len(df_display):,} of {len(df_nav):,} navigation labels")

                # Download full results
                output = BytesIO()
                df_nav.to_csv(output, index=False, encoding="utf-8-sig")
                output.seek(0)

                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=output,
                    file_name="nav_label_mismatches.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload both Screaming Frog exports to get started.")

    st.markdown("""
    ### How to export data from Screaming Frog:

    **All Inlinks export:**
    1. Crawl your website with Screaming Frog Spider
    2. Go to **Bulk Export > Links > All Inlinks**
    3. Save as CSV

    **Internal HTML export:**
    1. In the same crawl, open the **Internal** tab
    2. Set the filter to **HTML**
    3. Click **Export** and save as CSV

    ### Required columns:

    - **All Inlinks**: Source, Destination, Alt Text, Anchor
    - **Internal HTML**: Address, H1-1, Title 1

    ### What this tool does:

    - Identifies navigation links (rows where the alt text matches the anchor text)
    - Joins each navigation link to its destination page's H1 and title
    - Extracts the primary keyword from the page title (the part before the separator)
    - Flags navigation labels that do not match the destination H1 or title keyword
    """)
