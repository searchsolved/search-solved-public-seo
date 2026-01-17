import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Core Update Analyser", page_icon="📊", layout="wide")

st.title("Core Update Analyser")
st.markdown("*Created by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **Ahrefs Export Setup:**
    1. Go to Ahrefs > Organic Keywords > Organic Keywords 2.0
    2. Compare two dates (before and after a core update)
    3. Export the "Position changes" report as CSV

    **Required columns:**
    - Keyword, Volume, SERP features
    - Previous URL, Previous position, Previous date
    - Current URL, Current position, Current date

    **What this tool does:**
    - Groups keyword changes by URL folder/subfolder
    - Shows which site sections gained or lost rankings
    - Breaks down by position ranges: 1-3, 4-10, 11-100
    - Outputs an Excel file with analysis at each URL depth level
    """)


def process_ahrefs_data(df_ahrefs):
    """Process Ahrefs organic keywords export and analyze by URL folder."""

    # Count URL folder depth
    df_ahrefs["Folder Depth"] = df_ahrefs["Previous URL"].str.count("/") - 1
    max_depth = int(df_ahrefs["Folder Depth"].max())

    # Clean URLs
    for col in ["Previous URL", "Current URL"]:
        df_ahrefs[col] = df_ahrefs[col].str.replace(r"\r\n.*", "", regex=True)
        df_ahrefs[col] = df_ahrefs[col].str.replace(".html", "")
        df_ahrefs[col] = df_ahrefs[col].str.split("?").str[0]

    # Create position range flags
    df_ahrefs.loc[df_ahrefs["Previous position"] < 4, "Prev Top 3"] = "prev_top_three"
    df_ahrefs.loc[df_ahrefs["Previous position"] < 11, "Prev First Page"] = "prev_first_page"
    df_ahrefs.loc[df_ahrefs["Previous position"] > 10, "Prev Page 2+"] = "prev_second_page"

    df_ahrefs.loc[df_ahrefs["Current position"] < 4, "Curr Top 3"] = "curr_top_three"
    df_ahrefs.loc[df_ahrefs["Current position"] < 11, "Curr First Page"] = "curr_first_page"
    df_ahrefs.loc[df_ahrefs["Current position"] > 10, "Curr Page 2+"] = "curr_second_page"

    # Create filtered dataframes for each position range
    df_prev_top_3 = df_ahrefs[df_ahrefs["Prev Top 3"] == "prev_top_three"].copy()
    df_prev_first_page = df_ahrefs[df_ahrefs["Prev First Page"] == "prev_first_page"].copy()
    df_prev_page_2 = df_ahrefs[df_ahrefs["Prev Page 2+"] == "prev_second_page"].copy()
    df_curr_top_3 = df_ahrefs[df_ahrefs["Curr Top 3"] == "curr_top_three"].copy()
    df_curr_first_page = df_ahrefs[df_ahrefs["Curr First Page"] == "curr_first_page"].copy()
    df_curr_page_2 = df_ahrefs[df_ahrefs["Curr Page 2+"] == "curr_second_page"].copy()

    # Copy position values for grouping
    df_prev_top_3["Prev Top 3"] = df_prev_top_3["Previous position"]
    df_prev_first_page["Prev First Page"] = df_prev_first_page["Previous position"]
    df_prev_page_2["Prev Page 2+"] = df_prev_page_2["Previous position"]
    df_curr_top_3["Curr Top 3"] = df_curr_top_3["Current position"]
    df_curr_first_page["Curr First Page"] = df_curr_first_page["Current position"]
    df_curr_page_2["Curr Page 2+"] = df_curr_page_2["Current position"]

    # Create prev and current base dataframes
    df_prev = df_ahrefs[["Keyword", "SERP features", "Volume", "Previous position", "Previous URL",
                         "Prev Top 3", "Prev First Page", "Prev Page 2+"]].copy()
    df_curr = df_ahrefs[["Keyword", "SERP features", "Volume", "Current position", "Current URL",
                         "Curr Top 3", "Curr First Page", "Curr Page 2+"]].copy()

    df_appended_data = []

    # Loop through folder depths
    for depth in range(1, max_depth + 1):
        split_depth = depth + 1

        df_prev["Folder Name"] = df_prev["Previous URL"].str.split("/").str[split_depth]
        df_curr["Current Folder"] = df_curr["Current URL"].str.split("/").str[split_depth]

        # Group main metrics
        df_prev_grp = df_prev.groupby("Folder Name", as_index=False).agg({
            "Volume": "sum", "Keyword": "count", "Previous position": "median"
        })
        df_curr_grp = df_curr.groupby("Current Folder", as_index=False).agg({
            "Volume": "sum", "Keyword": "count", "Current position": "median"
        })

        # Group position range dataframes
        for df_temp in [df_prev_top_3, df_prev_first_page, df_prev_page_2,
                        df_curr_top_3, df_curr_first_page, df_curr_page_2]:
            url_col = "Previous URL" if "Prev" in df_temp.columns[6] else "Current URL"
            df_temp["Folder Name"] = df_temp[url_col].str.split("/").str[split_depth]

        # Aggregate by position range
        df_prev_top_3_grp = df_prev_top_3.groupby("Folder Name", as_index=False).agg({
            "Volume": "sum", "Prev Top 3": "count", "Previous position": "median"
        }).rename(columns={"Prev Top 3": "1-3 - Prev KWs", "Volume": "1-3 - Prev Volume",
                          "Previous position": "1-3 - Prev Avg. Position"})

        df_prev_first_page_grp = df_prev_first_page.groupby("Folder Name", as_index=False).agg({
            "Volume": "sum", "Prev First Page": "count", "Previous position": "median"
        }).rename(columns={"Prev First Page": "1-10 - Prev KWs", "Volume": "1-10 - Prev Volume",
                          "Previous position": "1-10 - Prev Avg. Position"})

        df_prev_page_2_grp = df_prev_page_2.groupby("Folder Name", as_index=False).agg({
            "Volume": "sum", "Prev Page 2+": "count", "Previous position": "median"
        }).rename(columns={"Prev Page 2+": "11-100 - Prev KWs", "Volume": "11-100 - Prev Volume",
                          "Previous position": "11-100 - Prev Avg. Position"})

        df_curr_top_3_grp = df_curr_top_3.groupby("Folder Name", as_index=False).agg({
            "Volume": "sum", "Curr Top 3": "count", "Current position": "median"
        }).rename(columns={"Curr Top 3": "1-3 - Curr KWs", "Volume": "1-3 - Curr Volume",
                          "Current position": "1-3 - Curr Avg. Position"})

        df_curr_first_page_grp = df_curr_first_page.groupby("Folder Name", as_index=False).agg({
            "Volume": "sum", "Curr First Page": "count", "Current position": "median"
        }).rename(columns={"Curr First Page": "1-10 - Curr KWs", "Volume": "1-10 - Curr Volume",
                          "Current position": "1-10 - Curr Avg. Position"})

        df_curr_page_2_grp = df_curr_page_2.groupby("Folder Name", as_index=False).agg({
            "Volume": "sum", "Curr Page 2+": "count", "Current position": "median"
        }).rename(columns={"Curr Page 2+": "11-100 - Curr KWs", "Volume": "11-100 - Curr Volume",
                          "Current position": "11-100 - Curr Avg. Position"})

        # Merge all grouped data
        df_merge = pd.merge(df_prev_grp, df_curr_grp, left_on="Folder Name", right_on="Current Folder", how="left")
        for grp_df in [df_prev_top_3_grp, df_prev_first_page_grp, df_prev_page_2_grp,
                       df_curr_top_3_grp, df_curr_first_page_grp, df_curr_page_2_grp]:
            df_merge = pd.merge(df_merge, grp_df, on="Folder Name", how="left")

        df_merge["Current Folder"] = df_merge["Current Folder"].fillna("Lost / URL Changed")
        df_merge["Depth"] = depth
        df_appended_data.append(df_merge)

    # Concatenate all depths
    df_result = pd.concat(df_appended_data, ignore_index=True)

    # Rename columns
    df_result.rename(columns={
        "Volume_x": "1-100 - Prev Volume", "Keyword_x": "1-100 - Prev KWs",
        "Previous position": "1-100 - Prev Avg. Position",
        "Volume_y": "1-100 - Curr Volume", "Keyword_y": "1-100 - Curr KWs",
        "Current position": "1-100 - Curr Avg. Position"
    }, inplace=True)

    # Calculate change columns
    df_result["1-100 +/- KW Change"] = df_result["1-100 - Curr KWs"] - df_result["1-100 - Prev KWs"]
    df_result["1-3 +/- KW Change"] = df_result.get("1-3 - Curr KWs", 0) - df_result.get("1-3 - Prev KWs", 0)
    df_result["1-10 +/- KW Change"] = df_result.get("1-10 - Curr KWs", 0) - df_result.get("1-10 - Prev KWs", 0)
    df_result["11-100 +/- KW Change"] = df_result.get("11-100 - Curr KWs", 0) - df_result.get("11-100 - Prev KWs", 0)

    df_result["1-100 +/- Vol Change"] = df_result["1-100 - Curr Volume"] - df_result["1-100 - Prev Volume"]
    df_result["1-3 +/- Vol Change"] = df_result.get("1-3 - Curr Volume", 0) - df_result.get("1-3 - Prev Volume", 0)
    df_result["1-10 +/- Vol Change"] = df_result.get("1-10 - Curr Volume", 0) - df_result.get("1-10 - Prev Volume", 0)
    df_result["11-100 +/- Vol Change"] = df_result.get("11-100 - Curr Volume", 0) - df_result.get("11-100 - Prev Volume", 0)

    # Fill NaN values
    change_cols = [c for c in df_result.columns if "+/-" in c]
    df_result[change_cols] = df_result[change_cols].fillna(0)

    # Select final columns
    final_cols = ["Depth", "Folder Name", "1-100 +/- KW Change", "1-3 +/- KW Change",
                  "1-10 +/- KW Change", "11-100 +/- KW Change", "1-100 +/- Vol Change",
                  "1-3 +/- Vol Change", "1-10 +/- Vol Change", "11-100 +/- Vol Change"]
    df_result = df_result[[c for c in final_cols if c in df_result.columns]]

    # Round and sort
    df_result = df_result.round(2)
    df_result = df_result.sort_values(by="1-100 +/- KW Change", ascending=False)

    # Create domain summary
    df_summary = df_result.groupby(lambda x: "Total").agg({
        c: "sum" for c in df_result.columns if "+/-" in c
    }).reset_index(drop=True)
    df_summary.insert(0, "Summary", "Entire Domain")

    return df_result, df_summary, max_depth


# File upload
uploaded_file = st.file_uploader("Upload Ahrefs Organic Keywords Position Changes CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Try different encodings
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        # Check required columns
        required_cols = ["Keyword", "Volume", "Previous URL", "Previous position", "Current URL", "Current position"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            st.stop()

        st.success(f"Loaded {len(df):,} keyword position changes")

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(20))

        if st.button("📊 Analyze Core Update Impact", type="primary"):
            with st.spinner("Analyzing keyword changes by URL folder..."):
                df_result, df_summary, max_depth = process_ahrefs_data(df)

                # Display summary
                st.subheader("Domain Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    kw_change = df_summary["1-100 +/- KW Change"].iloc[0] if "1-100 +/- KW Change" in df_summary.columns else 0
                    st.metric("Total KW Change (1-100)", f"{int(kw_change):+,}")
                with col2:
                    top3_change = df_summary["1-3 +/- KW Change"].iloc[0] if "1-3 +/- KW Change" in df_summary.columns else 0
                    st.metric("Top 3 KW Change", f"{int(top3_change):+,}")
                with col3:
                    page1_change = df_summary["1-10 +/- KW Change"].iloc[0] if "1-10 +/- KW Change" in df_summary.columns else 0
                    st.metric("Page 1 KW Change", f"{int(page1_change):+,}")
                with col4:
                    vol_change = df_summary["1-100 +/- Vol Change"].iloc[0] if "1-100 +/- Vol Change" in df_summary.columns else 0
                    st.metric("Total Volume Change", f"{int(vol_change):+,}")

                # Display by folder
                st.subheader("Changes by URL Folder")

                # Filter by depth
                depth_filter = st.selectbox("Filter by URL Depth", ["All Depths"] + list(range(1, max_depth + 1)))

                if depth_filter != "All Depths":
                    display_df = df_result[df_result["Depth"] == depth_filter]
                else:
                    display_df = df_result

                st.dataframe(display_df, use_container_width=True)

                # Create Excel output
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_summary.to_excel(writer, sheet_name='Domain Summary', index=False)
                    df_result.to_excel(writer, sheet_name='All Folders', index=False)

                    # Add sheet for each depth
                    for depth in range(1, max_depth + 1):
                        depth_df = df_result[df_result["Depth"] == depth]
                        if len(depth_df) > 0:
                            depth_df.to_excel(writer, sheet_name=f'Depth {depth}', index=False)

                # Download button
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name="core_update_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Upload an Ahrefs Organic Keywords position changes CSV to get started")

    st.subheader("Example Output")
    example_data = {
        "Depth": [1, 1, 2, 2],
        "Folder Name": ["blog", "products", "category-a", "category-b"],
        "1-100 +/- KW Change": [+150, -75, +45, -30],
        "1-3 +/- KW Change": [+12, -8, +5, -3],
        "1-10 +/- KW Change": [+35, -20, +15, -10],
        "1-100 +/- Vol Change": [+25000, -15000, +8000, -5000]
    }
    st.dataframe(pd.DataFrame(example_data))
