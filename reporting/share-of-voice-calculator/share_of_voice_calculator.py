import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Share of Voice Calculator", page_icon="📊", layout="wide")

st.title("Share of Voice Calculator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")
st.warning("**Experimental Tool** - This is a proof of concept and may have limitations.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Calculates Share of Voice (SOV) from Ahrefs traffic-by-domain data
    - Filters out unwanted domains (forums, aggregators, etc.)
    - Shows percentage of total traffic for each competitor
    - Outputs Excel with formatted data

    **How to get the data:**
    1. Go to Ahrefs > Keywords Explorer
    2. Enter your target keywords
    3. Go to Traffic share > By domains
    4. Export the CSV

    **Best for:**
    - Competitive analysis
    - Understanding market share in organic search
    - Tracking visibility changes over time
    """)

# Sidebar settings
st.sidebar.header("Settings")
top_n = st.sidebar.slider(
    "Number of top domains",
    min_value=5,
    max_value=50,
    value=10,
    help="Show top N domains by traffic"
)

# File uploads
st.subheader("Upload Data")

col1, col2 = st.columns(2)

with col1:
    traffic_file = st.file_uploader(
        "Ahrefs Traffic by Domain CSV",
        type=['csv'],
        help="Export from Ahrefs Keywords Explorer > Traffic share > By domains"
    )

with col2:
    bad_domains_file = st.file_uploader(
        "Bad Domains to Exclude (optional)",
        type=['csv'],
        help="CSV with a column of domains to filter out (forums, aggregators, etc.)"
    )

if traffic_file is not None:
    try:
        # Load traffic data
        try:
            df_traffic = pd.read_csv(traffic_file, encoding='utf-8')
        except:
            traffic_file.seek(0)
            df_traffic = pd.read_csv(traffic_file, encoding='latin-1')

        st.success(f"Loaded {len(df_traffic):,} domains")

        # Try to identify the relevant columns
        # Ahrefs exports can have different column names
        url_col = None
        traffic_col = None

        for col in df_traffic.columns:
            col_lower = col.lower()
            if 'url' in col_lower or 'domain' in col_lower:
                url_col = col
            if 'traffic' in col_lower and 'share' not in col_lower:
                traffic_col = col

        if url_col is None:
            url_col = st.selectbox("Select URL/Domain column", df_traffic.columns.tolist())

        if traffic_col is None:
            traffic_col = st.selectbox("Select Traffic column", df_traffic.columns.tolist())

        with st.expander("Preview uploaded data"):
            st.dataframe(df_traffic.head(20))

        # Load bad domains if provided
        bad_domains = []
        if bad_domains_file is not None:
            try:
                df_bad = pd.read_csv(bad_domains_file, encoding='utf-8')
            except:
                bad_domains_file.seek(0)
                df_bad = pd.read_csv(bad_domains_file, encoding='latin-1')

            # Use first column as bad domains list
            bad_domains = df_bad.iloc[:, 0].str.lower().tolist()
            st.info(f"Loaded {len(bad_domains)} domains to exclude")

        if st.button("Calculate Share of Voice", type="primary"):
            with st.spinner("Calculating SOV..."):
                df_work = df_traffic.copy()

                # Ensure traffic is numeric
                df_work[traffic_col] = pd.to_numeric(df_work[traffic_col], errors='coerce')
                df_work = df_work.dropna(subset=[traffic_col])

                # Filter out bad domains
                if bad_domains:
                    df_work["_exclude"] = df_work[url_col].str.lower().str.contains(
                        '|'.join(bad_domains), case=False, na=False
                    )
                    excluded_count = df_work["_exclude"].sum()
                    df_work = df_work[~df_work["_exclude"]]
                    del df_work["_exclude"]
                    st.info(f"Filtered out {excluded_count} domains")

                # Keep only top N
                df_work = df_work.head(top_n)

                # Calculate total traffic for SOV
                total_traffic = df_work[traffic_col].sum()

                # Calculate SOV percentage
                df_work['Share of Voice (%)'] = (df_work[traffic_col] / total_traffic * 100).round(2)

                # Reset index for display
                df_work = df_work.reset_index(drop=True)
                df_work.index = df_work.index + 1  # Start from 1

                # Display summary
                st.subheader("Share of Voice Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Domains Analyzed", f"{len(df_work):,}")
                with col2:
                    st.metric("Total Traffic", f"{int(total_traffic):,}")
                with col3:
                    leader = df_work.iloc[0][url_col] if len(df_work) > 0 else "N/A"
                    leader_sov = df_work.iloc[0]['Share of Voice (%)'] if len(df_work) > 0 else 0
                    st.metric("Market Leader", f"{leader_sov}%")

                # Show bar chart
                st.bar_chart(
                    df_work.set_index(url_col)['Share of Voice (%)'].head(10),
                    use_container_width=True
                )

                # Show data table
                st.dataframe(df_work, use_container_width=True)

                # Create Excel output
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_work.to_excel(writer, sheet_name='Share of Voice', index=True)

                    workbook = writer.book
                    worksheet = writer.sheets['Share of Voice']

                    # Add a chart
                    chart = workbook.add_chart({'type': 'column'})
                    chart.add_series({
                        'categories': f"='Share of Voice'!$B$2:$B${min(11, len(df_work)+1)}",
                        'values': f"='Share of Voice'!$D$2:$D${min(11, len(df_work)+1)}",
                        'gap': 10,
                    })
                    chart.set_y_axis({'major_gridlines': {'visible': False}})
                    chart.set_legend({'position': 'none'})
                    chart.set_title({'name': 'Share of Voice by Domain'})
                    worksheet.insert_chart('F2', chart, {'x_scale': 1.5, 'y_scale': 1.2})

                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name="share_of_voice.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload an Ahrefs Traffic by Domain export to get started")

    st.subheader("Example Output")
    example_data = {
        "Domain": ["competitor-a.com", "competitor-b.com", "your-site.com", "competitor-c.com"],
        "Traffic": [15000, 12000, 8000, 5000],
        "Share of Voice (%)": [37.5, 30.0, 20.0, 12.5]
    }
    st.dataframe(pd.DataFrame(example_data))
