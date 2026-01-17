####################################################################################
#                                                                                  #
#  Share of Voice Calculator (CTR Curve Based)                                     #
#                                                                                  #
#  Calculate organic traffic share using ranking positions and CTR curves.         #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                                   #
# Contact  : https://www.leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Share of Voice Calculator (CTR Curve Based)

Calculates estimated organic traffic share using ranking positions and
industry-standard CTR curves (Sistrix, Advanced Web Ranking, etc.).
Works with pre-scraped SERP data or keyword tracking exports.

Features:
- Upload CSV with keyword, volume, position, domain data
- Configurable CTR curves (Sistrix, AWR, Custom)
- Group by domain or by category
- Export with traffic estimates
"""

import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Share of Voice Calculator", page_icon="📊", layout="wide")

st.title("Share of Voice Calculator")
st.markdown("*Created by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Calculates Share of Voice (SOV) from ranking data
    - Uses CTR curves to estimate organic traffic
    - Aggregates by domain to show market share

    **Data requirements:**
    Your CSV should have these columns:
    - `keyword` - The search query
    - `volume` - Monthly search volume
    - `position` - Ranking position (1-10)
    - `domain` - The domain ranking
    - `category` (optional) - For grouping analysis

    **Where to get this data:**
    - SERP API exports (ValueSERP, DataForSEO)
    - Rank tracking tools (Ahrefs, SEMrush, Sistrix)
    - Custom crawl data

    **CTR Curves:**
    - Position 1: ~28-35% of clicks
    - Position 2: ~15-17% of clicks
    - Position 3: ~10-12% of clicks
    - And so on...
    """)

# CTR curve presets
CTR_CURVES = {
    "Sistrix (2020)": {
        1: 0.2848, 2: 0.1548, 3: 0.1100, 4: 0.0765, 5: 0.0535,
        6: 0.0491, 7: 0.0306, 8: 0.0313, 9: 0.0279, 10: 0.0274
    },
    "Advanced Web Ranking (Desktop)": {
        1: 0.3100, 2: 0.1560, 3: 0.0990, 4: 0.0700, 5: 0.0512,
        6: 0.0390, 7: 0.0305, 8: 0.0248, 9: 0.0209, 10: 0.0182
    },
    "Backlinko (2023)": {
        1: 0.2750, 2: 0.1510, 3: 0.1120, 4: 0.0810, 5: 0.0740,
        6: 0.0510, 7: 0.0410, 8: 0.0330, 9: 0.0290, 10: 0.0260
    },
    "Conservative Estimate": {
        1: 0.2000, 2: 0.1000, 3: 0.0800, 4: 0.0600, 5: 0.0500,
        6: 0.0400, 7: 0.0350, 8: 0.0300, 9: 0.0250, 10: 0.0200
    }
}

# Sidebar settings
st.sidebar.header("CTR Curve Settings")

ctr_preset = st.sidebar.selectbox(
    "CTR Curve Preset",
    list(CTR_CURVES.keys()),
    help="Select a CTR curve based on industry research"
)

selected_ctr = CTR_CURVES[ctr_preset].copy()

# Custom CTR adjustment
st.sidebar.markdown("---")
st.sidebar.subheader("Custom CTR Values")
use_custom = st.sidebar.checkbox("Customize CTR values")

if use_custom:
    for pos in range(1, 11):
        selected_ctr[pos] = st.sidebar.number_input(
            f"Position {pos} CTR",
            min_value=0.0,
            max_value=1.0,
            value=selected_ctr[pos],
            format="%.4f",
            key=f"ctr_{pos}"
        )

st.sidebar.markdown("---")
st.sidebar.header("Analysis Settings")

top_n = st.sidebar.slider(
    "Top N domains to show",
    min_value=5,
    max_value=100,
    value=20
)

group_by_category = st.sidebar.checkbox(
    "Group by category",
    value=False,
    help="Show SOV within each category"
)

# File upload
st.subheader("Upload Ranking Data")
ranking_file = st.file_uploader(
    "Upload CSV with ranking data",
    type=['csv'],
    help="CSV with keyword, volume, position, domain columns"
)


def find_column(df, possible_names):
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        for col in df.columns:
            if name.lower() == col.lower() or name.lower() in col.lower():
                return col
    return None


if ranking_file is not None:
    try:
        # Load data
        try:
            df = pd.read_csv(ranking_file, encoding='utf-8')
        except:
            ranking_file.seek(0)
            df = pd.read_csv(ranking_file, encoding='latin-1')

        st.success(f"Loaded {len(df):,} rows")

        # Find columns
        keyword_col = find_column(df, ['keyword', 'query', 'search term'])
        volume_col = find_column(df, ['volume', 'search volume', 'sv', 'monthly volume'])
        position_col = find_column(df, ['position', 'rank', 'ranking', 'pos'])
        domain_col = find_column(df, ['domain', 'url', 'site', 'website'])
        category_col = find_column(df, ['category', 'group', 'vertical', 'topic'])

        with st.expander("Column Mapping"):
            col1, col2 = st.columns(2)
            with col1:
                keyword_col = st.selectbox("Keyword column", df.columns.tolist(),
                                           index=df.columns.tolist().index(keyword_col) if keyword_col else 0)
                volume_col = st.selectbox("Volume column", df.columns.tolist(),
                                          index=df.columns.tolist().index(volume_col) if volume_col else 0)
            with col2:
                position_col = st.selectbox("Position column", df.columns.tolist(),
                                            index=df.columns.tolist().index(position_col) if position_col else 0)
                domain_col = st.selectbox("Domain column", df.columns.tolist(),
                                          index=df.columns.tolist().index(domain_col) if domain_col else 0)

            if group_by_category:
                category_col = st.selectbox("Category column", df.columns.tolist(),
                                            index=df.columns.tolist().index(category_col) if category_col else 0)

        with st.expander("Preview data"):
            st.dataframe(df.head(20))

        if st.button("Calculate Share of Voice", type="primary"):
            with st.spinner("Calculating SOV..."):
                df_work = df.copy()

                # Ensure numeric types
                df_work[volume_col] = pd.to_numeric(df_work[volume_col], errors='coerce')
                df_work[position_col] = pd.to_numeric(df_work[position_col], errors='coerce')

                # Filter to top 10 positions only
                df_work = df_work[(df_work[position_col] >= 1) & (df_work[position_col] <= 10)]
                df_work[position_col] = df_work[position_col].astype(int)

                # Apply CTR curve
                df_work['ctr'] = df_work[position_col].map(selected_ctr)

                # Calculate estimated traffic
                df_work['estimated_traffic'] = (df_work['ctr'] * df_work[volume_col]).round(0)

                # Display CTR curve used
                st.subheader("CTR Curve Applied")
                ctr_df = pd.DataFrame({
                    'Position': list(selected_ctr.keys()),
                    'CTR (%)': [f"{v*100:.2f}%" for v in selected_ctr.values()]
                })
                st.dataframe(ctr_df.T, use_container_width=True)

                if group_by_category and category_col:
                    # Group by category and domain
                    grouped = df_work.groupby([category_col, domain_col]).agg({
                        'estimated_traffic': 'sum',
                        keyword_col: 'count'
                    }).reset_index()
                    grouped.columns = [category_col, 'Domain', 'Estimated Traffic', 'Keywords']

                    # Calculate SOV within each category
                    category_totals = grouped.groupby(category_col)['Estimated Traffic'].transform('sum')
                    grouped['SOV (%)'] = (grouped['Estimated Traffic'] / category_totals * 100).round(2)

                    # Sort within categories
                    grouped = grouped.sort_values([category_col, 'Estimated Traffic'], ascending=[True, False])
                    grouped = grouped.groupby(category_col).head(top_n)

                    st.subheader("Share of Voice by Category")

                    # Show each category
                    for category in grouped[category_col].unique():
                        st.markdown(f"### {category}")
                        cat_data = grouped[grouped[category_col] == category]
                        st.dataframe(cat_data, use_container_width=True)
                        st.bar_chart(cat_data.set_index('Domain')['SOV (%)'].head(10))

                else:
                    # Group by domain only
                    grouped = df_work.groupby(domain_col).agg({
                        'estimated_traffic': 'sum',
                        keyword_col: 'count'
                    }).reset_index()
                    grouped.columns = ['Domain', 'Estimated Traffic', 'Keywords']

                    # Calculate total for SOV
                    total_traffic = grouped['Estimated Traffic'].sum()
                    grouped['SOV (%)'] = (grouped['Estimated Traffic'] / total_traffic * 100).round(2)

                    # Sort and limit
                    grouped = grouped.sort_values('Estimated Traffic', ascending=False)
                    grouped = grouped.head(top_n)

                    # Display results
                    st.subheader("Share of Voice Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Domains Analyzed", f"{len(grouped):,}")
                    with col2:
                        st.metric("Total Est. Traffic", f"{int(total_traffic):,}")
                    with col3:
                        unique_kws = df_work[keyword_col].nunique()
                        st.metric("Keywords Tracked", f"{unique_kws:,}")
                    with col4:
                        leader_sov = grouped.iloc[0]['SOV (%)'] if len(grouped) > 0 else 0
                        st.metric("Leader SOV", f"{leader_sov}%")

                    # Bar chart
                    st.subheader("SOV Distribution")
                    chart_data = grouped.head(15).set_index('Domain')['SOV (%)']
                    st.bar_chart(chart_data)

                    # Full table
                    st.subheader("Detailed Results")
                    grouped.index = range(1, len(grouped) + 1)
                    st.dataframe(grouped, use_container_width=True)

                # Download
                st.subheader("Download")

                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    grouped.to_excel(writer, sheet_name='Share of Voice', index=True)

                    # Add raw data sheet
                    df_work.to_excel(writer, sheet_name='Raw Data', index=False)

                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name="share_of_voice_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload a CSV with ranking data to begin")

    st.subheader("Required Data Format")
    example_data = {
        "keyword": ["seo tools", "seo tools", "seo tools", "keyword research"],
        "volume": [12100, 12100, 12100, 8100],
        "position": [1, 2, 3, 1],
        "domain": ["ahrefs.com", "semrush.com", "moz.com", "ahrefs.com"],
        "category": ["SEO Software", "SEO Software", "SEO Software", "SEO Software"]
    }
    st.dataframe(pd.DataFrame(example_data))

    st.subheader("Example Output")
    output_example = {
        "Domain": ["ahrefs.com", "semrush.com", "moz.com"],
        "Estimated Traffic": [5752, 1873, 1331],
        "Keywords": [2, 1, 1],
        "SOV (%)": [64.2, 20.9, 14.9]
    }
    st.dataframe(pd.DataFrame(output_example))
