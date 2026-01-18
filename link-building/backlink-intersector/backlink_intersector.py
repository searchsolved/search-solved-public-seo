import streamlit as st
import pandas as pd
from io import BytesIO
import re

st.set_page_config(page_title="Backlink Intersector", page_icon="🔗", layout="wide")

st.title("Backlink Intersector")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Finds link building opportunities by intersecting competitor backlink profiles
    - Identifies pages linking to multiple competitors but not to you
    - Filters by traffic to prioritize high-value opportunities

    **Files needed:**
    1. **Your backlinks** - Export from Ahrefs/SEMrush/Moz with at least "Referring Page URL" column
    2. **Competitor backlinks** - Multiple exports (one per competitor) with:
       - Referring Page URL
       - Traffic (optional but recommended)
       - Link URL (the page being linked to)

    **How to export from Ahrefs:**
    1. Go to Site Explorer > Backlinks
    2. Export as CSV
    3. Repeat for each competitor
    """)

# Sidebar settings
st.sidebar.header("Settings")

min_traffic = st.sidebar.number_input(
    "Minimum referring page traffic",
    min_value=0,
    value=10,
    help="Only include referring pages with at least this much traffic"
)

min_competitors = st.sidebar.number_input(
    "Minimum competitors linking",
    min_value=1,
    value=2,
    help="Referring page must link to at least this many competitors"
)

# File uploads
st.subheader("Upload Your Data")

your_backlinks = st.file_uploader(
    "Your Backlinks (single file)",
    type=['csv'],
    help="Your site's backlink export"
)

competitor_backlinks = st.file_uploader(
    "Competitor Backlinks (multiple files)",
    type=['csv'],
    accept_multiple_files=True,
    help="Upload backlink exports for each competitor"
)

if your_backlinks and competitor_backlinks and len(competitor_backlinks) > 0:
    try:
        # Load your backlinks
        try:
            df_yours = pd.read_csv(your_backlinks, encoding='utf-8')
        except:
            your_backlinks.seek(0)
            df_yours = pd.read_csv(your_backlinks, encoding='latin-1')

        # Load competitor backlinks
        competitor_dfs = []
        for i, f in enumerate(competitor_backlinks):
            try:
                df = pd.read_csv(f, encoding='utf-8')
            except:
                f.seek(0)
                df = pd.read_csv(f, encoding='latin-1')
            df['_competitor'] = f.name
            competitor_dfs.append(df)

        df_competitors = pd.concat(competitor_dfs, ignore_index=True)

        st.success(f"Loaded {len(df_yours):,} of your backlinks and {len(df_competitors):,} competitor backlinks from {len(competitor_backlinks)} files")

        # Column mapping
        with st.expander("Map columns (if needed)"):
            your_cols = df_yours.columns.tolist()
            comp_cols = df_competitors.columns.tolist()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Your backlinks:**")
                # Find referring page column
                ref_cols = [c for c in your_cols if 'referring' in c.lower() and 'page' in c.lower()]
                default_ref = your_cols.index(ref_cols[0]) if ref_cols else 0
                your_ref_col = st.selectbox(
                    "Referring Page URL",
                    your_cols,
                    index=default_ref,
                    key="your_ref"
                )

            with col2:
                st.markdown("**Competitor backlinks:**")
                ref_cols_c = [c for c in comp_cols if 'referring' in c.lower() and 'page' in c.lower()]
                default_ref_c = comp_cols.index(ref_cols_c[0]) if ref_cols_c else 0
                comp_ref_col = st.selectbox(
                    "Referring Page URL",
                    comp_cols,
                    index=default_ref_c,
                    key="comp_ref"
                )

                # Traffic column
                traffic_cols = [c for c in comp_cols if 'traffic' in c.lower()]
                default_traffic = comp_cols.index(traffic_cols[0]) if traffic_cols else 0
                traffic_col = st.selectbox(
                    "Traffic column",
                    comp_cols,
                    index=default_traffic
                )

                # Link URL column
                link_cols = [c for c in comp_cols if 'link' in c.lower() and 'url' in c.lower()]
                default_link = comp_cols.index(link_cols[0]) if link_cols else 0
                link_col = st.selectbox(
                    "Link URL column (page being linked to)",
                    comp_cols,
                    index=default_link
                )

                # Type column
                type_cols = [c for c in comp_cols if 'type' in c.lower()]
                default_type = comp_cols.index(type_cols[0]) if type_cols else None
                type_col = st.selectbox(
                    "Link Type column (optional)",
                    ["None"] + comp_cols,
                    index=0 if default_type is None else default_type + 1
                )

        if st.button("Find Opportunities", type="primary"):
            with st.spinner("Analyzing backlink intersection..."):
                # Normalize column names
                df_yours = df_yours.rename(columns={your_ref_col: "referring_page"})
                df_competitors = df_competitors.rename(columns={
                    comp_ref_col: "referring_page",
                    traffic_col: "traffic",
                    link_col: "link_url"
                })

                # Ensure traffic is numeric
                df_competitors["traffic"] = pd.to_numeric(df_competitors["traffic"], errors='coerce').fillna(0)

                # Filter by traffic
                df_competitors = df_competitors[df_competitors["traffic"] >= min_traffic]

                # Get your referring pages as a list
                your_links = df_competitors["referring_page"].str.lower().unique().tolist()

                # Mark which competitor links you already have
                df_competitors["already_have"] = df_competitors["referring_page"].str.lower().isin(
                    df_yours["referring_page"].str.lower().unique()
                )

                # Keep only links you don't have
                df_opportunities = df_competitors[~df_competitors["already_have"]].copy()

                # Extract domain from link_url for deduplication
                def extract_domain(url):
                    if pd.isna(url):
                        return None
                    match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', str(url))
                    return match.group(1) if match else url

                df_opportunities["target_domain"] = df_opportunities["link_url"].apply(extract_domain)

                # Drop duplicate opportunities (same referring page linking to same domain)
                df_opportunities = df_opportunities.drop_duplicates(
                    subset=["referring_page", "target_domain"],
                    keep="first"
                )

                # Count how many competitors each referring page links to
                df_opportunities["competitor_count"] = df_opportunities.groupby("referring_page")["referring_page"].transform("count")

                # Filter by minimum competitors
                df_opportunities = df_opportunities[df_opportunities["competitor_count"] >= min_competitors]

                # Aggregate by referring page
                df_grouped = df_opportunities.groupby("referring_page").agg({
                    "traffic": "mean",
                    "competitor_count": "first",
                    "_competitor": lambda x: ", ".join(x.unique())
                }).reset_index()

                df_grouped = df_grouped.rename(columns={
                    "_competitor": "competitors_with_links",
                    "traffic": "avg_traffic"
                })

                # Sort by competitor count and traffic
                df_grouped = df_grouped.sort_values(
                    ["competitor_count", "avg_traffic"],
                    ascending=[False, False]
                )

                # Display summary
                st.subheader("Results Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Opportunities", f"{len(df_grouped):,}")
                with col2:
                    st.metric("Links to 2+ Competitors", f"{(df_grouped['competitor_count'] >= 2).sum():,}")
                with col3:
                    st.metric("Links to 3+ Competitors", f"{(df_grouped['competitor_count'] >= 3).sum():,}")
                with col4:
                    st.metric("Avg Traffic", f"{df_grouped['avg_traffic'].mean():.0f}")

                # Display results
                st.subheader("Link Building Opportunities")
                st.dataframe(df_grouped, use_container_width=True)

                # Top opportunities
                st.subheader("Top 20 Opportunities (by competitor count)")
                top_opps = df_grouped.nlargest(20, "competitor_count")
                st.dataframe(top_opps, use_container_width=True)

                # Download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_grouped.to_excel(writer, sheet_name='All Opportunities', index=False)
                    top_opps.to_excel(writer, sheet_name='Top 20', index=False)

                    # Detailed view
                    df_opportunities[['referring_page', 'link_url', 'traffic', 'competitor_count', '_competitor']].to_excel(
                        writer, sheet_name='Detailed', index=False
                    )

                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name="backlink_intersection.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload your backlinks and at least one competitor backlink file to get started")

    st.subheader("Example Output")
    example_data = {
        "Referring Page": ["example.com/best-tools", "blog.site.com/roundup", "review.com/top-10"],
        "Avg Traffic": [500, 1200, 800],
        "Competitor Count": [3, 2, 2],
        "Competitors": ["comp1.csv, comp2.csv, comp3.csv", "comp1.csv, comp2.csv", "comp2.csv, comp3.csv"]
    }
    st.dataframe(pd.DataFrame(example_data))
