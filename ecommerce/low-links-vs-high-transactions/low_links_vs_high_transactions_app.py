"""
Low Links vs High Transactions - Streamlit App

Identifies high-value pages that lack internal linking opportunities.
Merges GSC internal links data with GA landing page data.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Low Links vs High Transactions",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Low Links vs High Transactions")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Identifies high-value pages with low internal links
    - Compares transaction data to link equity
    - Prioritizes internal linking opportunities

    **How to use:**
    1. Upload transaction/conversion data
    2. Upload internal link data (from crawl)
    3. Configure thresholds
    4. Review prioritized opportunities

    **Best for:**
    - Internal linking prioritization
    - Revenue-driven SEO decisions
    - Link equity optimization
    """)
st.markdown("Find high-value pages that need more internal links.")


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    except:
        return None


def process_data(df_gsc, df_ga, keep_percentage):
    """Process and merge GSC and GA data."""
    # Get domain from GSC data
    first_url = df_gsc.iloc[0]['Target page']
    domain = extract_domain(first_url)

    # Add domain to GA landing pages
    df_ga = df_ga.copy()
    df_ga['Landing Page'] = domain + df_ga['Landing Page'].astype(str)

    # Merge dataframes
    df_combined = pd.merge(
        df_gsc, df_ga,
        left_on="Target page",
        right_on="Landing Page",
        how="inner"
    )

    # Clean up columns
    if 'Landing Page' in df_combined.columns:
        df_combined = df_combined.drop(columns=['Landing Page'])

    # Drop unnecessary GA columns if present
    cols_to_drop = ['% New Sessions', 'New Users', 'Bounce Rate',
                    'Pages/Session', 'Avg. Session Duration', 'E-commerce Conversion Rate']
    for col in cols_to_drop:
        if col in df_combined.columns:
            df_combined = df_combined.drop(columns=[col])

    # Filter out zero transactions
    if 'Transactions' in df_combined.columns:
        df_combined = df_combined[df_combined['Transactions'] > 0]

    # Round floats
    df_combined = df_combined.round(2)

    # Calculate lowest X% of links threshold
    if 'Internal links' in df_combined.columns:
        max_links = df_combined['Internal links'].max()
        threshold = max_links * (keep_percentage / 100)
        df_combined = df_combined[df_combined['Internal links'] <= threshold]

    # Sort by internal links (ascending) and transactions (descending)
    sort_cols = []
    if 'Internal links' in df_combined.columns:
        sort_cols.append('Internal links')
    if 'Transactions' in df_combined.columns:
        sort_cols.append('Transactions')

    if sort_cols:
        df_combined = df_combined.sort_values(
            sort_cols,
            ascending=[True, False] if len(sort_cols) == 2 else [True]
        )

    return df_combined, domain


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    keep_percentage = st.slider(
        "Keep Bottom % of Links",
        min_value=5,
        max_value=50,
        value=10,
        help="Only show pages with internal links in the bottom X%"
    )

    st.markdown("---")
    st.markdown("### 📖 Data Export Guide")
    st.markdown("""
    **Google Search Console:**
    1. Go to Links → Internal Links
    2. Click 'Top linked pages - internally'
    3. Export as CSV

    **Google Analytics:**
    1. Go to Behavior → Site Content → Landing Pages
    2. Add 'Transactions' metric
    3. Export as Excel (.xlsx)
    """)

# Main content
st.markdown("### 📤 Upload Data Files")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### GSC Internal Links Export")
    gsc_file = st.file_uploader(
        "Upload GSC CSV",
        type=["csv"],
        key="gsc",
        help="Export from: Search Console > Links > Internal Links > Top linked pages"
    )

with col2:
    st.markdown("#### GA Landing Page Export")
    ga_file = st.file_uploader(
        "Upload GA Excel",
        type=["xlsx", "xls"],
        key="ga",
        help="Export from: Analytics > Behavior > Site Content > Landing Pages"
    )

if gsc_file and ga_file:
    try:
        # Load GSC data
        df_gsc = pd.read_csv(gsc_file)

        # Check for required columns
        if 'Target page' not in df_gsc.columns:
            possible_cols = [c for c in df_gsc.columns if 'page' in c.lower() or 'url' in c.lower()]
            if possible_cols:
                df_gsc = df_gsc.rename(columns={possible_cols[0]: 'Target page'})
            else:
                st.error("GSC file must have a 'Target page' column")
                st.stop()

        # Load GA data
        try:
            df_ga = pd.read_excel(ga_file, sheet_name="Dataset1")
        except:
            df_ga = pd.read_excel(ga_file, sheet_name=0)

        # Check for required columns
        if 'Landing Page' not in df_ga.columns:
            possible_cols = [c for c in df_ga.columns if 'page' in c.lower() or 'url' in c.lower()]
            if possible_cols:
                df_ga = df_ga.rename(columns={possible_cols[0]: 'Landing Page'})
            else:
                st.error("GA file must have a 'Landing Page' column")
                st.stop()

        st.success("✅ Files loaded successfully!")

        # Preview data
        with st.expander("Preview Uploaded Data"):
            st.markdown("**GSC Data:**")
            st.dataframe(df_gsc.head(), use_container_width=True)
            st.markdown("**GA Data:**")
            st.dataframe(df_ga.head(), use_container_width=True)

        if st.button("🔍 Find Opportunities", type="primary", use_container_width=True):
            with st.spinner("Processing data..."):
                df_result, domain = process_data(df_gsc, df_ga, keep_percentage)

            if len(df_result) > 0:
                st.success(f"✅ Found {len(df_result)} pages with low links but high transactions!")
                st.info(f"Domain: {domain}")

                # Results tabs
                tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Visualization", "💡 Insights"])

                with tab1:
                    st.dataframe(df_result, use_container_width=True, height=400)

                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results CSV",
                        data=csv,
                        file_name="low_links_high_transactions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with tab2:
                    # Scatter plot
                    if 'Internal links' in df_result.columns and 'Transactions' in df_result.columns:
                        fig = px.scatter(
                            df_result,
                            x='Internal links',
                            y='Transactions',
                            hover_data=['Target page'],
                            title='Internal Links vs Transactions',
                            labels={
                                'Internal links': 'Number of Internal Links',
                                'Transactions': 'Number of Transactions'
                            }
                        )
                        fig.update_traces(marker=dict(size=10))
                        st.plotly_chart(fig, use_container_width=True)

                    # Top opportunities
                    if 'Transactions' in df_result.columns:
                        st.subheader("🏆 Top Opportunities")
                        top_10 = df_result.nlargest(10, 'Transactions')

                        fig_bar = px.bar(
                            top_10,
                            x='Target page',
                            y='Transactions',
                            title='Top 10 Pages by Transactions (with low internal links)'
                        )
                        fig_bar.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_bar, use_container_width=True)

                with tab3:
                    st.subheader("💡 Key Insights")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if 'Transactions' in df_result.columns:
                            total_transactions = df_result['Transactions'].sum()
                            st.metric("Total Transactions", f"{total_transactions:,.0f}")

                    with col2:
                        if 'Internal links' in df_result.columns:
                            avg_links = df_result['Internal links'].mean()
                            st.metric("Avg Internal Links", f"{avg_links:.1f}")

                    with col3:
                        st.metric("Pages Found", len(df_result))

                    st.markdown("---")
                    st.markdown("""
                    ### 🎯 Recommendations

                    These pages have high transaction value but few internal links.
                    Consider:

                    1. **Add contextual links** from related category pages
                    2. **Include in main navigation** if applicable
                    3. **Link from blog content** about related topics
                    4. **Add to footer/sidebar** widgets
                    5. **Create hub pages** that link to these products
                    """)
            else:
                st.warning("No pages found matching the criteria. Try increasing the percentage threshold.")

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")

else:
    st.info("👆 Upload both GSC and GA files to get started")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool helps identify **internal linking opportunities** by finding pages that:

        - Have **high transaction value** (from Google Analytics)
        - But **few internal links** pointing to them (from Search Console)

        These are prime candidates for internal linking optimization to boost their visibility
        and drive more traffic to high-converting pages.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
