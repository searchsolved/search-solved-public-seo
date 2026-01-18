import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from io import BytesIO
import time
from random import randint

st.set_page_config(page_title="[LEGACY] Keyword Trends Analyzer", page_icon="⚠️", layout="wide")

st.title("⚠️ [LEGACY] Keyword Trends Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

st.warning("**Legacy Tool:** This tool uses the unofficial Google Trends API (pytrends) which is frequently rate-limited and unreliable. Results may be incomplete or the tool may fail entirely. Use with caution.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Analyzes Google Trends data for your keywords
    - Calculates year-over-year trend slope (rising vs declining)
    - Helps identify growing or declining keyword opportunities

    **Files needed:**
    - CSV/Excel with keywords to analyze
    - Optional: search volume data to prioritize

    **How it works:**
    1. Upload your keyword list
    2. Tool queries Google Trends for each keyword (5 at a time)
    3. Calculates trend slope by comparing last year vs previous year
    4. Positive slope = growing trend, Negative = declining

    **Note:** Uses Google Trends API which has rate limits. Large lists may take time.
    """)

# Check if pytrends is available
try:
    from pytrends.request import TrendReq
    pytrends_available = True
except ImportError:
    pytrends_available = False
    st.error("pytrends library not installed. Please install with: pip install pytrends")

# Sidebar settings
st.sidebar.header("Settings")

geo_var = st.sidebar.selectbox(
    "Geographic region",
    ["GB", "US", "CA", "AU", "DE", "FR", "ES", "IT", "NL", "BE", ""],
    help="Leave empty for worldwide"
)

time_var = st.sidebar.selectbox(
    "Time range",
    ["today 5-y", "today 3-y", "today 12-m", "all"],
    help="Time range for trend analysis"
)

random_delay = st.sidebar.slider(
    "Delay between requests (seconds)",
    min_value=1,
    max_value=10,
    value=3,
    help="Delay to avoid rate limiting"
)

# File upload
st.subheader("Upload Your Keywords")

keyword_file = st.file_uploader(
    "Keyword List (CSV/Excel)",
    type=['csv', 'xlsx'],
    help="File with a 'Keyword' column"
)

# Or manual input
manual_keywords = st.text_area(
    "Or enter keywords manually (one per line)",
    placeholder="keyword 1\nkeyword 2\nkeyword 3",
    height=100
)

if (keyword_file or manual_keywords) and pytrends_available:
    try:
        if keyword_file:
            # Load keywords from file
            if keyword_file.name.endswith('.xlsx'):
                df_keywords = pd.read_excel(keyword_file)
            else:
                try:
                    df_keywords = pd.read_csv(keyword_file, encoding='utf-8')
                except:
                    keyword_file.seek(0)
                    df_keywords = pd.read_csv(keyword_file, encoding='latin-1')

            # Find keyword column
            cols = df_keywords.columns.tolist()
            kw_cols = [c for c in cols if 'keyword' in c.lower()]
            kw_col = kw_cols[0] if kw_cols else cols[0]

            keywords = df_keywords[kw_col].dropna().unique().tolist()

            # Check for volume column
            vol_cols = [c for c in cols if 'volume' in c.lower()]
            if vol_cols:
                vol_col = vol_cols[0]
                df_keywords["_volume"] = pd.to_numeric(df_keywords[vol_col], errors='coerce').fillna(0)
        else:
            # Parse manual keywords
            keywords = [k.strip() for k in manual_keywords.split('\n') if k.strip()]
            df_keywords = pd.DataFrame({"Keyword": keywords})
            kw_col = "Keyword"

        st.success(f"Loaded {len(keywords):,} keywords")

        # Limit keywords
        max_keywords = st.sidebar.number_input(
            "Max keywords to analyze",
            min_value=5,
            max_value=500,
            value=min(100, len(keywords)),
            help="Limit keywords to avoid rate limiting"
        )

        keywords = keywords[:max_keywords]

        if st.button("Analyze Trends", type="primary"):
            with st.spinner(f"Analyzing {len(keywords)} keywords (this may take a while)..."):
                # Initialize pytrends
                pytrend = TrendReq(hl='en-US', tz=0)

                # Process in chunks of 5 (Google Trends limit)
                def chunks(lst, n):
                    for i in range(0, len(lst), n):
                        yield lst[i:i + n]

                all_results = []
                progress = st.progress(0)
                status = st.empty()

                keyword_chunks = list(chunks(keywords, 5))

                for i, chunk in enumerate(keyword_chunks):
                    status.text(f"Processing batch {i+1}/{len(keyword_chunks)}: {', '.join(chunk)}")

                    try:
                        pytrend.build_payload(kw_list=chunk, timeframe=time_var, geo=geo_var)
                        interest_df = pytrend.interest_over_time()

                        if not interest_df.empty:
                            # Drop isPartial column if exists
                            if 'isPartial' in interest_df.columns:
                                interest_df = interest_df.drop('isPartial', axis=1)

                            # Calculate slope for each keyword
                            year_today = datetime.now().year
                            last_year = year_today - 1
                            prev_year = year_today - 2

                            # Reset index to get date column
                            interest_df = interest_df.reset_index()

                            for kw in chunk:
                                if kw in interest_df.columns:
                                    # Get last year and previous year data
                                    interest_df['year'] = interest_df['date'].dt.year

                                    last_year_data = interest_df[interest_df['year'] == last_year][kw].mean()
                                    prev_year_data = interest_df[interest_df['year'] == prev_year][kw].mean()

                                    # Handle NaN
                                    last_year_data = last_year_data if pd.notna(last_year_data) else 0
                                    prev_year_data = prev_year_data if pd.notna(prev_year_data) else 0

                                    # Calculate slope
                                    if prev_year_data > 0:
                                        slope = ((last_year_data - prev_year_data) / prev_year_data) * 100
                                    else:
                                        slope = 0

                                    # Get average interest
                                    avg_interest = interest_df[kw].mean()

                                    all_results.append({
                                        "keyword": kw,
                                        "avg_interest": round(avg_interest, 1),
                                        "last_year_avg": round(last_year_data, 1),
                                        "prev_year_avg": round(prev_year_data, 1),
                                        "slope_pct": round(slope, 1),
                                        "trend": "📈 Rising" if slope > 10 else ("📉 Declining" if slope < -10 else "➡️ Stable")
                                    })

                    except Exception as e:
                        st.warning(f"Error processing {chunk}: {str(e)}")

                    progress.progress((i + 1) / len(keyword_chunks))

                    # Random delay
                    if i < len(keyword_chunks) - 1:
                        time.sleep(randint(1, random_delay))

                progress.empty()
                status.empty()

                if all_results:
                    df_results = pd.DataFrame(all_results)

                    # Sort by slope
                    df_results = df_results.sort_values("slope_pct", ascending=False)

                    # Display results
                    st.subheader("Results Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Keywords Analyzed", f"{len(df_results):,}")
                    with col2:
                        rising = (df_results["slope_pct"] > 10).sum()
                        st.metric("Rising Trends 📈", f"{rising:,}")
                    with col3:
                        declining = (df_results["slope_pct"] < -10).sum()
                        st.metric("Declining Trends 📉", f"{declining:,}")
                    with col4:
                        stable = len(df_results) - rising - declining
                        st.metric("Stable ➡️", f"{stable:,}")

                    # Rising trends
                    st.subheader("📈 Top Rising Keywords")
                    df_rising = df_results[df_results["slope_pct"] > 0].head(20)
                    st.dataframe(df_rising, use_container_width=True)

                    # Declining trends
                    st.subheader("📉 Top Declining Keywords")
                    df_declining = df_results[df_results["slope_pct"] < 0].sort_values("slope_pct").head(20)
                    st.dataframe(df_declining, use_container_width=True)

                    # All results
                    st.subheader("All Results")
                    st.dataframe(df_results, use_container_width=True)

                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_results.to_excel(writer, sheet_name='All Results', index=False)
                        df_rising.to_excel(writer, sheet_name='Rising Trends', index=False)
                        df_declining.to_excel(writer, sheet_name='Declining Trends', index=False)

                    st.download_button(
                        label="Download Excel Report",
                        data=output.getvalue(),
                        file_name="keyword_trends_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No trend data retrieved. Try different keywords or check your internet connection.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    if not pytrends_available:
        st.warning("Install pytrends to use this tool: `pip install pytrends`")
    else:
        st.info("Upload a keyword file or enter keywords manually to get started")

    st.subheader("Example Output")
    example_data = {
        "Keyword": ["ai tools", "nft marketplace", "remote work"],
        "Avg Interest": [75, 45, 60],
        "Slope %": [45.2, -32.5, 12.3],
        "Trend": ["📈 Rising", "📉 Declining", "📈 Rising"]
    }
    st.dataframe(pd.DataFrame(example_data))
