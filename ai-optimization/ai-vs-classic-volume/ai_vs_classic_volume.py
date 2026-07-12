####################################################################################
#                                                                                  #
#  AI vs Classic Search Volume                                                     #
#                                                                                  #
#  Compare AI search volume (AI Overviews/ChatGPT) against traditional Google      #
#  search volume per keyword to identify keywords migrating to AI platforms.        #
#                                                                                  #
####################################################################################
# Author: Lee Foot                                                                 #
# Website: https://leefoot.com                                                     #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
AI vs Classic Search Volume

Compares AI search volume (from AI Overviews/ChatGPT) against traditional
Google search volume per keyword to show which keywords are migrating to
AI platforms.

Features:
- CSV upload or paste keywords
- Batched API calls (up to 1000 keywords per request)
- AI share percentage calculation
- Horizontal bar chart comparison
- CSV/Excel download
"""

import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import io
import math

st.set_page_config(
    page_title="AI vs Classic Search Volume",
    page_icon="🤖",
    layout="wide"
)

st.title("AI vs Classic Search Volume")
st.markdown(
    "*Created by* "
    "[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)]"
    "(https://www.leefoot.com) "
    "[![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)]"
    "(https://www.leefoot.com/contact) "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)]"
    "(https://www.linkedin.com/in/lee-foot/) "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)]"
    "(https://bsky.app/profile/leefootseo.bsky.social) "
    "[![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)]"
    "(https://leefoot.com/tools) "
    "[![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)]"
    "(https://github.com/searchsolved/search-solved-public-seo)"
)

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Compares AI search volume (AI Overviews, ChatGPT) against traditional Google search volume
    - Calculates the AI share percentage for each keyword
    - Identifies keywords that are migrating to AI platforms

    **Requirements:**
    - DataForSEO API credentials (login and password)
    - Get free credits at [dataforseo.com](https://dataforseo.com)

    **Pricing (approximate):**
    - AI Keyword Search Volume: ~$0.01 per batch (up to 1000 keywords)
    - Google Ads Search Volume: ~$0.05 per batch (up to 1000 keywords)
    - Total cost shown before you run the query

    **Interpreting results:**
    - **AI Share %** = AI volume / (AI volume + Classic volume) * 100
    - High AI share means the keyword is heavily queried on AI platforms
    - Keywords with 0 AI volume simply have no AI search data yet
    """)

# Sidebar - API credentials
st.sidebar.header("API Credentials")
api_login = st.sidebar.text_input(
    "DataForSEO Login (Email)",
    help="Your DataForSEO account email"
)
api_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    help="Your DataForSEO API password (not account password)"
)

st.sidebar.markdown("---")
st.sidebar.header("Settings")

# Location code mapping
LOCATIONS = {
    "United Kingdom (2826)": 2826,
    "United States (2840)": 2840,
    "Australia (2036)": 2036,
    "Canada (2124)": 2124,
    "Germany (2276)": 2276,
    "France (2250)": 2250,
    "Spain (2724)": 2724,
    "Italy (2380)": 2380,
    "Netherlands (2528)": 2528,
    "Brazil (2076)": 2076,
    "India (2356)": 2356,
    "Japan (2392)": 2392,
}

LANGUAGES = {
    "English (en)": "en",
    "German (de)": "de",
    "French (fr)": "fr",
    "Spanish (es)": "es",
    "Italian (it)": "it",
    "Dutch (nl)": "nl",
    "Portuguese (pt)": "pt",
    "Japanese (ja)": "ja",
}

location_label = st.sidebar.selectbox(
    "Location",
    list(LOCATIONS.keys()),
    index=0,
    help="Target location for search volume data"
)
location_code = LOCATIONS[location_label]

language_label = st.sidebar.selectbox(
    "Language",
    list(LANGUAGES.keys()),
    index=0,
    help="Target language"
)
language_code = LANGUAGES[language_label]


# Main input
st.subheader("Enter Keywords")

input_method = st.radio(
    "Input method",
    ["Paste keywords", "Upload CSV"],
    horizontal=True
)

keywords = []

if input_method == "Paste keywords":
    keyword_text = st.text_area(
        "Enter keywords (one per line)",
        height=200,
        help="Enter your keywords, one per line. Maximum 1000."
    )
    if keyword_text:
        keywords = [k.strip() for k in keyword_text.strip().split('\n') if k.strip()]
        st.info(f"Found {len(keywords)} keywords")

else:
    keyword_file = st.file_uploader(
        "Upload CSV with keywords",
        type=['csv'],
        help="CSV file with a column containing keywords"
    )

    if keyword_file is not None:
        try:
            df_upload = pd.read_csv(keyword_file)
            kw_col = st.selectbox("Select keyword column", df_upload.columns.tolist())
            keywords = df_upload[kw_col].dropna().astype(str).str.strip().tolist()
            keywords = [k for k in keywords if k]
            st.info(f"Found {len(keywords)} keywords")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Enforce limit
if len(keywords) > 1000:
    st.warning("Maximum 1000 keywords per run. Only the first 1000 will be processed.")
    keywords = keywords[:1000]


def estimate_cost(num_keywords):
    """Estimate the API cost for a given number of keywords."""
    ai_batches = math.ceil(num_keywords / 1000)
    classic_batches = math.ceil(num_keywords / 1000)
    ai_cost = ai_batches * 0.01
    classic_cost = classic_batches * 0.05
    return ai_cost + classic_cost


def fetch_ai_volume(login, password, kw_list, loc_code, lang_code):
    """Fetch AI search volume from DataForSEO AI Keyword Data endpoint."""
    url = "https://api.dataforseo.com/v3/ai_optimization/ai_keyword_data/keywords_search_volume/live"

    post_data = [{
        "keywords": kw_list,
        "location_code": loc_code,
        "language_code": lang_code,
    }]

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=120
        )
        response_data = response.json()

        if response_data.get("status_code") == 20000:
            results = {}
            tasks = response_data.get("tasks", [])

            for task in tasks:
                task_result = task.get("result")
                if not task_result:
                    continue

                # Handle both response shapes defensively:
                # Shape A: result is a list of keyword objects directly
                # Shape B: result contains items[] with keyword data
                for item in task_result:
                    if isinstance(item, dict):
                        # Check if this dict has keyword + search_volume directly
                        if "keyword" in item:
                            kw = item["keyword"]
                            vol = item.get("search_volume") or 0
                            results[kw] = vol
                        # Or if it has nested items
                        elif "items" in item:
                            for sub_item in item["items"]:
                                if isinstance(sub_item, dict) and "keyword" in sub_item:
                                    kw = sub_item["keyword"]
                                    vol = sub_item.get("search_volume") or 0
                                    results[kw] = vol

            return results, None
        else:
            error_msg = response_data.get("status_message", "Unknown error")
            return None, f"AI Volume API Error: {error_msg}"

    except requests.exceptions.RequestException as e:
        return None, f"AI Volume request failed: {str(e)}"
    except Exception as e:
        return None, f"AI Volume error: {str(e)}"


def fetch_classic_volume(login, password, kw_list, loc_code, lang_code):
    """Fetch traditional Google Ads search volume from DataForSEO."""
    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"

    post_data = [{
        "keywords": kw_list,
        "location_code": loc_code,
        "language_code": lang_code,
    }]

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=120
        )
        response_data = response.json()

        if response_data.get("status_code") == 20000:
            results = {}
            tasks = response_data.get("tasks", [])

            for task in tasks:
                task_result = task.get("result")
                if not task_result:
                    continue

                for item in task_result:
                    if isinstance(item, dict) and "keyword" in item:
                        kw = item["keyword"]
                        vol = item.get("search_volume") or 0
                        results[kw] = {
                            "classic_search_volume": vol,
                            "competition": item.get("competition", 0),
                            "cpc": item.get("cpc", 0),
                        }

            return results, None
        else:
            error_msg = response_data.get("status_message", "Unknown error")
            return None, f"Classic Volume API Error: {error_msg}"

    except requests.exceptions.RequestException as e:
        return None, f"Classic Volume request failed: {str(e)}"
    except Exception as e:
        return None, f"Classic Volume error: {str(e)}"


# Show cost estimate and run button
if keywords:
    cost = estimate_cost(len(keywords))
    st.info(
        f"Estimated cost: **${cost:.2f}** "
        f"(AI volume: ${math.ceil(len(keywords)/1000) * 0.01:.2f} + "
        f"Classic volume: ${math.ceil(len(keywords)/1000) * 0.05:.2f})"
    )

if keywords and st.button("Compare AI vs Classic Volume", type="primary"):
    if not api_login or not api_password:
        st.error("Please enter your DataForSEO API credentials in the sidebar.")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Batch keywords (max 1000 per request)
    batches = [keywords[i:i + 1000] for i in range(0, len(keywords), 1000)]

    # Fetch AI volume
    status_text.text("Fetching AI search volume...")
    ai_results = {}
    ai_errors = []

    for idx, batch in enumerate(batches):
        result, error = fetch_ai_volume(api_login, api_password, batch, location_code, language_code)
        if result:
            ai_results.update(result)
        if error:
            ai_errors.append(error)
        progress_bar.progress(0.25 * (idx + 1) / len(batches))

    # Fetch classic volume
    status_text.text("Fetching classic Google search volume...")
    classic_results = {}
    classic_errors = []

    for idx, batch in enumerate(batches):
        result, error = fetch_classic_volume(api_login, api_password, batch, location_code, language_code)
        if result:
            classic_results.update(result)
        if error:
            classic_errors.append(error)
        progress_bar.progress(0.5 + 0.25 * (idx + 1) / len(batches))

    # Combine results
    status_text.text("Combining results...")
    progress_bar.progress(0.85)

    rows = []
    for kw in keywords:
        ai_vol = ai_results.get(kw, 0) or 0
        classic_data = classic_results.get(kw, {})
        classic_vol = classic_data.get("classic_search_volume", 0) if isinstance(classic_data, dict) else 0
        classic_vol = classic_vol or 0

        total = ai_vol + classic_vol
        ai_share = round((ai_vol / total) * 100, 1) if total > 0 else 0.0
        delta = ai_vol - classic_vol

        rows.append({
            "keyword": kw,
            "ai_search_volume": int(ai_vol),
            "classic_search_volume": int(classic_vol),
            "ai_share_pct": ai_share,
            "delta": int(delta),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("ai_share_pct", ascending=False).reset_index(drop=True)

    progress_bar.progress(1.0)
    status_text.text("Complete!")

    # Show errors if any
    all_errors = ai_errors + classic_errors
    if all_errors:
        with st.expander(f"API Errors ({len(all_errors)})"):
            for err in all_errors:
                st.warning(err)

    # Summary stats
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Keywords", f"{len(df):,}")
    with col2:
        avg_share = df["ai_share_pct"].mean()
        st.metric("Average AI Share", f"{avg_share:.1f}%")
    with col3:
        ai_wins = int((df["ai_search_volume"] > df["classic_search_volume"]).sum())
        st.metric("AI > Classic", f"{ai_wins:,}")
    with col4:
        classic_wins = int((df["classic_search_volume"] > df["ai_search_volume"]).sum())
        st.metric("Classic > AI", f"{classic_wins:,}")

    # Results table
    st.subheader("Results")
    st.dataframe(
        df.style.format({
            "ai_search_volume": "{:,.0f}",
            "classic_search_volume": "{:,.0f}",
            "ai_share_pct": "{:.1f}%",
            "delta": "{:+,.0f}",
        }),
        use_container_width=True,
        height=500
    )

    # Visualisation - top 20 by AI share
    st.subheader("Top 20 Keywords by AI Share")

    df_chart = df.head(20).copy()
    df_chart = df_chart.sort_values("ai_share_pct", ascending=True)  # ascending for horizontal bar

    chart_data = df_chart[["keyword", "ai_search_volume", "classic_search_volume"]].set_index("keyword")
    st.bar_chart(chart_data, horizontal=True)

    # Downloads
    st.subheader("Download Results")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_output,
            file_name="ai_vs_classic_volume.csv",
            mime="text/csv"
        )

    with col_dl2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="AI vs Classic Volume")
        excel_data = excel_buffer.getvalue()
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name="ai_vs_classic_volume.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif not keywords:
    st.info("Enter keywords above to compare AI vs classic search volume.")

    st.subheader("Example Output")
    example_data = {
        "keyword": ["best crm software", "how to train a puppy", "weather tomorrow"],
        "ai_search_volume": [4200, 8100, 12000],
        "classic_search_volume": [6600, 14800, 33100],
        "ai_share_pct": [38.9, 35.4, 26.6],
        "delta": [-2400, -6700, -21100],
    }
    st.dataframe(pd.DataFrame(example_data))
