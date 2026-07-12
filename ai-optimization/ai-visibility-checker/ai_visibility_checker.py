####################################################################################
#                                                                                  #
#  AI Visibility Checker                                                           #
#                                                                                  #
#  See who gets cited in Google AI Overviews and ChatGPT answers                   #
#  for a given domain's topic space.                                               #
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
AI Visibility Checker

Shows who gets cited in Google AI Overviews and ChatGPT answers
for a given domain's topic space using the DataForSEO AI Optimization API.

Features:
- Check up to 10 entities/domains per query
- Platform selector: Google AI Overviews, ChatGPT, or Both
- Location and language selectors
- Expandable per-mention detail (AI answer, cited sources, fan-out queries)
- Summary stats with top cited domains
- CSV/Excel download
"""

import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
from io import BytesIO
from collections import Counter
from datetime import datetime


st.set_page_config(
    page_title="AI Visibility Checker",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Visibility Checker")
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
    - Queries the DataForSEO AI Optimization API to find LLM mentions of your domain(s)
    - Shows which questions trigger AI answers that cite you (or your competitors)
    - Works with Google AI Overviews, ChatGPT, or both platforms

    **Requirements:**
    - DataForSEO API credentials (login and password)
    - Get credits at [dataforseo.com](https://dataforseo.com)

    **Pricing:**
    - Approximately $0.10 per API call
    - Each call returns up to 1000 mentions

    **Tips:**
    - Enter domains without protocol (e.g. `example.com` not `https://example.com`)
    - You can check up to 10 entities per query
    - Use "Both" platforms to get a complete picture of AI visibility
    """)


# --- Sidebar: API credentials ---
st.sidebar.header("API Credentials")
api_login = st.sidebar.text_input(
    "DataForSEO Login (Email)",
    help="Your DataForSEO account email"
)
api_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    help="Your DataForSEO API password (not your account password)"
)

st.sidebar.markdown("---")
st.sidebar.header("Settings")

# Platform selector
platform_options = {
    "Google AI Overviews": "google",
    "ChatGPT": "chat_gpt",
    "Both": "both"
}
platform_label = st.sidebar.selectbox(
    "Platform",
    list(platform_options.keys()),
    help="Which AI platform to check for citations"
)
platform = platform_options[platform_label]

# Location options
locations = {
    "United Kingdom": "United Kingdom",
    "United States": "United States",
    "Australia": "Australia",
    "Canada": "Canada",
    "Germany": "Germany",
    "France": "France",
    "Spain": "Spain",
    "Italy": "Italy",
    "Netherlands": "Netherlands",
    "Brazil": "Brazil",
    "India": "India",
    "Japan": "Japan"
}

location = st.sidebar.selectbox(
    "Location",
    list(locations.keys()),
    index=0,
    help="Target location for search data"
)

# Language options
languages = {
    "English": "English",
    "German": "German",
    "French": "French",
    "Spanish": "Spanish",
    "Italian": "Italian",
    "Dutch": "Dutch",
    "Portuguese": "Portuguese",
    "Japanese": "Japanese"
}

language = st.sidebar.selectbox(
    "Language",
    list(languages.keys()),
    help="Target language"
)

# Results limit
limit = st.sidebar.slider(
    "Results limit",
    min_value=10,
    max_value=1000,
    value=50,
    step=10,
    help="Maximum number of mentions to retrieve per API call"
)


# --- Helper functions ---

def fetch_llm_mentions(login, password, entities, platform_value, loc, lang, max_results):
    """
    Call the DataForSEO AI Optimization LLM Mentions Search endpoint.

    Returns (list_of_mentions, error_string_or_none).
    """
    url = "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/search/live"

    post_data = [{
        "target": entities,
        "platform": platform_value,
        "location_name": loc,
        "language_name": lang,
        "limit": max_results
    }]

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=120
        )

        data = response.json()

        if data.get("status_code") == 20000:
            mentions = []
            tasks = data.get("tasks", [])

            if tasks and tasks[0].get("result"):
                for result_block in tasks[0]["result"]:
                    items = result_block.get("items", [])
                    if items:
                        mentions.extend(items)
            return mentions, None
        else:
            error_msg = data.get("status_message", "Unknown error")
            task_errors = ""
            tasks = data.get("tasks", [])
            if tasks and tasks[0].get("status_message"):
                task_errors = f" - {tasks[0]['status_message']}"
            return None, f"API Error ({data.get('status_code')}): {error_msg}{task_errors}"

    except requests.exceptions.Timeout:
        return None, "Request timed out. Try reducing the results limit."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def parse_mentions_to_dataframe(mentions):
    """Convert raw mention items to a structured DataFrame."""
    rows = []
    for mention in mentions:
        sources = mention.get("sources") or []
        fan_out = mention.get("fan_out_queries") or []

        rows.append({
            "question": mention.get("question", ""),
            "ai_search_volume": mention.get("ai_search_volume", 0),
            "sources_count": len(sources),
            "fan_out_queries_count": len(fan_out),
            "platform": mention.get("platform", ""),
            "model_name": mention.get("model_name", ""),
            "first_response_at": mention.get("first_response_at", ""),
            "last_response_at": mention.get("last_response_at", ""),
            "monthly_searches": mention.get("monthly_searches", []),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ai_search_volume", ascending=False).reset_index(drop=True)
    return df


def aggregate_cited_domains(mentions):
    """Aggregate source domains across all mentions to find top cited sites."""
    domain_counter = Counter()
    for mention in mentions:
        sources = mention.get("sources") or []
        for source in sources:
            domain = source.get("domain", "")
            if domain:
                domain_counter[domain] += 1
    return domain_counter.most_common(50)


def build_full_export(mentions):
    """Build a flat DataFrame suitable for CSV/Excel export."""
    rows = []
    for mention in mentions:
        sources = mention.get("sources") or []
        fan_out = mention.get("fan_out_queries") or []
        brand_entities = mention.get("brand_entities") or []

        source_domains = "; ".join(
            s.get("domain", "") for s in sources if s.get("domain")
        )
        source_urls = "; ".join(
            s.get("url", "") for s in sources if s.get("url")
        )
        fan_out_list = "; ".join(fan_out)
        brand_list = "; ".join(brand_entities)

        rows.append({
            "question": mention.get("question", ""),
            "ai_search_volume": mention.get("ai_search_volume", 0),
            "platform": mention.get("platform", ""),
            "model_name": mention.get("model_name", ""),
            "sources_count": len(sources),
            "source_domains": source_domains,
            "source_urls": source_urls,
            "fan_out_queries_count": len(fan_out),
            "fan_out_queries": fan_out_list,
            "brand_entities": brand_list,
            "first_response_at": mention.get("first_response_at", ""),
            "last_response_at": mention.get("last_response_at", ""),
            "answer": mention.get("answer", ""),
        })

    return pd.DataFrame(rows)


# --- Main input ---
st.subheader("Enter Domains or Entities to Check")

entity_text = st.text_area(
    "Enter domains or brand entities (one per line, max 10)",
    height=120,
    help="Enter domains (e.g. example.com) or brand names to check for AI citations",
    placeholder="example.com\nanothersite.co.uk"
)

entities = []
if entity_text:
    raw_entities = [e.strip() for e in entity_text.strip().split("\n") if e.strip()]
    if len(raw_entities) > 10:
        st.warning("Maximum 10 entities allowed. Only the first 10 will be used.")
        raw_entities = raw_entities[:10]
    entities = [
        {
            "domain": e,
            "search_filter": "include",
            "search_scope": ["any"],
            "include_subdomains": True,
        }
        for e in raw_entities
    ]
    st.info(f"{len(entities)} entity/entities entered")


# --- Cost estimate and execution ---
if entities:
    call_count = 2 if platform == "both" else 1
    estimated_cost = call_count * 0.10
    st.caption(
        f"Estimated cost: ~${estimated_cost:.2f} "
        f"({call_count} API call{'s' if call_count > 1 else ''})"
    )

if entities and st.button("Check AI Visibility", type="primary"):
    if not api_login or not api_password:
        st.error("Please enter your DataForSEO API credentials in the sidebar.")
        st.stop()

    # Determine which platforms to query
    platforms_to_query = []
    if platform == "both":
        platforms_to_query = ["google", "chat_gpt"]
    else:
        platforms_to_query = [platform]

    all_mentions = []
    errors = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pf in enumerate(platforms_to_query):
        pf_label = "Google AI Overviews" if pf == "google" else "ChatGPT"
        status_text.text(f"Querying {pf_label}...")
        progress_bar.progress((i + 1) / len(platforms_to_query))

        mentions, error = fetch_llm_mentions(
            api_login,
            api_password,
            entities,
            pf,
            locations[location],
            languages[language],
            limit
        )

        if mentions:
            all_mentions.extend(mentions)
        if error:
            errors.append(f"{pf_label}: {error}")

    progress_bar.progress(1.0)
    status_text.text("Complete!")

    if errors:
        for err in errors:
            st.error(err)

    if all_mentions:
        # --- Summary stats ---
        st.subheader("Summary")

        df_summary = parse_mentions_to_dataframe(all_mentions)
        top_domains = aggregate_cited_domains(all_mentions)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Mentions", f"{len(all_mentions):,}")
        with col2:
            total_ai_volume = df_summary["ai_search_volume"].sum()
            st.metric("Total AI Search Volume", f"{int(total_ai_volume):,}")
        with col3:
            unique_questions = df_summary["question"].nunique()
            st.metric("Unique Questions", f"{unique_questions:,}")
        with col4:
            if top_domains:
                st.metric("Top Cited Domains", f"{len(top_domains)}")
            else:
                st.metric("Top Cited Domains", "0")

        # --- Top cited domains ---
        if top_domains:
            st.subheader("Top Cited Domains")
            df_domains = pd.DataFrame(top_domains, columns=["Domain", "Citation Count"])
            st.dataframe(df_domains.head(20), use_container_width=True, hide_index=True)

        # --- Results table ---
        st.subheader("Mentions")
        st.dataframe(
            df_summary[[
                "question", "ai_search_volume", "sources_count",
                "fan_out_queries_count", "platform", "model_name",
                "first_response_at", "last_response_at"
            ]],
            use_container_width=True,
            hide_index=True
        )

        # --- Expandable detail per mention ---
        st.subheader("Mention Details")
        for idx, mention in enumerate(all_mentions[:100]):
            question = mention.get("question", f"Mention {idx + 1}")
            ai_vol = mention.get("ai_search_volume", 0)
            pf_name = mention.get("platform", "")

            with st.expander(f"[{pf_name}] {question} (AI vol: {ai_vol})"):
                # AI Answer
                answer = mention.get("answer", "")
                if answer:
                    st.markdown("**AI Answer:**")
                    st.markdown(answer)
                else:
                    st.caption("No answer text available.")

                st.markdown("---")

                # Cited sources
                sources = mention.get("sources") or []
                if sources:
                    st.markdown(f"**Cited Sources ({len(sources)}):**")
                    for s_idx, source in enumerate(sources):
                        domain = source.get("domain", "N/A")
                        title = source.get("title", "N/A")
                        url = source.get("url", "")
                        position = source.get("position", "N/A")
                        st.markdown(
                            f"{s_idx + 1}. **{domain}** (pos: {position}) "
                            f"- {title}"
                        )
                        if url:
                            st.caption(url)
                else:
                    st.caption("No sources cited.")

                st.markdown("---")

                # Fan-out queries
                fan_out = mention.get("fan_out_queries") or []
                if fan_out:
                    st.markdown(f"**Fan-out Queries ({len(fan_out)}):**")
                    for fq in fan_out:
                        st.markdown(f"- {fq}")
                else:
                    st.caption("No fan-out queries.")

                # Brand entities
                brand_entities = mention.get("brand_entities") or []
                if brand_entities:
                    st.markdown(f"**Brand Entities:** {', '.join(brand_entities)}")

        if len(all_mentions) > 100:
            st.info(
                f"Showing detail for the first 100 mentions. "
                f"Download the full export below for all {len(all_mentions)} results."
            )

        # --- Download ---
        st.subheader("Download Results")

        df_export = build_full_export(all_mentions)

        col_csv, col_excel = st.columns(2)

        with col_csv:
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="ai_visibility_results.csv",
                mime="text/csv"
            )

        with col_excel:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Mentions")
                if top_domains:
                    df_dom = pd.DataFrame(
                        top_domains, columns=["Domain", "Citation Count"]
                    )
                    df_dom.to_excel(
                        writer, index=False, sheet_name="Top Cited Domains"
                    )
            buffer.seek(0)
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name="ai_visibility_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.warning(
            "No mentions found. This could mean the domain has no AI citations yet, "
            "or the API returned no results for this configuration."
        )

elif not entities:
    st.info("Enter one or more domains or brand entities above to check AI visibility.")

    st.subheader("Example Output")
    example_data = {
        "Question": [
            "What is the best project management tool?",
            "How to improve website speed?",
            "What are the top CRM platforms?",
        ],
        "AI Search Volume": [4800, 2900, 3200],
        "Sources Count": [3, 2, 4],
        "Fan-out Queries": [5, 3, 6],
        "Platform": ["google", "chat_gpt", "google"],
        "First Seen": ["2024-11-01", "2024-12-15", "2025-01-03"],
    }
    st.dataframe(pd.DataFrame(example_data), hide_index=True)
