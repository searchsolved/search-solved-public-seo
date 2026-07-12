####################################################################################
#                                                                                  #
#  Fan-Out Query Explorer                                                          #
#                                                                                  #
#  Surfaces the sub-questions (fan-out queries) that AI generates when             #
#  answering queries in your topic space. Content-planning angle: answer the       #
#  fan-outs to get cited in AI responses.                                          #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Fan-Out Query Explorer

Surfaces the sub-questions (fan-out queries) that AI generates when answering
queries in your topic space. Use this to discover content gaps and plan content
that gets cited in AI responses.

Uses the DataForSEO LLM Mentions Search endpoint.

Author: Lee Foot
Website: https://leefoot.com
"""

import os
import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from collections import Counter
import json

st.set_page_config(
    page_title="Fan-Out Query Explorer",
    page_icon="🔀",
    layout="wide"
)

st.title("Fan-Out Query Explorer")
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
    - Queries the DataForSEO LLM Mentions endpoint to find which questions AI
      platforms (Google AI Overview, ChatGPT) associate with your keyword or domain
    - Extracts and deduplicates the **fan-out queries** (sub-questions the AI
      generates while researching its answer)
    - Ranks them by frequency so you can prioritise content creation

    **Why it matters:**
    - AI systems break complex queries into sub-questions (fan-out queries)
    - If your content answers those sub-questions, you are more likely to be cited
    - This tool reveals the sub-questions so you can build content around them

    **Requirements:**
    - DataForSEO API credentials (login and password)
    - Get credits at [dataforseo.com](https://dataforseo.com)

    **Estimated cost:**
    - Approximately $0.10 per request to the LLM Mentions Search endpoint
    """)

# ─── Sidebar: API credentials ─────────────────────────────────────────────────

st.sidebar.header("API Credentials")

api_login = st.sidebar.text_input(
    "DataForSEO Login (Email)",
    value=os.environ.get("DATAFORSEO_LOGIN", ""),
    help="Your DataForSEO account email"
)
api_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    value=os.environ.get("DATAFORSEO_PASSWORD", ""),
    help="Your DataForSEO API password (not account password)"
)

st.sidebar.markdown("---")
st.sidebar.header("Settings")

# Target type
target_type = st.sidebar.radio(
    "Target type",
    ["Keyword", "Domain"],
    help="Search by keyword topic or by domain mentions"
)

# Platform
platform = st.sidebar.selectbox(
    "AI Platform",
    ["google", "chat_gpt"],
    format_func=lambda x: "Google AI Overview" if x == "google" else "ChatGPT",
    help="Which AI platform to query"
)

# Location
locations = {
    "United Kingdom": 2826,
    "United States": 2840,
    "Australia": 2036,
    "Canada": 2124,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "India": 2356,
    "Brazil": 2076,
    "Japan": 2392,
}

location_name = st.sidebar.selectbox(
    "Location",
    list(locations.keys()),
    index=0,
    help="Target location for the query"
)

# Language
languages = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl",
    "Portuguese": "pt",
    "Japanese": "ja",
}

language_name = st.sidebar.selectbox(
    "Language",
    list(languages.keys()),
    index=0,
    help="Target language"
)

# Limit
limit = st.sidebar.slider(
    "Results limit",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Maximum mention items to retrieve (more = more fan-out queries discovered)"
)


# ─── Main input ───────────────────────────────────────────────────────────────

st.subheader("Enter Target")

if target_type == "Keyword":
    target_input = st.text_input(
        "Keyword or topic",
        placeholder="e.g. welding helmets",
        help="The keyword or topic to explore fan-out queries for"
    )
    include_subdomains = False
else:
    target_input = st.text_input(
        "Domain",
        placeholder="e.g. example.com",
        help="The domain to find fan-out queries for"
    )
    include_subdomains = st.checkbox(
        "Include subdomains",
        value=True,
        help="Include subdomains of the target domain"
    )

# Cost estimate
if target_input:
    st.info(f"Estimated cost: ~$0.10 for this request ({limit} items max)")


# ─── API call function ────────────────────────────────────────────────────────

def fetch_llm_mentions(login, password, target, target_is_domain, platform_choice,
                       loc_code, lang_code, result_limit, subdomains=True):
    """
    Call the DataForSEO LLM Mentions Search endpoint and return parsed results.

    Returns:
        tuple: (mentions_list, error_message)
            mentions_list is a list of dicts with keys:
                question, fan_out_queries, ai_search_volume, sources, brand_entities
    """
    url = "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/search/live"

    # Build the target entity per DataForSEO spec
    if target_is_domain:
        entity = {
            "domain": target,
            "search_filter": "include",
            "search_scope": ["any"],
            "include_subdomains": subdomains,
        }
    else:
        entity = {
            "keyword": target,
            "search_filter": "include",
            "search_scope": ["any"],
            "match_type": "word_match",
        }

    payload = {
        "target": [entity],
        "platform": platform_choice,
        "location_code": loc_code,
        "language_code": lang_code,
        "limit": result_limit,
    }

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps([payload]),
            timeout=120,
        )

        data = response.json()

        if data.get("status_code") != 20000:
            error_msg = data.get("status_message", "Unknown API error")
            return None, f"API Error ({data.get('status_code')}): {error_msg}"

        tasks = data.get("tasks", [])
        if not tasks or not tasks[0].get("result"):
            return [], None

        result = tasks[0]["result"][0]
        items = result.get("items", [])

        mentions = []
        for item in items:
            question = item.get("question", "")
            fan_outs = item.get("fan_out_queries", []) or []
            ai_sv = item.get("ai_search_volume", 0)
            sources = item.get("sources", []) or []
            brands = item.get("brand_entities", []) or []

            mentions.append({
                "question": question,
                "fan_out_queries": fan_outs,
                "ai_search_volume": ai_sv,
                "source_count": len(sources),
                "sources": sources,
                "brand_entities": brands,
                "fan_out_count": len(fan_outs),
            })

        return mentions, None

    except requests.exceptions.Timeout:
        return None, "Request timed out. Try reducing the limit."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except (KeyError, IndexError, TypeError) as e:
        return None, f"Error parsing response: {str(e)}"


# ─── Processing ───────────────────────────────────────────────────────────────

if target_input and st.button("Explore Fan-Out Queries", type="primary"):
    if not api_login or not api_password:
        st.error("Please enter your DataForSEO API credentials in the sidebar.")
        st.stop()

    with st.spinner("Querying DataForSEO LLM Mentions endpoint..."):
        mentions, error = fetch_llm_mentions(
            login=api_login,
            password=api_password,
            target=target_input.strip(),
            target_is_domain=(target_type == "Domain"),
            platform_choice=platform,
            loc_code=locations[location_name],
            lang_code=languages[language_name],
            result_limit=limit,
            subdomains=include_subdomains,
        )

    if error:
        st.error(error)
        st.stop()

    if not mentions:
        st.warning("No mentions returned for this target. Try a broader keyword or different platform.")
        st.stop()

    # ─── Build fan-out frequency table ────────────────────────────────────────

    # Collect all fan-out queries with their parent questions
    fan_out_records = []
    for mention in mentions:
        parent_q = mention["question"]
        for fq in mention["fan_out_queries"]:
            fan_out_records.append({
                "fan_out_query": fq,
                "parent_question": parent_q,
            })

    if not fan_out_records:
        st.warning("Mentions were returned but none contained fan-out queries.")
        st.stop()

    df_fan_outs_raw = pd.DataFrame(fan_out_records)

    # Deduplicate and count frequency
    fan_out_counter = Counter(df_fan_outs_raw["fan_out_query"].tolist())

    # Build parent questions mapping (semicolon-separated for CSV friendliness)
    fan_out_parents = (
        df_fan_outs_raw.groupby("fan_out_query")["parent_question"]
        .apply(lambda x: "; ".join(sorted(set(x))))
        .reset_index()
    )
    fan_out_parents.columns = ["fan_out_query", "parent_questions"]

    # Merge frequency
    df_fan_outs = pd.DataFrame(
        [(q, count) for q, count in fan_out_counter.most_common()],
        columns=["fan_out_query", "frequency"]
    )
    df_fan_outs = df_fan_outs.merge(fan_out_parents, on="fan_out_query", how="left")

    # ─── Build parent questions table ─────────────────────────────────────────

    df_parents = pd.DataFrame(mentions)
    df_parents = df_parents[["question", "fan_out_count", "ai_search_volume", "source_count"]]
    df_parents = df_parents.sort_values("fan_out_count", ascending=False)
    df_parents.columns = ["Parent Question", "Fan-Out Count", "AI Search Volume", "Source Count"]

    # ─── Display results ──────────────────────────────────────────────────────

    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Parent Questions", f"{len(mentions):,}")
    with col2:
        st.metric("Total Fan-Out Queries", f"{len(df_fan_outs_raw):,}")
    with col3:
        st.metric("Unique Fan-Outs", f"{len(df_fan_outs):,}")
    with col4:
        avg_fan_outs = len(df_fan_outs_raw) / len(mentions) if mentions else 0
        st.metric("Avg Fan-Outs per Question", f"{avg_fan_outs:.1f}")

    # PRIMARY OUTPUT: Fan-out queries ranked by frequency
    st.subheader("Fan-Out Queries (by frequency)")
    st.caption(
        "These are the sub-questions AI generates when researching answers. "
        "Higher frequency means the query appears under multiple parent questions."
    )

    st.dataframe(
        df_fan_outs,
        use_container_width=True,
        column_config={
            "fan_out_query": st.column_config.TextColumn("Fan-Out Query", width="large"),
            "frequency": st.column_config.NumberColumn("Times Appeared", format="%d"),
            "parent_questions": st.column_config.TextColumn("Parent Questions", width="large"),
        },
        hide_index=True,
    )

    # CLUSTERING: group fan-out queries by semantic similarity
    if len(df_fan_outs) >= 3:
        st.subheader("Clustered Fan-Out Queries")
        st.caption(
            "Fan-out queries grouped by semantic similarity using sentence-transformers. "
            "Clusters are named after their shortest member."
        )

        with st.spinner("Clustering queries..."):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_distances

            queries = df_fan_outs["fan_out_query"].tolist()
            tfidf = TfidfVectorizer(stop_words="english", min_df=1)
            matrix = tfidf.fit_transform(queries)
            distance_matrix = cosine_distances(matrix)

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.7,
                metric="precomputed",
                linkage="average",
            )
            labels = clustering.fit_predict(distance_matrix)

            cluster_labels = []
            from collections import defaultdict
            groups = defaultdict(list)
            for i, label in enumerate(labels):
                groups[label].append(i)

            label_names = {}
            for label, members in groups.items():
                if len(members) >= 2:
                    label_names[label] = min((queries[i] for i in members), key=len)
                else:
                    label_names[label] = "Unclustered"

            df_fan_outs["cluster"] = [label_names[l] for l in labels]

        clustered = df_fan_outs[df_fan_outs["cluster"] != "Unclustered"]
        unclustered = df_fan_outs[df_fan_outs["cluster"] == "Unclustered"]

        cluster_summary = (
            clustered.groupby("cluster")
            .agg(queries=("fan_out_query", "count"), total_frequency=("frequency", "sum"))
            .sort_values("total_frequency", ascending=False)
            .reset_index()
        )
        cluster_summary.columns = ["Cluster Name", "Queries in Cluster", "Total Frequency"]

        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        with st.expander(f"View all clustered queries ({len(clustered)} queries in {len(cluster_summary)} clusters)"):
            st.dataframe(
                clustered[["cluster", "fan_out_query", "frequency"]].sort_values(["cluster", "frequency"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True,
            )

        if len(unclustered) > 0:
            with st.expander(f"Unclustered queries ({len(unclustered)})"):
                st.dataframe(unclustered[["fan_out_query", "frequency"]], use_container_width=True, hide_index=True)

        # Add cluster to download
        csv_clustered = df_fan_outs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Clustered Fan-Outs CSV",
            data=csv_clustered,
            file_name="fan_out_queries_clustered.csv",
            mime="text/csv",
        )

    # SECONDARY OUTPUT: Parent questions table
    st.subheader("Parent Questions")
    st.caption(
        "The original questions that triggered these fan-out queries, "
        "with their AI search volume and source counts."
    )

    st.dataframe(
        df_parents,
        use_container_width=True,
        hide_index=True,
    )

    # ─── Downloads ────────────────────────────────────────────────────────────

    st.subheader("Download")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_fan_outs = df_fan_outs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Fan-Out Queries CSV",
            data=csv_fan_outs,
            file_name="fan_out_queries.csv",
            mime="text/csv",
        )

    with col_dl2:
        csv_parents = df_parents.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Parent Questions CSV",
            data=csv_parents,
            file_name="parent_questions.csv",
            mime="text/csv",
        )

elif not target_input:
    st.info("Enter a keyword or domain above to explore fan-out queries.")

    st.subheader("Example Output")
    example_data = {
        "Fan-Out Query": [
            "what is the best auto-darkening shade for MIG",
            "how much does a welding helmet weigh",
            "can you wear glasses under a welding helmet",
        ],
        "Frequency": [4, 3, 2],
        "Parent Questions": [
            "best welding helmets 2025; top auto-darkening helmets",
            "welding helmet comfort; lightweight welding gear",
            "welding helmet fit guide; welding PPE for glasses wearers",
        ],
    }
    st.dataframe(pd.DataFrame(example_data), hide_index=True)
