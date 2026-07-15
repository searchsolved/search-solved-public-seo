# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  Keyword Difficulty Checker                                                      #
#                                                                                  #
#  Check keyword difficulty and search intent using the DataForSEO Labs API.       #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                      #
####################################################################################

import math
import os
from base64 import b64encode
from io import BytesIO

import chardet
import pandas as pd
import requests
import streamlit as st
from stqdm import stqdm

st.set_page_config(
    page_title="Keyword Difficulty Checker",
    page_icon="mag",
    layout="wide",
)

st.title("Keyword Difficulty Checker")
st.markdown(
    "*Created by* "
    "[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) "
    "· [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) "
    "· [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) "
    "· [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) "
    "· [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) "
    "· [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)"
)

with st.expander("How to use this tool"):
    st.markdown(
        """
    **What this tool does:**
    - Retrieves keyword difficulty scores (0 to 100) via the DataForSEO Labs API
    - Classifies search intent (informational, navigational, commercial, transactional)
    - Grades keywords from "Very Easy" to "Very Hard"
    - Exports a prioritised Excel workbook

    **How to use:**
    1. Enter your DataForSEO credentials in the sidebar (or set environment variables)
    2. Upload a CSV containing your keyword list
    3. Select the keyword column, location, and any filters
    4. Review the estimated cost, then click **Run Analysis**
    5. Download the results as Excel

    **Difficulty grades:**
    - 0 to 14: Very Easy
    - 15 to 29: Easy
    - 30 to 49: Possible
    - 50 to 69: Difficult
    - 70 to 84: Hard
    - 85 to 100: Very Hard
    """
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCATION_CODES = {
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
    "Ireland": 2372,
}

BATCH_SIZE = 1000
COST_PER_KEYWORD = 0.002  # $0.001 for KD + $0.001 for intent

DIFFICULTY_GRADES = [
    (14, "Very Easy"),
    (29, "Easy"),
    (49, "Possible"),
    (69, "Difficult"),
    (84, "Hard"),
    (100, "Very Hard"),
]

# ---------------------------------------------------------------------------
# Sidebar: credentials and settings
# ---------------------------------------------------------------------------

st.sidebar.header("DataForSEO Credentials")

dataforseo_login = st.sidebar.text_input(
    "DataForSEO Login",
    value=os.environ.get("DATAFORSEO_LOGIN", ""),
    type="password",
    help="Your DataForSEO API login. Can also be set via the DATAFORSEO_LOGIN environment variable.",
)

dataforseo_password = st.sidebar.text_input(
    "DataForSEO Password",
    value=os.environ.get("DATAFORSEO_PASSWORD", ""),
    type="password",
    help="Your DataForSEO API password. Can also be set via the DATAFORSEO_PASSWORD environment variable.",
)

st.sidebar.header("Settings")

location_select = st.sidebar.selectbox(
    "Select the search location",
    list(LOCATION_CODES.keys()),
)

max_diff = st.sidebar.slider(
    "Maximum keyword difficulty to include",
    value=100,
    min_value=0,
    max_value=100,
    help="Keywords with a difficulty score above this value will be excluded from the results.",
)

filter_questions = st.sidebar.checkbox("Select only question keywords?", value=False)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload your keyword CSV")

if uploaded_file is None:
    st.info("Upload a .csv or .txt file to get started.")
    st.stop()

try:
    result = chardet.detect(uploaded_file.getvalue())
    encoding_value = result["encoding"]

    if encoding_value == "UTF-16":
        white_space = True
    else:
        white_space = False

    df_upload = pd.read_csv(
        uploaded_file,
        encoding=encoding_value,
        delim_whitespace=white_space,
        on_bad_lines="skip",
    )
    number_of_rows = len(df_upload)

    if number_of_rows == 0:
        st.caption("Your file appears to be empty.")
        st.stop()

    with st.expander("View raw data", expanded=False):
        st.dataframe(df_upload)

except (UnicodeDecodeError, pd.errors.ParserError):
    st.warning("The file could not be loaded. Please check the file type and format.")
    st.stop()

# ---------------------------------------------------------------------------
# Column selection form
# ---------------------------------------------------------------------------

with st.form(key="column_select_form"):
    st.subheader("Select the Keyword Column")
    kw_col = st.selectbox("Keyword column:", df_upload.columns)
    submitted = st.form_submit_button("Submit")

if not submitted:
    st.stop()

# ---------------------------------------------------------------------------
# Prepare keyword list
# ---------------------------------------------------------------------------

df_comp = df_upload.copy()
df_comp.rename(columns={kw_col: "Keyword"}, inplace=True)

if filter_questions:
    q_words = r"who |what |where |why |when |how |is |are |does |do |can "
    df_comp = df_comp[df_comp["Keyword"].str.contains(q_words, case=False, na=False)]

if len(df_comp) == 0:
    st.warning("No keywords remain after applying filters. Please adjust your settings.")
    st.stop()

# Deduplicate keywords for the API calls (preserve all rows for the final merge)
keywords = df_comp["Keyword"].dropna().unique().tolist()
num_keywords = len(keywords)
num_batches = math.ceil(num_keywords / BATCH_SIZE)

with st.expander("View keywords to process", expanded=False):
    st.write(keywords)

# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

estimated_cost = num_keywords * COST_PER_KEYWORD

st.info(
    f"**Keywords to process:** {num_keywords}  |  "
    f"**API batches:** {num_batches}  |  "
    f"**Estimated cost:** ${estimated_cost:.2f} "
    f"(${COST_PER_KEYWORD:.3f} per keyword for difficulty + intent)"
)

# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

if not dataforseo_login or not dataforseo_password:
    st.error(
        "Please enter your DataForSEO login and password in the sidebar, "
        "or set the DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

run_analysis = st.button("Run Analysis", type="primary")

if not run_analysis:
    st.stop()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_headers(login: str, password: str) -> dict:
    """Build HTTP headers with Basic auth for DataForSEO."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {cred}",
        "Content-Type": "application/json",
    }


def _grade_difficulty(score) -> str:
    """Return a human-readable difficulty grade."""
    if pd.isna(score):
        return ""
    score = int(score)
    for threshold, label in DIFFICULTY_GRADES:
        if score <= threshold:
            return label
    return "Very Hard"


def _batch_list(items: list, batch_size: int) -> list[list]:
    """Split a list into batches of the given size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def fetch_bulk_keyword_difficulty(
    keywords_batch: list[str],
    location_code: int,
    headers: dict,
) -> list[dict]:
    """Call the DataForSEO Bulk Keyword Difficulty endpoint for one batch."""
    url = "https://api.dataforseo.com/v3/dataforseo_labs/google/bulk_keyword_difficulty/live"
    payload = [
        {
            "keywords": keywords_batch,
            "location_code": location_code,
            "language_code": "en",
        }
    ]
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()

    # Validate response structure
    if data.get("status_code") != 20000:
        raise RuntimeError(
            f"DataForSEO API error: {data.get('status_message', 'Unknown error')}"
        )

    tasks = data.get("tasks", [])
    if not tasks or tasks[0].get("status_code") != 20000:
        msg = tasks[0].get("status_message", "Unknown task error") if tasks else "No tasks returned"
        raise RuntimeError(f"DataForSEO task error: {msg}")

    return tasks[0].get("result", []) or []


def fetch_search_intent(
    keywords_batch: list[str],
    location_code: int,
    headers: dict,
) -> list[dict]:
    """Call the DataForSEO Search Intent endpoint for one batch."""
    url = "https://api.dataforseo.com/v3/dataforseo_labs/google/search_intent/live"
    payload = [
        {
            "keywords": keywords_batch,
            "location_code": location_code,
            "language_code": "en",
        }
    ]
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()

    if data.get("status_code") != 20000:
        raise RuntimeError(
            f"DataForSEO API error: {data.get('status_message', 'Unknown error')}"
        )

    tasks = data.get("tasks", [])
    if not tasks or tasks[0].get("status_code") != 20000:
        msg = tasks[0].get("status_message", "Unknown task error") if tasks else "No tasks returned"
        raise RuntimeError(f"DataForSEO task error: {msg}")

    return tasks[0].get("result", []) or []


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

headers = _build_headers(dataforseo_login, dataforseo_password)
location_code = LOCATION_CODES[location_select]
batches = _batch_list(keywords, BATCH_SIZE)

# Collect results
kd_records = []  # list of {"keyword": ..., "keyword_difficulty": ...}
intent_records = []  # list of {"keyword": ..., "keyword_intent": ..., "secondary_keyword_intents": ...}

total_steps = num_batches * 2  # KD + intent for each batch

try:
    with stqdm(total=total_steps, desc="Fetching data from DataForSEO") as pbar:
        for batch in batches:
            # Keyword difficulty
            pbar.set_description("Fetching keyword difficulty")
            kd_result = fetch_bulk_keyword_difficulty(batch, location_code, headers)
            kd_records.extend(kd_result)
            pbar.update(1)

            # Search intent
            pbar.set_description("Fetching search intent")
            intent_result = fetch_search_intent(batch, location_code, headers)
            intent_records.extend(intent_result)
            pbar.update(1)

except (requests.RequestException, RuntimeError) as exc:
    st.error(f"API request failed: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Build results dataframe
# ---------------------------------------------------------------------------

# Keyword difficulty
df_kd = pd.DataFrame(kd_records)
if "keyword" in df_kd.columns and "keyword_difficulty" in df_kd.columns:
    df_kd = df_kd[["keyword", "keyword_difficulty"]].copy()
    df_kd.rename(
        columns={"keyword": "Keyword", "keyword_difficulty": "Keyword Difficulty"},
        inplace=True,
    )
else:
    st.warning("No keyword difficulty data was returned. Please check your credentials and try again.")
    st.stop()

# Search intent
if intent_records:
    intent_rows = []
    for item in intent_records:
        kw = item.get("keyword", "")
        primary_intent = ""
        secondary_intents = ""

        ki = item.get("keyword_intent", {})
        if ki and isinstance(ki, dict):
            primary_intent = ki.get("label", "")

        sec = item.get("secondary_keyword_intents")
        if sec and isinstance(sec, list):
            secondary_intents = ", ".join(
                s.get("label", "") for s in sec if isinstance(s, dict) and s.get("label")
            )

        intent_rows.append(
            {
                "Keyword": kw,
                "Search Intent": primary_intent,
                "Secondary Intents": secondary_intents,
            }
        )

    df_intent = pd.DataFrame(intent_rows)
else:
    df_intent = pd.DataFrame(columns=["Keyword", "Search Intent", "Secondary Intents"])

# Merge KD and intent on keyword
df_api = pd.merge(df_kd, df_intent, on="Keyword", how="outer")

# Merge API results back into the original dataframe
df_comp = pd.merge(df_comp, df_api, on="Keyword", how="left")

# Add difficulty grade
df_comp["Difficulty Grade"] = df_comp["Keyword Difficulty"].apply(_grade_difficulty)

# Apply max difficulty filter
df_comp["Keyword Difficulty"] = pd.to_numeric(df_comp["Keyword Difficulty"], errors="coerce")
df_comp = df_comp[df_comp["Keyword Difficulty"].isna() | (df_comp["Keyword Difficulty"] <= max_diff)]

# Reorder columns: put the key columns first, then any extras from the upload
priority_cols = [
    "Keyword",
    "Keyword Difficulty",
    "Difficulty Grade",
    "Search Intent",
    "Secondary Intents",
]
remaining_cols = [c for c in df_comp.columns if c not in priority_cols]
df_comp = df_comp[priority_cols + remaining_cols]

# Sort by difficulty ascending
df_comp = df_comp.sort_values(by="Keyword Difficulty", ascending=True, na_position="last")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

st.subheader("Results")
st.dataframe(df_comp, use_container_width=True)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Keywords", len(df_comp))
with col2:
    avg_diff = df_comp["Keyword Difficulty"].mean()
    st.metric("Average Difficulty", f"{avg_diff:.1f}" if not pd.isna(avg_diff) else "N/A")
with col3:
    easy_count = len(df_comp[df_comp["Keyword Difficulty"] <= 29])
    st.metric("Easy or Below", easy_count)
with col4:
    hard_count = len(df_comp[df_comp["Keyword Difficulty"] >= 70])
    st.metric("Hard or Above", hard_count)

# ---------------------------------------------------------------------------
# Question keywords sheet
# ---------------------------------------------------------------------------

q_words_filter = r"who |what |where |why |when |how |is |are |does |do |can "
df_questions = df_comp[df_comp["Keyword"].str.contains(q_words_filter, case=False, na=False)]

# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------


@st.cache_data
def build_excel(df_main: pd.DataFrame, df_q: pd.DataFrame) -> bytes:
    """Write the main and questions dataframes to an Excel workbook."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_main.to_excel(writer, sheet_name="Keyword Analysis", index=False)
        df_q.to_excel(writer, sheet_name="Questions Only", index=False)
    return output.getvalue()


excel_data = build_excel(df_comp, df_questions)

st.download_button(
    label="Download Results as Excel",
    data=excel_data,
    file_name="keyword_difficulty_analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
