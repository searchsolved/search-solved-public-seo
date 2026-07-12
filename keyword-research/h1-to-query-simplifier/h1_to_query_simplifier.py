# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  H1 to Query Simplifier                                                          #
#                                                                                  #
#  Convert marketing-heavy H1 headings into clean, natural search queries          #
#  using the Anthropic API.                                                        #
#                                                                                  #
####################################################################################
# Author: Lee Foot                                                              #
# Website  : https://www.leefoot.com                                                   #
# Contact  : https://www.leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
H1 to Query Simplifier

Converts marketing-heavy H1 headings into clean, natural search queries
using the Anthropic API. Useful as a pre-processing step before SERP
clustering or keyword matching, where promotional H1s make poor queries.

Features:
- Upload a CSV and pick the H1 column
- Claude-powered query simplification
- Resume support: re-upload a partially processed file and completed rows are skipped
- Partial results kept if a run fails part-way through
- Export results to CSV
"""

import time

import anthropic
import pandas as pd
import streamlit as st

DEFAULT_MODEL = "claude-haiku-4-5"
OUTPUT_COLUMN = "H1_Simplified"

PROMPT_TEMPLATE = """Convert this H1 into a natural Google search query. Remove marketing language, step counts, and promotional words. Focus on the core search intent.

H1: "{h1_text}"

Search query:"""

st.set_page_config(page_title="H1 to Query Simplifier", page_icon="🔍", layout="wide")

st.title("H1 to Query Simplifier")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown(f"""
    **What this tool does:**
    - Converts marketing-heavy H1s (e.g. "10 Proven Ways to Boost Your Rankings Fast!") into
      clean, natural search queries (e.g. "how to improve search rankings")
    - Useful as a pre-processing step before SERP clustering or keyword matching

    **Requirements:**
    - Anthropic API key
    - CSV with a column of H1 headings (a Screaming Frog export with an `H1-1` column works out of the box)

    **How to use:**
    1. Enter your Anthropic API key in the sidebar
    2. Upload your CSV and pick the H1 column
    3. Click "Simplify H1s"
    4. Download the results

    **Resuming:** results are written to an `{OUTPUT_COLUMN}` column. If you re-upload a
    partially processed file, rows that already have a value are skipped, so you can
    resume an interrupted run without re-paying for completed rows.

    **Note:** each H1 uses one API call.
    """)

# Sidebar settings
st.sidebar.header("API Settings")

api_key = st.sidebar.text_input(
    "Anthropic API Key",
    type="password",
    help="Your Anthropic API key. Used only for this session and never stored.",
)

model = st.sidebar.text_input(
    "Model",
    value=DEFAULT_MODEL,
    help="Anthropic model to use. Haiku is fast and cheap, which suits this task well.",
)

st.sidebar.markdown("---")
st.sidebar.header("Processing Settings")

delay = st.sidebar.slider(
    "Delay between requests (seconds)",
    min_value=0.0,
    max_value=2.0,
    value=0.1,
    step=0.1,
    help="Small delay between API calls to respect rate limits",
)


def simplify_h1(client, h1_text, model_name):
    """Simplify a single H1 into a natural search query. Returns (result, error)."""
    try:
        message = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(h1_text=h1_text)}
            ],
        )
        simplified = message.content[0].text.strip()
        # Remove any remaining quotation marks or extra formatting
        simplified = simplified.replace('"', "").replace("'", "").strip()
        return simplified, None

    except anthropic.RateLimitError:
        return None, "Rate limited. Increase the delay in the sidebar and try again."
    except anthropic.AuthenticationError:
        return None, "Invalid API key. Check your Anthropic API key in the sidebar."
    except anthropic.NotFoundError:
        return None, f"Model '{model_name}' not found. Check the model name in the sidebar."
    except anthropic.APIStatusError as e:
        return None, f"API error ({e.status_code}): {e.message}"
    except anthropic.APIConnectionError as e:
        return None, f"Connection error: {e}"
    except Exception as e:
        return None, str(e)


# File upload
st.subheader("Upload H1s")

uploaded_file = st.file_uploader(
    "Upload CSV with H1 headings",
    type=["csv"],
    help="Any CSV with a column of H1 text, e.g. a Screaming Frog export",
)

df = None
h1_column = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")

        cols = df.columns.tolist()
        h1_options = [c for c in cols if "h1" in c.lower()]
        default_idx = cols.index(h1_options[0]) if h1_options else 0
        h1_column = st.selectbox(
            "H1 column",
            cols,
            index=default_idx,
            help="The column containing the H1 headings to simplify",
        )

        if OUTPUT_COLUMN in df.columns:
            already_done = (
                df[OUTPUT_COLUMN].notna() & (df[OUTPUT_COLUMN].astype(str).str.strip() != "")
            ).sum()
            if already_done:
                st.info(
                    f"Found an existing '{OUTPUT_COLUMN}' column with {already_done} completed "
                    f"rows. These will be skipped, so you are resuming a previous run."
                )

        with st.expander("Preview H1s"):
            st.write(df[h1_column].dropna().astype(str).head(20).tolist())

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None

if st.button("Simplify H1s", type="primary", disabled=not api_key or df is None):
    if not api_key:
        st.error("Please enter your Anthropic API key")
    elif df is None or h1_column is None:
        st.error("Please upload a CSV and select the H1 column")
    else:
        work_df = df.copy()
        if OUTPUT_COLUMN not in work_df.columns:
            work_df[OUTPUT_COLUMN] = ""

        # Work out which rows still need processing (supports resume)
        items_to_process = []
        for idx in range(len(work_df)):
            h1_text = work_df.iloc[idx][h1_column]
            if (
                pd.notna(work_df.iloc[idx][OUTPUT_COLUMN])
                and str(work_df.iloc[idx][OUTPUT_COLUMN]).strip() != ""
            ):
                continue
            if pd.isna(h1_text) or str(h1_text).strip() == "":
                continue
            items_to_process.append(idx)

        if not items_to_process:
            st.session_state["results"] = work_df
            st.success("All rows are already processed!")
        else:
            client = anthropic.Anthropic(api_key=api_key)
            output_col_loc = work_df.columns.get_loc(OUTPUT_COLUMN)

            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_count = 0
            failed_count = 0
            stopped_early = False

            for i, idx in enumerate(items_to_process):
                h1_text = str(work_df.iloc[idx][h1_column])
                status_text.text(f"Processing {i + 1}/{len(items_to_process)}: {h1_text[:80]}")

                simplified, error = simplify_h1(client, h1_text, model)

                if simplified:
                    work_df.iloc[idx, output_col_loc] = simplified
                    processed_count += 1
                elif error and ("API key" in error or "Rate limited" in error or "not found" in error):
                    # Fatal errors: stop and keep partial results
                    st.error(error)
                    stopped_early = True
                    break
                else:
                    failed_count += 1

                # Keep partial results so progress is not lost if the run fails
                if (i + 1) % 10 == 0:
                    st.session_state["results"] = work_df

                progress_bar.progress((i + 1) / len(items_to_process))
                time.sleep(delay)

            st.session_state["results"] = work_df
            status_text.empty()

            if stopped_early:
                st.warning(
                    f"Stopped after {processed_count} rows. Partial results are available "
                    f"below. Download them and re-upload the file later to resume."
                )
            else:
                st.success(f"Simplified {processed_count} H1s!")
                if failed_count:
                    st.warning(f"{failed_count} rows failed and were left blank. Re-run to retry them.")

# Display results
if "results" in st.session_state:
    df_results = st.session_state["results"]

    done_mask = df_results[OUTPUT_COLUMN].notna() & (
        df_results[OUTPUT_COLUMN].astype(str).str.strip() != ""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", len(df_results))
    with col2:
        st.metric("Simplified", int(done_mask.sum()))

    st.subheader("Results")
    st.dataframe(df_results, use_container_width=True)

    st.subheader("Download")
    csv_data = df_results.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="h1s_simplified.csv",
        mime="text/csv",
    )

else:
    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to get started")

    st.subheader("Example Output")

    example_data = {
        "H1-1": [
            "10 Proven Ways to Boost Your Rankings Fast!",
            "The Ultimate Guide to Hiring a Plumber (2024 Edition)",
            "Why Thousands Trust Us for Garden Maintenance",
        ],
        OUTPUT_COLUMN: [
            "how to improve search rankings",
            "how to hire a plumber",
            "garden maintenance services",
        ],
    }
    st.dataframe(pd.DataFrame(example_data))

    st.markdown("""
    **Use Cases:**
    - Turn H1 exports into natural queries before SERP clustering
    - Build keyword-to-page matching datasets from crawl exports
    - Normalise promotional headings for search intent analysis
    """)
