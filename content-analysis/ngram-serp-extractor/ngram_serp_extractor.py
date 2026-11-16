# Author: Lee Foot
# Website: https://leefoot.com

import os
from collections import Counter
from itertools import chain

import nltk
import pandas as pd
import streamlit as st
import trafilatura
from trafilatura import bare_extraction
from trafilatura.settings import use_config

from dataforseo_queue import (
    QueueError,
    collect_results,
    estimate_cost,
    is_unresolved,
    job_from_csv,
    job_to_csv,
    organic_items,
    send_searches,
)

st.set_page_config(page_title="SERP N-gram & Title Extractor", page_icon="📈",
                   layout="wide")

# Download NLTK stopwords data (required for Streamlit Cloud)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)

download_nltk_data()
stop = nltk.corpus.stopwords.words('english')

newconfig = use_config()
newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

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
}

# start strip out all special characters from a column
spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")", "*", "+", ",", ".", "/", ":", ";", "<", "=", ">", "?", "@",
              "[", "\\", "]", "^", "-", "_", "`", "{", "|", "}", "~", "\u2013"]


def serp_links(items):
    """Organic result URLs to fetch, and how many came back as unresolved google.com/goto links."""
    links, unresolved = [], 0
    for result in organic_items(items):
        url = result.get('url')
        if is_unresolved(url):
            unresolved += 1
        elif url:
            links.append(url)
    return links, unresolved


def extract_pages(links):
    """Fetch each ranking page and extract its content and title."""
    # store the extracted data
    text = []
    title = []
    extracted_url = []

    # loop through page 1 urls extracting content and h1s
    for url in links:
        try:
            downloaded = trafilatura.fetch_url(url)
            # trafilatura 2.x returns a Document object, not a dict
            d = bare_extraction(downloaded, config=newconfig, with_metadata=True)
            if d is None or not d.text:
                continue
            text.append(d.text)
            title.append(d.title or '')
            extracted_url.append(d.url or url)
        except Exception:
            pass
    return extracted_url, text, title


def analyse(kw, extracted_url, text, title):
    """Content bigrams, title keywords and extracted titles for one keyword."""
    # make serp extraction df
    df_serp = pd.DataFrame(None)
    df_serp['url'] = extracted_url
    df_serp['text'] = text
    df_serp['title'] = title
    df_serp['query'] = kw

    # get most frequent words from all ages tiles combined]
    df_title = df_serp.groupby(['query'])['title'].apply(' '.join).reset_index()

    for char in spec_chars:
        df_title['title'] = df_title['title'].str.replace(char, ' ', regex=False)

    # get word counts of page title
    word_str = df_title['title']
    word_str = [words for segments in word_str for words in segments.split()]
    counts = pd.Series(word_str).value_counts()
    counts = counts.to_dict()

    # make the dataframe with the most popular page titles
    df_title = pd.DataFrame(counts.items(), columns=['keyword', 'frequency'])
    df_title = df_title[df_title.frequency > 1]
    df_title['query'] = kw

    # create n-grams dataframe
    df_serp['text'] = df_serp['text'].str.lower()

    for char in spec_chars:
        df_serp['text'] = df_serp['text'].str.replace(char, ' ', regex=False)

    # remove stop words
    df_serp['text'] = df_serp['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    my_l = df_serp['text']
    my_l = [words for segments in my_l for words in segments.split()]
    my_str = " ".join(my_l)

    df_ngrams = pd.DataFrame({'text': [my_str]})

    def find_ngrams(input_list, n):
        return list(zip(*[input_list[i:] for i in range(n)]))

    df_ngrams['bigrams'] = df_ngrams['text'].map(lambda x: find_ngrams(x.split(" "), 2))

    # store the data
    bigram_list = []

    bigrams = df_ngrams['bigrams'].tolist()

    for bi in bigrams:
        bigrams = list(chain(*bigrams))
        bigrams = [(x.lower(), y.lower()) for x, y in bi]
        bigram_counts = Counter(bigrams)
        common = bigram_counts.most_common(10)
        bigram_list.append(common)

    # # make the ngram dataframe
    df_ngrams = pd.DataFrame(None)
    df_ngrams['bigrams'] = bigram_list
    df_ngrams = df_ngrams.explode("bigrams")
    df_ngrams['bigrams'] = df_ngrams['bigrams'].astype(str)

    df_serp_display = df_serp[["title", "url"]]
    df_title_display = df_title[["keyword", "frequency"]]
    return df_ngrams, df_title_display, df_serp_display


st.title("SERP N-gram & Title Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Finds the ten most common word pairs (bigrams) across the pages ranking for a keyword
    - Lists the words used more than once in their titles
    - Gives you the titles and URLs of the ranking pages

    **Why two steps:** Google now routes result links through google.com/goto. DataForSEO's
    Standard queue turns those back into real URLs, but a search can take 25 minutes or
    more to finish. So you send the search, keep the job file, and collect the results later.

    **Requirements:** a DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar
    2. **Send search:** enter a seed keyword, choose the region, device and number of pages,
       check the estimated cost and click "Send search". Download the job file.
    3. **Collect results** (about 30 minutes later): upload the job file and click "Collect results".
       The tool extracts page titles and content from the ranking pages
    4. Use this data to optimise your page titles and expand content coverage

    **Best for:**
    - SERP content analysis
    - Identifying ranking content patterns
    - Competitive phrase research
    """)


dataforseo_login = st.sidebar.text_input(
    'DataForSEO Login',
    type='password',
    value=os.environ.get('DATAFORSEO_LOGIN', ''),
    help="Your login from dataforseo.com"
)

dataforseo_password = st.sidebar.text_input(
    'DataForSEO Password',
    type='password',
    value=os.environ.get('DATAFORSEO_PASSWORD', ''),
    help="Your password from dataforseo.com"
)

has_credentials = bool(dataforseo_login and dataforseo_password)

send_tab, collect_tab = st.tabs(["1. Send search", "2. Collect results"])

with send_tab:
    kw = st.text_input('Input Your Search Keyword')

    col1, col2, col3 = st.columns(3)
    with col1:
        location_select = st.selectbox(
            "Select The Region To Search Google From",
            list(LOCATION_CODES.keys()),
        )
    with col2:
        device_select = st.selectbox(
            "Select The Host Device To Use To Search Google",
            (
                "Desktop",
                "Mobile",
            ),
            help="DataForSEO returns desktop or mobile results only.",
        )
    with col3:
        num_pages = st.slider("Set Number of Pages to Analyse", min_value=1, max_value=10, value=1)
    depth = num_pages * 10

    # Show estimated cost
    st.info(f"Estimated cost per search: ${estimate_cost(1, depth):.4f}")

    if st.button("Send search", type="primary"):
        if not has_credentials:
            st.error("Please enter your DataForSEO login and password in the sidebar")
        elif not kw.strip():
            st.error("Please enter a search keyword")
        else:
            try:
                with st.spinner('Sending search to DataForSEO...'):
                    job_rows, send_warnings, send_cost = send_searches(
                        [kw.strip()], dataforseo_login, dataforseo_password,
                        LOCATION_CODES[location_select], "en",
                        device=device_select.lower(), depth=depth)
            except QueueError as e:
                st.error(str(e))
            else:
                for warning in send_warnings:
                    st.warning(warning)
                if job_rows:
                    st.session_state['ngram_job'] = job_rows
                    st.session_state['ngram_send_cost'] = send_cost

    if st.session_state.get('ngram_job'):
        job_rows = st.session_state['ngram_job']
        st.success(f"Search sent for \"{job_rows[0]['keyword']}\". Cost reported by DataForSEO: "
                   f"${st.session_state['ngram_send_cost']:.4f}")
        st.markdown("DataForSEO can take **25 minutes or more** to finish. Download the job file, "
                    "then use **2. Collect results**. While this page stays open, the job is kept "
                    "here too, so you can collect without uploading it.")
        st.download_button("Download job file", job_to_csv(job_rows).encode('utf-8'),
                           "ngram_serp_job.csv", "text/csv")

with collect_tab:
    job_rows = st.session_state.get('ngram_job')
    job_file = st.file_uploader("Job file from step 1", type=['csv'],
                                help="The ngram_serp_job.csv you downloaded when you sent the search")
    if job_file is not None:
        try:
            job_rows = job_from_csv(job_file.getvalue().decode('utf-8', errors='replace'))
        except ValueError as e:
            st.error(str(e))
            job_rows = None
    elif job_rows:
        st.caption("Using the search sent from this page.")

    if st.button("Collect results", type="primary"):
        if not has_credentials:
            st.error("Please enter your DataForSEO login and password in the sidebar")
        elif not job_rows:
            st.error("Upload the job file from step 1, or send a search first")
        else:
            try:
                with st.spinner('Collecting SERP data...'):
                    result = collect_results(job_rows, dataforseo_login, dataforseo_password)
            except QueueError as e:
                st.error(str(e))
            else:
                analyses = {}
                unresolved_total = 0
                with st.spinner('Extracting content from pages...'):
                    for query, items in result['ready'].items():
                        links, unresolved = serp_links(items)
                        unresolved_total += unresolved
                        if not links:
                            analyses[query] = None
                            continue
                        extracted_url, text, title = extract_pages(links)
                        if not text:
                            analyses[query] = None
                            continue
                        analyses[query] = analyse(query, extracted_url, text, title)
                st.session_state['ngram_result'] = {
                    'analyses': analyses,
                    'pending': result['pending'],
                    'errors': result['errors'],
                    'unresolved': unresolved_total,
                    'total': len(job_rows),
                }

    report = st.session_state.get('ngram_result')
    if report:
        if report['pending']:
            st.warning(f"{len(report['analyses'])} of {report['total']} search(es) have finished; "
                       f"{len(report['pending'])} still running at DataForSEO: {', '.join(report['pending'])}. "
                       "Try again in a few minutes.")
        if report['errors']:
            with st.expander(f"{len(report['errors'])} search(es) failed"):
                for error in report['errors']:
                    st.text(error)
        if report['unresolved']:
            st.warning(f"{report['unresolved']} result(s) came back as unresolved google.com/goto links "
                       "and were skipped.")

        analyses = report['analyses']
        if analyses:
            query = list(analyses)[0]
            if len(analyses) > 1:
                query = st.selectbox("Keyword", list(analyses))

            if analyses[query] is None:
                st.error(f"No pages could be extracted for \"{query}\". The organic results were "
                         "missing, unresolved or could not be fetched.")
            else:
                df_ngrams, df_title_display, df_serp_display = analyses[query]

                st.success("Analysis complete!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Content Bigrams")
                    st.write("Most common word pairs from page content")
                    st.dataframe(df_ngrams)

                with col2:
                    st.subheader("Title Keywords")
                    st.write("Words appearing in multiple titles")
                    st.dataframe(df_title_display)

                with col3:
                    st.subheader("Extracted Titles")
                    st.write("Page titles from ranking URLs")
                    st.dataframe(df_serp_display)

                # Download buttons
                st.divider()
                col1, col2, col3 = st.columns(3)

                with col1:
                    csv_ngrams = df_ngrams.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Bigrams", csv_ngrams, "bigrams.csv", "text/csv")

                with col2:
                    csv_titles = df_title_display.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Title Keywords", csv_titles, "title_keywords.csv", "text/csv")

                with col3:
                    csv_serp = df_serp_display.to_csv(index=False).encode('utf-8')
                    st.download_button("Download SERP Titles", csv_serp, "serp_titles.csv", "text/csv")
