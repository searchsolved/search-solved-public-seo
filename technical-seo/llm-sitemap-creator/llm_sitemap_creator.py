####################################################################################
#                                                                                  #
#  LLM Sitemap Creator                                                             #
#                                                                                  #
#  Use GPT to generate hierarchical sitemap structures from keywords.              #
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
LLM Sitemap Creator

Uses OpenAI GPT to generate hierarchical sitemap structures from keywords.
Takes keyword list with search volumes and creates an organized site structure.
Validates output includes all input keywords.

Features:
- Upload keyword CSV with search volumes
- Configurable max categories and depth
- AI-generated sitemap structure
- Validates all keywords are included
- Tree visualization and export
"""

import streamlit as st
import pandas as pd
import json
from io import BytesIO
from openai import OpenAI

st.set_page_config(page_title="LLM Sitemap Creator", page_icon="🗺️", layout="wide")

st.title("LLM Sitemap Creator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Uses AI (GPT) to organize keywords into a logical sitemap structure
    - Creates hierarchical categories based on keyword relationships
    - Considers search volume when determining hierarchy

    **Requirements:**
    - OpenAI API key
    - CSV with keywords and search volumes

    **How to use:**
    1. Enter your OpenAI API key in the sidebar
    2. Upload a CSV with 'keyword' and 'volume' columns
    3. Configure max categories and depth
    4. Click "Generate Sitemap"
    5. Review and export the structure

    **Tip:** Higher volume keywords are typically placed higher in the hierarchy.
    """)

# Sidebar settings
st.sidebar.header("API Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your OpenAI API key"
)

st.sidebar.markdown("---")
st.sidebar.header("Sitemap Settings")

max_categories = st.sidebar.slider(
    "Max top-level categories",
    min_value=3,
    max_value=15,
    value=8,
    help="Maximum number of main categories"
)

max_depth = st.sidebar.slider(
    "Max depth",
    min_value=2,
    max_value=5,
    value=3,
    help="Maximum nesting depth"
)

model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
    index=0,
    help="GPT model to use (mini is cheaper)"
)


def flatten_sitemap(sitemap):
    """Extract all keywords from sitemap structure."""
    keywords = []
    for key, value in sitemap.items():
        keywords.append(key)
        if isinstance(value, dict):
            keywords.extend(flatten_sitemap(value))
    return keywords


def generate_sitemap_with_llm(keyword_volumes, api_key, max_categories, max_depth, model):
    """Generate sitemap structure using OpenAI."""
    client = OpenAI(api_key=api_key)

    keywords_str = json.dumps(keyword_volumes, indent=2)

    messages = [
        {
            "role": "system",
            "content": "You are an SEO expert assistant. Create a sitemap structure using the provided keywords and their search volumes. Return ONLY valid JSON."
        },
        {
            "role": "user",
            "content": f"""Given the following keywords and their search volumes:

{keywords_str}

Create a sitemap structure for an SEO strategy. Requirements:

1. Create a maximum of {max_categories} top-level categories.
2. The sitemap should have a maximum depth of {max_depth} levels.
3. Group related keywords together under appropriate categories.
4. Higher volume keywords should generally be higher in the structure.
5. IMPORTANT: Every single provided keyword MUST be included in the sitemap.

Return the sitemap as JSON where each key is a keyword and its value is either:
- An empty object {{}} for leaf nodes
- Another object for nodes with children

Example format:
{{
  "main category 1": {{
    "subcategory a": {{
      "leaf keyword 1": {{}},
      "leaf keyword 2": {{}}
    }},
    "leaf keyword 3": {{}}
  }},
  "main category 2": {{}}
}}

Return ONLY the JSON, no explanations."""
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000
        )

        response_content = completion.choices[0].message.content
        sitemap = json.loads(response_content)

        return sitemap, None

    except Exception as e:
        return None, str(e)


def print_sitemap(sitemap, keyword_to_volume, level=0, max_depth=3):
    """Generate string representation of sitemap tree."""
    output = []

    if level == 0:
        output.append("Home")

    for i, (key, value) in enumerate(sitemap.items()):
        prefix = "├── " if i < len(sitemap) - 1 else "└── "
        volume = keyword_to_volume.get(key, "N/A")
        volume_str = f" ({volume:,})" if isinstance(volume, (int, float)) else ""
        output.append(f"{'    ' * level}{prefix}{key}{volume_str}")

        if isinstance(value, dict) and level < max_depth - 1:
            output.extend(print_sitemap(value, keyword_to_volume, level + 1, max_depth))

    return output


def sitemap_to_df(sitemap, parent="", level=0, rows=None):
    """Convert sitemap to DataFrame rows."""
    if rows is None:
        rows = []

    for key, value in sitemap.items():
        rows.append({
            'Level': level,
            'Parent': parent,
            'Keyword': key
        })
        if isinstance(value, dict):
            sitemap_to_df(value, key, level + 1, rows)

    return rows


# Main content
st.subheader("Upload Keywords")

uploaded_file = st.file_uploader(
    "Upload CSV with keywords and volumes",
    type=['csv'],
    help="CSV with 'keyword' and 'volume' columns"
)

keywords_data = {}

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} keywords")

        with st.expander("Column Mapping"):
            cols = df.columns.tolist()

            keyword_col = st.selectbox(
                "Keyword column",
                cols,
                index=cols.index('keyword') if 'keyword' in [c.lower() for c in cols] else 0
            )
            volume_col = st.selectbox(
                "Volume column",
                cols,
                index=cols.index('volume') if 'volume' in [c.lower() for c in cols] else (1 if len(cols) > 1 else 0)
            )

        with st.expander("Preview data"):
            st.dataframe(df.head(20))

        # Build keyword dict
        for _, row in df.iterrows():
            kw = str(row[keyword_col]).strip()
            vol = row[volume_col]
            if kw and pd.notna(vol):
                keywords_data[kw] = int(vol)

        st.info(f"Prepared {len(keywords_data)} keywords for sitemap generation")

    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")

# Manual input alternative
with st.expander("Or enter keywords manually"):
    manual_input = st.text_area(
        "Enter keywords (one per line: keyword,volume)",
        height=150,
        placeholder="seo tools,12000\nkeyword research,8000\ncontent marketing,5000"
    )

    if manual_input and not uploaded_file:
        for line in manual_input.strip().split('\n'):
            if ',' in line:
                parts = line.split(',')
                kw = parts[0].strip()
                try:
                    vol = int(parts[1].strip())
                    keywords_data[kw] = vol
                except:
                    pass

        if keywords_data:
            st.info(f"Parsed {len(keywords_data)} keywords from manual input")

if st.button("Generate Sitemap", type="primary", disabled=not api_key or not keywords_data):
    if not api_key:
        st.error("Please enter your OpenAI API key")
    elif not keywords_data:
        st.error("Please upload keywords or enter them manually")
    else:
        with st.spinner(f"Generating sitemap with {model}..."):
            sitemap, error = generate_sitemap_with_llm(
                keywords_data,
                api_key,
                max_categories,
                max_depth,
                model
            )

            if error:
                st.error(f"Error: {error}")
            elif sitemap:
                # Validate all keywords included
                generated_kws = set(kw.lower() for kw in flatten_sitemap(sitemap))
                input_kws = set(kw.lower() for kw in keywords_data.keys())

                missing = input_kws - generated_kws
                extra = generated_kws - input_kws

                if missing:
                    st.warning(f"Missing keywords: {', '.join(list(missing)[:10])}...")
                if extra:
                    st.info(f"New category keywords created: {', '.join(list(extra)[:10])}")

                # Store results
                st.session_state['sitemap'] = sitemap
                st.session_state['keyword_volumes'] = keywords_data

                st.success("Sitemap generated successfully!")

# Display results
if 'sitemap' in st.session_state:
    sitemap = st.session_state['sitemap']
    keyword_volumes = st.session_state['keyword_volumes']

    # Metrics
    all_kws = flatten_sitemap(sitemap)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Keywords", len(all_kws))
    with col2:
        top_level = len(sitemap)
        st.metric("Top-Level Categories", top_level)
    with col3:
        total_vol = sum(keyword_volumes.get(kw, 0) for kw in all_kws)
        st.metric("Total Volume", f"{total_vol:,}")

    # Tree visualization
    st.subheader("Sitemap Structure")
    tree_lines = print_sitemap(sitemap, keyword_volumes, max_depth=max_depth)
    st.code('\n'.join(tree_lines), language=None)

    # Table view
    st.subheader("Sitemap Table")
    rows = sitemap_to_df(sitemap)
    df_sitemap = pd.DataFrame(rows)

    # Add volume column
    df_sitemap['Volume'] = df_sitemap['Keyword'].apply(
        lambda x: keyword_volumes.get(x, '')
    )

    st.dataframe(df_sitemap, use_container_width=True)

    # Downloads
    st.subheader("Download")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = df_sitemap.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="sitemap_structure.csv",
            mime="text/csv"
        )

    with col2:
        json_data = json.dumps(sitemap, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="sitemap_structure.json",
            mime="application/json"
        )

    with col3:
        tree_text = '\n'.join(tree_lines)
        st.download_button(
            label="Download Tree (TXT)",
            data=tree_text,
            file_name="sitemap_tree.txt",
            mime="text/plain"
        )

else:
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to get started")

    st.subheader("Example Output")

    example_tree = """Home
├── seo (41,000)
│   ├── seo tools (12,000)
│   └── seo audit (3,500)
├── digital marketing (29,000)
│   ├── content marketing (5,800)
│   └── social media marketing (13,000)
└── analytics (8,000)
    └── marketing analytics (700)
"""
    st.code(example_tree, language=None)

    example_df = {
        "Level": [0, 1, 1, 0, 1, 1],
        "Parent": ["", "seo", "seo", "", "digital marketing", "digital marketing"],
        "Keyword": ["seo", "seo tools", "seo audit", "digital marketing", "content marketing", "social media marketing"],
        "Volume": [41000, 12000, 3500, 29000, 5800, 13000]
    }
    st.dataframe(pd.DataFrame(example_df))
