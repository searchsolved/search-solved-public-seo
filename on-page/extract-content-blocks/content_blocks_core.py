"""
Content Block Extractor - Core Library

Shared by the Streamlit app (extract_content_blocks_app.py) and the command
line script (extract_content_blocks.py), so both produce the same output.

Fetches a page, removes scripts, styles and site chrome, asks Claude to name
the major content blocks with an XPath for each, and counts how often each
XPath appears across the pages processed.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import json

import pandas as pd
import requests
from bs4 import BeautifulSoup

CLAUDE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
MODEL = "claude-haiku-5-5"
# Thinking counts against this cap, so it leaves room for the model's
# reasoning as well as the JSON reply.
MAX_TOKENS = 16000
EFFORT = "medium"
# Filtered HTML longer than this is cut before it is sent to the model.
MAX_HTML_CHARS = 200_000
COLUMNS = ["url", "name", "xpath", "notes"]
REMOVED_TAGS = ["script", "style", "noscript", "meta", "link", "header", "footer", "nav"]
SYSTEM_PROMPT = (
    "You are an expert web scraper. Extract content blocks and provide XPath "
    "selectors. Return only valid JSON."
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_webpage(url, timeout=30):
    """Fetch a page and return its HTML. Raises on network and HTTP errors."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def filter_html(html_content):
    """Remove scripts, styles, header, footer and nav to reduce tokens."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(REMOVED_TAGS):
        tag.decompose()
    return str(soup)


def build_prompt(html_content, max_html_chars=MAX_HTML_CHARS):
    """Build the user prompt. Returns (prompt, truncated)."""
    truncated = len(html_content) > max_html_chars
    return f"""Analyze this HTML and identify major content blocks/sections.

For each block provide:
- name: descriptive name
- xpath: robust XPath expression to select this element
- notes: brief description

Focus on main content areas (hero sections, feature blocks, carousels). Skip small utility elements.

Return ONLY a JSON array with this exact format:
[
  {{"name": "Hero Section", "xpath": "//div[@class='hero']", "notes": "Main hero banner"}},
  {{"name": "Features", "xpath": "//section[@class='features']", "notes": "Feature grid"}}
]

HTML:
{html_content[:max_html_chars]}
""", truncated


def parse_blocks(text):
    """Read the JSON array of blocks out of the model's reply.

    Returns a list of dicts with name, xpath and notes. Entries without an
    xpath are dropped. Raises ValueError when the reply holds no JSON array.
    """
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON array found in the model's reply")

    blocks = []
    for item in json.loads(text[start:end]):
        if not isinstance(item, dict) or not item.get("xpath"):
            continue
        blocks.append({
            "name": str(item.get("name", "")).strip(),
            "xpath": str(item["xpath"]).strip(),
            "notes": str(item.get("notes", "")).strip(),
        })
    return blocks


def reply_text(response):
    """Text of a Claude reply, read by block type. Raises if the model declined."""
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request.")
    return "".join(block.text for block in response.content if block.type == "text")


def extract_blocks(client, html_content, model=MODEL, max_html_chars=MAX_HTML_CHARS):
    """Ask Claude for the content blocks in filtered HTML.

    Returns (blocks, info) where info has input_tokens, output_tokens and
    truncated. Raises on API errors and on a reply with no JSON array.
    """
    prompt, truncated = build_prompt(html_content, max_html_chars)
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    info = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "truncated": truncated,
    }
    return parse_blocks(reply_text(response)), info


def extract_from_url(client, url, model=MODEL, max_html_chars=MAX_HTML_CHARS, timeout=30):
    """Fetch one URL and return (rows, info). Each row has url, name, xpath, notes."""
    html = fetch_webpage(url, timeout=timeout)
    blocks, info = extract_blocks(client, filter_html(html), model, max_html_chars)
    return [{"url": url, **block} for block in blocks], info


def blocks_to_frame(rows):
    """Rows as a DataFrame with the columns url, name, xpath, notes."""
    return pd.DataFrame(rows, columns=COLUMNS)


def add_frequency(df):
    """Add a frequency column and give each XPath one name.

    frequency is the number of rows that share the XPath. The name is the
    first one seen for that XPath. Sorted by frequency, highest first.
    """
    out = df.copy()
    if out.empty:
        out["frequency"] = pd.Series(dtype="int64")
        return out
    out["frequency"] = out["xpath"].map(out["xpath"].value_counts())
    out["name"] = out["xpath"].map(out.groupby("xpath")["name"].first())
    return out.sort_values("frequency", ascending=False, kind="stable").reset_index(drop=True)


def xpath_reference(df):
    """One row per XPath: xpath, name, notes, frequency. Highest frequency first."""
    if df.empty:
        return pd.DataFrame(columns=["xpath", "name", "notes", "frequency"])
    return (
        df.groupby("xpath")
        .agg(name=("name", "first"), notes=("notes", "first"), frequency=("url", "count"))
        .reset_index()
        .sort_values("frequency", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
