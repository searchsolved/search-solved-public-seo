# Author   : Lee Foot
# Website  : https://leefoot.com
"""
Auto CSS Selector Detector - Core Library

Uses an LLM to automatically identify the best CSS selector for a page's main
content area, then extracts and converts that content to Markdown.

Author: Lee Foot
Website: https://leefoot.com
"""

import json
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse, urljoin, urlunparse

import html2text
import markdown
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


def create_client(api_key, base_url="https://api.openai.com/v1"):
    """Create an OpenAI-compatible API client."""
    return OpenAI(api_key=api_key, base_url=base_url)


def get_page_content(url):
    """Fetch the HTML content of a URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.encoding = response.apparent_encoding
    return response.text


def generate_selector(element):
    """Generate a CSS selector path for a given BeautifulSoup element."""
    if element.name == "body":
        return "body"

    siblings = element.find_previous_siblings(element.name)
    index = len(siblings) + 1 if siblings else 1
    selector = f"{element.name}:nth-of-type({index})"
    parent = element.find_parent()

    if parent and parent.name != "body":
        return f"{generate_selector(parent)} > {selector}"
    return selector


def find_most_specific_selector(html, initial_selector):
    """Refine a selector to target the most content-rich descendant."""
    soup = BeautifulSoup(html, "html.parser")
    initial_element = soup.select_one(initial_selector)

    if not initial_element:
        return initial_selector

    current_selector = initial_selector
    current_element = initial_element

    while True:
        children = current_element.find_all(recursive=False)
        if not children:
            break

        best_child = None
        best_score = 0

        for child in children:
            child_text = child.get_text(strip=True)
            parent_text_len = len(current_element.get_text(strip=True))
            if parent_text_len == 0:
                continue
            score = len(child_text) * (len(child_text) / parent_text_len)

            if score > best_score:
                best_child = child
                best_score = score

        if best_child:
            current_selector = f"{current_selector} > {best_child.name}"
            current_element = best_child
        else:
            break

    return current_selector


def deduplicate_content(text):
    """Remove near-duplicate paragraphs from extracted text."""
    paragraphs = text.split("\n\n")
    unique_paragraphs = []

    for para in paragraphs:
        if not any(
            SequenceMatcher(None, para, up).ratio() > 0.8
            for up in unique_paragraphs
        ):
            unique_paragraphs.append(para)

    return "\n\n".join(unique_paragraphs)


def clean_content(html):
    """Remove navigation, headers, footers, and other non-content elements."""
    soup = BeautifulSoup(html, "html.parser")

    for elem in soup(["nav", "header", "footer", "aside", "script", "style"]):
        elem.decompose()

    for elem in soup.find_all(
        class_=lambda x: x
        and any(
            word in x.lower()
            for word in ["nav", "menu", "sidebar", "footer", "header"]
        )
    ):
        elem.decompose()

    for elem in soup.find_all(
        id=lambda x: x
        and any(
            word in x.lower()
            for word in ["nav", "menu", "sidebar", "footer", "header"]
        )
    ):
        elem.decompose()

    return str(soup)


def score_content_relevance(element):
    """Score an element based on how likely it is to be main content."""
    text = element.get_text(strip=True)
    word_count = len(text.split())
    link_density = len(element.find_all("a")) / max(word_count, 1)

    score = word_count * (1 - link_density)

    if element.find(["h1", "h2", "h3"]):
        score *= 1.5

    return score


def find_main_content(html):
    """Identify the main content element by scoring candidates."""
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.find_all(["div", "article", "section", "main"])

    elements = [
        elem
        for elem in elements
        if not elem.find_parent(["header", "footer", "nav", "aside"])
    ]

    if not elements:
        return str(soup.body) if soup.body else str(soup)

    scored_elements = [(elem, score_content_relevance(elem)) for elem in elements]

    if not scored_elements:
        return str(soup.body) if soup.body else str(soup)

    main_content = max(scored_elements, key=lambda x: x[1])[0]
    return str(main_content)


def remove_html_tags(text):
    """Strip all HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text)


def extract_links(markdown_text, base_url):
    """Extract internal links from markdown content."""
    link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
    links = re.findall(link_pattern, markdown_text)

    base_domain = urlparse(base_url).netloc
    formatted_links = []

    for anchor, url in links:
        full_url = urljoin(base_url, url)
        parsed_url = urlparse(full_url)

        if parsed_url.netloc == base_domain:
            clean_url = urlunparse(parsed_url._replace(query=""))
            formatted_links.append(f"('{anchor}', '{clean_url}')")

    return str(formatted_links)


def clean_text(markdown_text):
    """Convert markdown to clean plaintext with deduplication."""
    html_content = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html_content, "html.parser")

    for a in soup.find_all("a"):
        a.replace_with(a.text)

    for img in soup.find_all("img"):
        img.decompose()

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.string = f"\n\n{tag.text.upper()}\n\n"

    for p in soup.find_all("p"):
        p.insert_after(soup.new_string("\n\n"))

    for ul in soup.find_all("ul"):
        for li in ul.find_all("li"):
            li.insert_before(soup.new_string("* "))
        ul.insert_after(soup.new_string("\n\n"))

    text = soup.get_text()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    text = deduplicate_content(text)

    return text


def summarize_page(html):
    """Generate section summaries from top-level content containers."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if not body:
        return []

    summaries = []
    for child in body.find_all(
        ["div", "main", "article", "section"], recursive=False
    ):
        section_data = {
            "selector": generate_selector(child),
            "text_length": len(child.get_text(strip=True)),
            "sample": child.get_text(strip=True)[:200] + "...",
        }
        summaries.append(section_data)

    summaries.sort(key=lambda x: x["text_length"], reverse=True)
    return summaries


def extract_h1(html):
    """Extract the first H1 heading from the page."""
    soup = BeautifulSoup(html, "html.parser")
    h1_tag = soup.find("h1")
    return h1_tag.get_text(strip=True) if h1_tag else None


def ask_llm(client, model, summaries, html, url):
    """
    Ask the LLM to identify the best CSS selector for main content,
    then extract and clean the content.
    """
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0

    truncated_summaries = [
        {
            **s,
            "sample": s["sample"][:200] + "..."
            if len(s["sample"]) > 200
            else s["sample"],
        }
        for s in summaries[:5]
    ]

    prompt = f"""Given the following summaries of webpage sections, identify the selector \
that most likely contains the main body content:

{json.dumps(truncated_summaries, indent=2)}

The sections are sorted by a content score, which considers factors like text length, \
text-to-HTML ratio, number of paragraphs, average paragraph length, presence of headings, \
links, and images.

Provide your answer as a JSON object with the following keys:
- 'selector': The CSS selector for the main content (string)
- 'reasoning': Your reasoning for choosing this selector (string)
"""

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant designed to analyse webpage structures "
                    "and identify main content areas. Provide valid CSS selectors."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    result = json.loads(response.choices[0].message.content)

    specific_selector = find_most_specific_selector(html, result["selector"])
    result["specific_selector"] = specific_selector

    cleaned_html = clean_content(html)
    main_content_html = find_main_content(cleaned_html)
    markdown_text = h.handle(main_content_html)
    result["links"] = extract_links(markdown_text, url)
    result["extracted_text"] = remove_html_tags(clean_text(markdown_text))

    return result


def detect_and_extract(url, api_key, model="gpt-4o-mini", base_url="https://api.openai.com/v1"):
    """
    Main entry point: fetch a URL, detect the content selector via LLM,
    and extract the main content as markdown.

    Returns a dict with keys: url, h1, selector, specific_selector,
    reasoning, extracted_text, links.
    """
    client = create_client(api_key, base_url)
    html = get_page_content(url)
    clean_html = clean_content(html)
    summaries = summarize_page(clean_html)
    result = ask_llm(client, model, summaries, clean_html, url)
    result["url"] = url
    result["h1"] = extract_h1(html)
    return result
