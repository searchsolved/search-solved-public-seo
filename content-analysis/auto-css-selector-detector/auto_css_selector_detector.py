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
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return response.text


CONTAINER_TAGS = ["div", "main", "article", "section"]
NON_CONTENT_TAGS = ["nav", "header", "footer", "aside", "script", "style"]
NON_CONTENT_WORDS = ["nav", "menu", "sidebar", "footer", "header"]


def is_non_content(element):
    """True for navigation, header, footer and sidebar elements."""
    if element.name in NON_CONTENT_TAGS:
        return True
    names = " ".join(element.get("class") or []) + " " + (element.get("id") or "")
    return any(word in names.lower() for word in NON_CONTENT_WORDS)


def content_soup(element):
    """Parse an element with its non-content descendants removed."""
    return BeautifulSoup(clean_content(str(element)), "html.parser")


def content_text(element):
    """Text of an element once navigation, headers and footers are removed."""
    return content_soup(element).get_text(" ", strip=True)


def _has_unique_simple_id(element):
    element_id = element.get("id")
    if not element_id or not re.fullmatch(r"[A-Za-z][\w-]*", element_id):
        return False
    root = element
    while root.parent is not None:
        root = root.parent
    return len(root.find_all(id=element_id)) == 1


def generate_selector(element):
    """Generate a CSS selector that matches only this element on the page.

    Uses #id when the element has a simple id that is unique in the document,
    otherwise a path of nth-of-type steps anchored at body.
    """
    if element.name == "body":
        return "body"

    if _has_unique_simple_id(element):
        return f"#{element['id']}"

    index = len(element.find_previous_siblings(element.name)) + 1
    selector = f"{element.name}:nth-of-type({index})"
    parent = element.find_parent()

    if parent is None or parent.name in ("body", "html", "[document]"):
        return f"body > {selector}"
    return f"{generate_selector(parent)} > {selector}"


def find_most_specific_selector(html, initial_selector, min_share=0.8):
    """Narrow a selector to the innermost container that still holds the content.

    Steps down into a child container only while that child holds at least
    min_share of the text, and never past the element that contains the
    content's H1. Returns a selector that matches one element.
    """
    soup = BeautifulSoup(html, "html.parser")
    try:
        current = soup.select_one(initial_selector)
    except Exception:
        current = None

    if current is None:
        return initial_selector

    while True:
        total = len(content_text(current))
        if total == 0:
            break

        children = [
            child
            for child in current.find_all(CONTAINER_TAGS, recursive=False)
            if not is_non_content(child)
        ]
        if not children:
            break

        best_child = max(children, key=lambda child: len(content_text(child)))
        if len(content_text(best_child)) / total < min_share:
            break
        if content_soup(current).find("h1") and not content_soup(best_child).find("h1"):
            break

        current = best_child

    return generate_selector(current)


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
    """Summarise the top-level content containers of the page.

    Works on the page as fetched, so each selector is valid on the live page.
    Navigation, header and footer containers are skipped, and text is measured
    with those parts removed. If the body has no container children, the body
    itself is offered.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if not body:
        return []

    summaries = []
    for child in body.find_all(CONTAINER_TAGS, recursive=False):
        if is_non_content(child):
            continue
        text = content_text(child)
        if not text:
            continue
        summaries.append({
            "selector": generate_selector(child),
            "text_length": len(text),
            "sample": text[:200] + "...",
        })

    if not summaries:
        text = content_text(body)
        summaries.append({
            "selector": "body",
            "text_length": len(text),
            "sample": text[:200] + "...",
        })

    summaries.sort(key=lambda x: x["text_length"], reverse=True)
    return summaries


def select_content_html(html, result):
    """Return the HTML to extract from, and which selector it came from.

    Tries the specific selector, then the selector the LLM chose, and falls
    back to the scoring heuristic when neither matches an element with text.
    """
    soup = BeautifulSoup(html, "html.parser")
    for key in ("specific_selector", "selector"):
        selector = result.get(key)
        if not selector:
            continue
        try:
            element = soup.select_one(selector)
        except Exception:
            element = None
        if element is None:
            continue
        cleaned = clean_content(str(element))
        if BeautifulSoup(cleaned, "html.parser").get_text(strip=True):
            return cleaned, key
    return find_main_content(clean_content(html)), "heuristic"


def extract_h1(html):
    """Extract the first H1 heading from the page."""
    soup = BeautifulSoup(html, "html.parser")
    h1_tag = soup.find("h1")
    return h1_tag.get_text(strip=True) if h1_tag else None


def ask_llm(client, model, summaries, html, url):
    """
    Ask the LLM to identify the best CSS selector for main content,
    then extract and clean the content from that selector.
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

The sections are sorted by text length, longest first. Navigation, headers and footers \
have already been left out.

Provide your answer as a JSON object with the following keys:
- 'selector': The CSS selector for the main content, copied exactly from the list (string)
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

    main_content_html, extraction_source = select_content_html(html, result)
    result["extraction_source"] = extraction_source
    markdown_text = h.handle(main_content_html)
    result["links"] = extract_links(markdown_text, url)
    result["extracted_text"] = remove_html_tags(clean_text(markdown_text))

    return result


def detect_and_extract(url, api_key, model="gpt-4o-mini", base_url="https://api.openai.com/v1"):
    """
    Main entry point: fetch a URL, detect the content selector via LLM,
    and extract the main content as markdown.

    Returns a dict with keys: url, h1, selector, specific_selector,
    reasoning, extraction_source, extracted_text, links.
    extraction_source is "specific_selector", "selector" or "heuristic".
    """
    client = create_client(api_key, base_url)
    html = get_page_content(url)
    summaries = summarize_page(html)
    result = ask_llm(client, model, summaries, html, url)
    result["url"] = url
    result["h1"] = extract_h1(html)
    return result
