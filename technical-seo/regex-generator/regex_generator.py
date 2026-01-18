"""
Regex Generator for SEO - Streamlit App
Generate regex patterns from plain English for SEO tasks.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
import re
import json

st.set_page_config(
    page_title="Regex Generator for SEO",
    page_icon="🔤",
    layout="wide"
)

st.title("🔤 Regex Generator for SEO")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Generates regex patterns using AI
    - Tests patterns against sample data
    - Explains regex components

    **How to use:**
    1. Enter your OpenAI API key
    2. Describe the pattern you need
    3. Provide example matches
    4. Generate and test regex

    **Best for:**
    - GA4/GTM regex creation
    - URL pattern matching
    - Data validation rules
    """)
st.markdown("Describe what you need in plain English → get regex patterns for redirects, GSC filters, and more.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    provider = st.selectbox("AI Provider", ["Anthropic (Claude)", "OpenAI (GPT)"])

    if provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251015"])
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])

    st.header("Common SEO Regex Uses")
    st.markdown("""
    - GSC query/page filters
    - Redirect rules (.htaccess)
    - Screaming Frog extraction
    - Google Analytics filters
    - Log file analysis
    - URL pattern matching
    """)


# Preset patterns
PRESET_PATTERNS = {
    "URL contains specific folder": {
        "description": "Match URLs containing a specific folder path",
        "example": "/blog/",
        "pattern": r"/blog/",
        "explanation": "Matches any URL containing /blog/"
    },
    "URL ends with specific extension": {
        "description": "Match URLs ending with .html, .php, etc.",
        "example": ".html",
        "pattern": r"\.html$",
        "explanation": "The $ anchors to end of string, \\. escapes the dot"
    },
    "URL with numeric ID": {
        "description": "Match URLs with numeric IDs like /product/123",
        "example": "/product/123",
        "pattern": r"/product/\d+",
        "explanation": "\\d+ matches one or more digits"
    },
    "URL with date pattern": {
        "description": "Match URLs with dates like /2024/01/15/",
        "example": "/2024/01/15/post-title",
        "pattern": r"/\d{4}/\d{2}/\d{2}/",
        "explanation": "\\d{4} matches exactly 4 digits (year), etc."
    },
    "Query parameter": {
        "description": "Match URLs with specific query parameter",
        "example": "?page=2",
        "pattern": r"[?&]page=\d+",
        "explanation": "[?&] matches ? or &, then page= and digits"
    },
    "Remove trailing slash": {
        "description": "Match URLs ending with trailing slash (for redirects)",
        "example": "/category/",
        "pattern": r"^(.+)/$",
        "explanation": "Captures everything before trailing slash for redirect"
    },
    "Case insensitive match": {
        "description": "Match text regardless of case",
        "example": "Blog, blog, BLOG",
        "pattern": r"(?i)blog",
        "explanation": "(?i) flag makes pattern case-insensitive"
    },
    "Match multiple extensions": {
        "description": "Match .jpg, .png, .gif, .webp",
        "example": "image.jpg, photo.png",
        "pattern": r"\.(jpg|jpeg|png|gif|webp)$",
        "explanation": "Pipe | means OR, parentheses group options"
    },
    "UTM parameters": {
        "description": "Match URLs with UTM tracking parameters",
        "example": "?utm_source=google",
        "pattern": r"[?&]utm_[a-z]+=",
        "explanation": "Matches any utm_ parameter"
    },
    "Pagination URLs": {
        "description": "Match pagination like /page/2/ or ?page=2",
        "example": "/page/2/, ?page=2",
        "pattern": r"(/page/\d+/?|[?&]page=\d+)",
        "explanation": "Matches both URL path and query string pagination"
    }
}


def generate_regex_claude(client, model, description, test_strings, context):
    """Generate regex using Claude."""
    prompt = f"""Generate a regex pattern for SEO use based on this description:

Description: {description}

Context/Platform: {context}

Test strings that should match:
{chr(10).join(f'- {s}' for s in test_strings if s)}

Provide your response as JSON with these fields:
{{
    "pattern": "the regex pattern",
    "explanation": "brief explanation of how it works",
    "flags": "any flags needed (i for case-insensitive, etc.)",
    "matches": ["list of test strings that would match"],
    "non_matches": ["examples of strings that would NOT match"],
    "platform_notes": "any platform-specific notes for {context}"
}}

Important:
- For .htaccess, use Apache mod_rewrite syntax
- For GSC, use RE2 syntax (no lookbehinds)
- For Screaming Frog, use standard regex
- Escape special characters properly
- Make the pattern as precise as possible while matching all test cases"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system="You are a regex expert specializing in SEO applications. Generate precise, well-documented regex patterns.",
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Clean JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        return json.loads(response_text), None
    except Exception as e:
        return None, str(e)


def generate_regex_openai(client, model, description, test_strings, context):
    """Generate regex using OpenAI."""
    prompt = f"""Generate a regex pattern for SEO use based on this description:

Description: {description}

Context/Platform: {context}

Test strings that should match:
{chr(10).join(f'- {s}' for s in test_strings if s)}

Provide your response as JSON with these fields:
{{
    "pattern": "the regex pattern",
    "explanation": "brief explanation of how it works",
    "flags": "any flags needed (i for case-insensitive, etc.)",
    "matches": ["list of test strings that would match"],
    "non_matches": ["examples of strings that would NOT match"],
    "platform_notes": "any platform-specific notes for {context}"
}}"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a regex expert specializing in SEO applications. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(completion.choices[0].message.content), None
    except Exception as e:
        return None, str(e)


def test_regex(pattern, test_strings, flags=""):
    """Test regex against strings."""
    results = []
    try:
        re_flags = 0
        if 'i' in flags.lower():
            re_flags |= re.IGNORECASE

        compiled = re.compile(pattern, re_flags)

        for s in test_strings:
            if s:
                match = compiled.search(s)
                results.append({
                    "string": s,
                    "matches": bool(match),
                    "match_text": match.group() if match else None
                })
    except re.error as e:
        return None, str(e)

    return results, None


# Main interface
tab1, tab2, tab3 = st.tabs(["Generate from Description", "Preset Patterns", "Test Regex"])

with tab1:
    st.subheader("Describe What You Need")

    context = st.selectbox(
        "Platform/Context",
        ["General", "Google Search Console", ".htaccess (Apache)", "Screaming Frog", "Google Analytics", "Nginx", "Log Files"]
    )

    description = st.text_area(
        "Describe the pattern you need",
        height=100,
        placeholder="E.g., Match all URLs that contain /blog/ but not /blog/tag/ or /blog/author/"
    )

    st.markdown("**Test Strings** (strings that should match)")
    col1, col2 = st.columns(2)
    with col1:
        test1 = st.text_input("Test string 1", placeholder="/blog/my-post/")
        test2 = st.text_input("Test string 2", placeholder="/blog/another-post/")
    with col2:
        test3 = st.text_input("Test string 3", placeholder="/blog/category/seo/")
        test4 = st.text_input("Test string 4", placeholder="")

    test_strings = [test1, test2, test3, test4]

    if st.button("Generate Regex", type="primary", disabled=not api_key or not description):
        with st.spinner("Generating pattern..."):
            if provider == "Anthropic (Claude)":
                client = Anthropic(api_key=api_key)
                result, error = generate_regex_claude(client, model, description, test_strings, context)
            else:
                client = OpenAI(api_key=api_key)
                result, error = generate_regex_openai(client, model, description, test_strings, context)

        if error:
            st.error(f"Error: {error}")
        elif result:
            st.success("Pattern generated!")

            # Display pattern prominently
            st.markdown("### Generated Pattern")
            pattern = result.get('pattern', '')
            flags = result.get('flags', '')

            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(pattern, language="regex")
            with col2:
                if flags:
                    st.info(f"Flags: {flags}")

            # Copy button
            st.text_input("Copy pattern:", value=pattern, key="copy_pattern")

            # Explanation
            st.markdown("### Explanation")
            st.write(result.get('explanation', ''))

            # Platform notes
            if result.get('platform_notes'):
                st.markdown("### Platform Notes")
                st.info(result['platform_notes'])

            # Test results
            if any(test_strings):
                st.markdown("### Test Results")
                test_results, test_error = test_regex(pattern, [s for s in test_strings if s], flags)
                if test_results:
                    for r in test_results:
                        if r['matches']:
                            st.success(f"✅ `{r['string']}` → matched: `{r['match_text']}`")
                        else:
                            st.error(f"❌ `{r['string']}` → no match")

            # Examples
            col1, col2 = st.columns(2)
            with col1:
                if result.get('matches'):
                    st.markdown("**Would Match:**")
                    for m in result['matches']:
                        st.write(f"✅ `{m}`")
            with col2:
                if result.get('non_matches'):
                    st.markdown("**Would NOT Match:**")
                    for m in result['non_matches']:
                        st.write(f"❌ `{m}`")

with tab2:
    st.subheader("Common SEO Regex Patterns")

    for name, data in PRESET_PATTERNS.items():
        with st.expander(f"**{name}**"):
            st.markdown(f"*{data['description']}*")
            st.markdown(f"**Example:** `{data['example']}`")
            st.code(data['pattern'], language="regex")
            st.markdown(f"**Explanation:** {data['explanation']}")

            # Quick copy
            st.text_input("Copy:", value=data['pattern'], key=f"preset_{name}")

with tab3:
    st.subheader("Test Your Regex")

    test_pattern = st.text_input("Regex Pattern", placeholder=r"/blog/\d+/")
    test_flags = st.text_input("Flags (optional)", placeholder="i", help="i = case insensitive")

    test_input = st.text_area(
        "Test Strings (one per line)",
        height=150,
        placeholder="/blog/123/\n/blog/456/\n/products/789/"
    )

    if st.button("Test Pattern", type="primary"):
        if test_pattern and test_input:
            strings = [s.strip() for s in test_input.split('\n') if s.strip()]
            results, error = test_regex(test_pattern, strings, test_flags)

            if error:
                st.error(f"Invalid regex: {error}")
            elif results:
                st.markdown("### Results")
                matches = sum(1 for r in results if r['matches'])
                st.metric("Matches", f"{matches}/{len(results)}")

                for r in results:
                    if r['matches']:
                        st.success(f"✅ `{r['string']}` → matched: `{r['match_text']}`")
                    else:
                        st.warning(f"❌ `{r['string']}` → no match")
        else:
            st.warning("Please enter a pattern and test strings.")

# Quick reference
with st.expander("Regex Quick Reference"):
    st.markdown("""
    | Symbol | Meaning | Example |
    |--------|---------|---------|
    | `.` | Any character | `a.c` matches "abc", "a1c" |
    | `*` | 0 or more | `ab*` matches "a", "ab", "abbb" |
    | `+` | 1 or more | `ab+` matches "ab", "abbb" (not "a") |
    | `?` | 0 or 1 | `ab?` matches "a", "ab" |
    | `^` | Start of string | `^/blog` matches "/blog..." |
    | `$` | End of string | `\.html$` matches "...html" |
    | `\d` | Any digit | `\d+` matches "123" |
    | `\w` | Word character | `\w+` matches "hello" |
    | `\s` | Whitespace | `\s+` matches spaces |
    | `[abc]` | Character class | `[aeiou]` matches vowels |
    | `[^abc]` | Negated class | `[^0-9]` matches non-digits |
    | `(a|b)` | Alternation | `(cat|dog)` matches either |
    | `(?i)` | Case insensitive | `(?i)blog` matches "Blog" |
    | `\\.` | Literal dot | `\\.html` matches ".html" |
    """)

# Footer
st.markdown("---")
