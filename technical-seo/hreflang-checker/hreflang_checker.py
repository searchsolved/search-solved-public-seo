"""
Hreflang Checker - Streamlit App
Extracts and validates hreflang tags from web pages.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin
import io

st.set_page_config(
    page_title="Hreflang Checker",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Hreflang Checker")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")
st.markdown("Extract and validate hreflang tags from any website.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts hreflang tags from HTML `<link rel="alternate" hreflang="...">` tags
    - Checks HTTP headers for hreflang Link headers
    - Validates implementation for common errors

    **Validation checks:**
    - Missing self-referencing hreflang tag
    - Missing x-default tag (for multi-language sites)
    - Duplicate hreflang codes
    - Invalid language code formats

    **Valid hreflang formats:**
    - `en` - Language only
    - `en-US` - Language + region
    - `x-default` - Default/fallback page

    **How to use:**
    1. Enter URLs to check (one per line or upload CSV)
    2. Configure timeout and delay settings
    3. Click "Check Hreflang"
    4. Download the validation report

    **Best for:**
    - International SEO audits
    - Multi-language site QA
    - Pre-migration documentation
    """)

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    user_agent = st.text_input(
        "User Agent",
        value="Mozilla/5.0 (compatible; HreflangChecker/1.0)",
        help="User agent string for requests"
    )

    request_delay = st.slider(
        "Request Delay (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Delay between requests to avoid overwhelming the server"
    )

    timeout = st.slider(
        "Request Timeout (seconds)",
        min_value=5,
        max_value=60,
        value=30,
        help="Timeout for each HTTP request"
    )

    check_http_headers = st.checkbox(
        "Check HTTP Headers",
        value=True,
        help="Also check Link headers for hreflang"
    )

    validate_tags = st.checkbox(
        "Validate Hreflang Tags",
        value=True,
        help="Check for common hreflang errors"
    )


def extract_hreflang_from_html(html_content, base_url):
    """Extract hreflang tags from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    hreflang_data = []

    # Find all hreflang link tags
    hreflang_tags = soup.find_all('link', rel='alternate', hreflang=True)

    for tag in hreflang_tags:
        hreflang = tag.get('hreflang', '').strip()
        href = tag.get('href', '').strip()

        if hreflang and href:
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            hreflang_data.append({
                'hreflang': hreflang,
                'url': full_url,
                'source': 'HTML'
            })

    return hreflang_data


def extract_hreflang_from_headers(response):
    """Extract hreflang from HTTP Link headers."""
    hreflang_data = []

    link_header = response.headers.get('Link', '')
    if not link_header:
        return hreflang_data

    # Parse Link header format: <url>; rel="alternate"; hreflang="en"
    link_pattern = r'<([^>]+)>;\s*rel=["\']alternate["\'];\s*hreflang=["\']([^"\']+)["\']'
    matches = re.findall(link_pattern, link_header, re.IGNORECASE)

    for url, hreflang in matches:
        hreflang_data.append({
            'hreflang': hreflang.strip(),
            'url': url.strip(),
            'source': 'HTTP Header'
        })

    return hreflang_data


def validate_hreflang_data(hreflang_list, source_url):
    """Validate hreflang implementation and return issues."""
    issues = []

    # Get all hreflang codes
    hreflang_codes = [item['hreflang'] for item in hreflang_list]
    hreflang_urls = [item['url'] for item in hreflang_list]

    # Check 1: Self-referencing tag
    source_domain = urlparse(source_url).netloc
    has_self_reference = False
    for item in hreflang_list:
        item_domain = urlparse(item['url']).netloc
        if item_domain == source_domain or item['url'] == source_url:
            # Check if this could be self-referencing
            if source_url.rstrip('/') == item['url'].rstrip('/'):
                has_self_reference = True
                break

    if not has_self_reference and hreflang_list:
        issues.append("Missing self-referencing hreflang tag")

    # Check 2: x-default tag
    if 'x-default' not in hreflang_codes and len(hreflang_codes) > 1:
        issues.append("Missing x-default hreflang tag (recommended for multi-language sites)")

    # Check 3: Duplicate hreflang codes
    seen_codes = set()
    for code in hreflang_codes:
        if code in seen_codes:
            issues.append(f"Duplicate hreflang code: {code}")
        seen_codes.add(code)

    # Check 4: Invalid language codes (basic check)
    valid_lang_pattern = r'^[a-z]{2}(-[A-Z]{2})?$|^x-default$'
    for code in hreflang_codes:
        if not re.match(valid_lang_pattern, code, re.IGNORECASE):
            issues.append(f"Potentially invalid hreflang code format: {code}")

    return issues


def fetch_url_hreflang(url, user_agent, timeout, check_headers=True):
    """Fetch a URL and extract hreflang data."""
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        # Extract from HTML
        hreflang_data = extract_hreflang_from_html(response.text, response.url)

        # Extract from HTTP headers
        if check_headers:
            header_data = extract_hreflang_from_headers(response)
            hreflang_data.extend(header_data)

        return {
            'success': True,
            'final_url': response.url,
            'status_code': response.status_code,
            'hreflang': hreflang_data,
            'error': None
        }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'final_url': url,
            'status_code': None,
            'hreflang': [],
            'error': str(e)
        }


# Main app - Input section
st.subheader("Enter URLs to Check")

input_method = st.radio(
    "Input Method",
    options=["Single URL", "Paste URLs", "Upload CSV"],
    horizontal=True
)

urls_to_check = []

if input_method == "Single URL":
    single_url = st.text_input(
        "URL",
        placeholder="https://example.com/page"
    )
    if single_url:
        urls_to_check = [single_url]

elif input_method == "Paste URLs":
    urls_text = st.text_area(
        "URLs (one per line)",
        height=150,
        placeholder="https://example.com/page1\nhttps://example.com/page2\nhttps://example.com/page3"
    )
    if urls_text:
        urls_to_check = [u.strip() for u in urls_text.strip().split('\n') if u.strip()]

elif input_method == "Upload CSV":
    csv_file = st.file_uploader("Upload CSV with URLs", type=['csv'])
    if csv_file:
        df = pd.read_csv(csv_file)
        url_column = st.selectbox("Select URL column", options=df.columns.tolist())
        urls_to_check = df[url_column].dropna().tolist()
        st.info(f"Found {len(urls_to_check)} URLs in the CSV")

# Run button
if urls_to_check:
    if st.button("Check Hreflang Tags", type="primary"):
        results = []
        all_hreflang = []
        all_issues = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, url in enumerate(urls_to_check):
            status_text.text(f"Processing {i+1}/{len(urls_to_check)}: {url[:60]}...")

            result = fetch_url_hreflang(url, user_agent, timeout, check_http_headers)

            if result['success']:
                # Validate if enabled
                issues = []
                if validate_tags and result['hreflang']:
                    issues = validate_hreflang_data(result['hreflang'], result['final_url'])

                results.append({
                    'source_url': url,
                    'final_url': result['final_url'],
                    'status': 'OK',
                    'hreflang_count': len(result['hreflang']),
                    'issues_count': len(issues),
                    'issues': '; '.join(issues) if issues else ''
                })

                for item in result['hreflang']:
                    all_hreflang.append({
                        'source_url': url,
                        'hreflang_code': item['hreflang'],
                        'alternate_url': item['url'],
                        'detection_source': item['source']
                    })

                for issue in issues:
                    all_issues.append({
                        'source_url': url,
                        'issue': issue
                    })
            else:
                results.append({
                    'source_url': url,
                    'final_url': result['final_url'],
                    'status': 'ERROR',
                    'hreflang_count': 0,
                    'issues_count': 1,
                    'issues': result['error']
                })

            progress_bar.progress((i + 1) / len(urls_to_check))

            if request_delay > 0 and i < len(urls_to_check) - 1:
                time.sleep(request_delay)

        progress_bar.progress(100)
        status_text.text("Done!")

        # Results summary
        st.subheader("Results Summary")

        results_df = pd.DataFrame(results)
        hreflang_df = pd.DataFrame(all_hreflang)
        issues_df = pd.DataFrame(all_issues)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("URLs Checked", len(results_df))
        with col2:
            successful = results_df[results_df['status'] == 'OK'].shape[0]
            st.metric("Successful", successful)
        with col3:
            total_tags = results_df['hreflang_count'].sum()
            st.metric("Total Hreflang Tags", int(total_tags))
        with col4:
            total_issues = len(issues_df)
            st.metric("Validation Issues", total_issues)

        # Tabs for detailed results
        tab1, tab2, tab3 = st.tabs(["URL Summary", "All Hreflang Tags", "Validation Issues"])

        with tab1:
            st.dataframe(results_df, use_container_width=True, height=300)

        with tab2:
            if not hreflang_df.empty:
                st.dataframe(hreflang_df, use_container_width=True, height=300)

                # Language distribution
                st.markdown("**Hreflang Code Distribution:**")
                code_counts = hreflang_df['hreflang_code'].value_counts()
                st.bar_chart(code_counts)
            else:
                st.warning("No hreflang tags found.")

        with tab3:
            if not issues_df.empty:
                st.dataframe(issues_df, use_container_width=True, height=300)
            else:
                st.success("No validation issues found!")

        # Download section
        st.subheader("Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = hreflang_df.to_csv(index=False) if not hreflang_df.empty else ""
            st.download_button(
                "Download Hreflang Tags (CSV)",
                csv,
                file_name=f"hreflang_tags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                disabled=hreflang_df.empty
            )

        with col2:
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Summary (CSV)",
                csv,
                file_name=f"hreflang_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col3:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Summary')
                if not hreflang_df.empty:
                    hreflang_df.to_excel(writer, index=False, sheet_name='Hreflang Tags')
                if not issues_df.empty:
                    issues_df.to_excel(writer, index=False, sheet_name='Issues')
            excel_data = output.getvalue()
            st.download_button(
                "Download Full Report (Excel)",
                excel_data,
                file_name=f"hreflang_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("Enter URLs above to check hreflang tags.")


# Footer
st.markdown("---")
