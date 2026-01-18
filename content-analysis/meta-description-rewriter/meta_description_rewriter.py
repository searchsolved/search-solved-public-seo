"""
Meta Description Rewriter - Streamlit App
Rewrite meta descriptions using AI with proper length optimization.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from openai import OpenAI
from time import sleep
from io import BytesIO

st.set_page_config(
    page_title="Meta Description Rewriter",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Meta Description Rewriter")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Rewrites meta descriptions using AI (GPT-4)
    - Optimizes length for desktop and mobile displays
    - Applies customizable tone and style settings

    **How to use:**
    1. Enter your OpenAI API key in the sidebar
    2. Upload a CSV with columns: URL, Meta Description, H1, Title
    3. Configure tone and length settings
    4. Click "Rewrite Descriptions" to process

    **Best for:**
    - Bulk meta description optimization
    - SEO content refreshes
    - Ensuring consistent brand voice across pages
    """)
st.markdown("Rewrite meta descriptions with AI - professional tone, proper length, SEO-optimized.")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")

    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"], index=0)

    st.header("📏 Length Settings")
    max_desktop = st.slider("Desktop max chars", 140, 170, 160)
    max_mobile = st.slider("Mobile max chars", 100, 140, 130)

    st.header("🎨 Tone Settings")
    tone = st.selectbox("Writing Tone", [
        "Professional",
        "Friendly",
        "Authoritative",
        "Conversational",
        "Technical"
    ])

    avoid_exclamations = st.checkbox("Avoid exclamation marks", value=True)
    avoid_hyperbole = st.checkbox("Avoid hyperbole/clickbait", value=True)


def smart_truncate(text, max_chars):
    """Truncate at word boundary if exceeding max chars."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars - 3]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.strip() + "..."


def optimize_length(text, max_desktop, max_mobile):
    """Optimize for desktop and mobile lengths."""
    desktop = text if len(text) <= max_desktop else smart_truncate(text, max_desktop)
    mobile = text if len(text) <= max_mobile else smart_truncate(text, max_mobile)
    return desktop, mobile


def rewrite_meta_description(client, model, url, current_meta, h1, title, tone, avoid_excl, avoid_hype, max_chars):
    """Rewrite a meta description using AI."""

    tone_instructions = {
        "Professional": "Use clear, professional language suitable for business audiences.",
        "Friendly": "Use warm, approachable language that feels welcoming.",
        "Authoritative": "Use confident, expert language that establishes credibility.",
        "Conversational": "Use natural, casual language as if talking to a friend.",
        "Technical": "Use precise, technical language appropriate for expert audiences."
    }

    restrictions = []
    if avoid_excl:
        restrictions.append("No exclamation marks")
    if avoid_hype:
        restrictions.append("No hyperbole, clickbait phrases, or artificial urgency")

    restrictions_text = "\n".join(f"- {r}" for r in restrictions) if restrictions else "None"

    messages = [
        {
            "role": "system",
            "content": f"""You are an expert SEO copywriter specializing in meta descriptions.

Tone: {tone_instructions.get(tone, tone_instructions['Professional'])}

Restrictions:
{restrictions_text}

Key principles:
1. Use natural, readable language
2. Clearly state the main value proposition
3. Include relevant keywords naturally from the H1/title
4. Focus on user benefit
5. Length: {max_chars} characters maximum
6. Must be a complete sentence (no truncation)

Format: Start with benefit or key information, then supporting detail.

Example transformations:
- Bad: "Skyrocket your business! The BEST tools for SUCCESS! Act NOW!"
- Good: "Compare 10 cloud tools for growing businesses. A practical guide to improving efficiency."

- Bad: "AMAZING tips that will TRANSFORM your life forever!!!"
- Good: "10 practical productivity tips to help manage your workday more effectively."
"""
        },
        {
            "role": "user",
            "content": f"""Rewrite this meta description:

URL: {url or 'Not provided'}
H1: {h1 or 'Not provided'}
Title: {title or 'Not provided'}
Current Meta: {current_meta or 'Not provided'}

Return ONLY the rewritten meta description, no quotes or explanation."""
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        result = completion.choices[0].message.content.strip()
        # Remove any surrounding quotes
        result = result.strip('"\'')
        return result, None
    except Exception as e:
        return None, str(e)


# Main interface
tab1, tab2 = st.tabs(["✍️ Single Rewrite", "📊 Bulk Rewrite"])

with tab1:
    st.subheader("Rewrite a Single Meta Description")

    col1, col2 = st.columns(2)

    with col1:
        url = st.text_input("URL (optional)", placeholder="https://example.com/page")
        h1 = st.text_input("H1 Heading", placeholder="Page heading for context")
        title = st.text_input("Page Title", placeholder="Page title for context")
        current_meta = st.text_area("Current Meta Description", height=100,
                                     placeholder="Enter the current meta description...")

        if current_meta:
            st.caption(f"Current length: {len(current_meta)} characters")

    with col2:
        if st.button("Rewrite", type="primary", disabled=not api_key or not current_meta):
            with st.spinner("Rewriting meta description..."):
                client = OpenAI(api_key=api_key)
                rewritten, error = rewrite_meta_description(
                    client, model, url, current_meta, h1, title,
                    tone, avoid_exclamations, avoid_hyperbole, max_desktop
                )

            if error:
                st.error(f"Error: {error}")
            elif rewritten:
                desktop, mobile = optimize_length(rewritten, max_desktop, max_mobile)

                st.success("Rewritten successfully!")

                st.markdown("**Original:**")
                st.text(current_meta)
                st.caption(f"{len(current_meta)} characters")

                st.markdown("**Rewritten (Desktop):**")
                st.text(desktop)
                st.caption(f"{len(desktop)} characters")

                if desktop != mobile:
                    st.markdown("**Rewritten (Mobile):**")
                    st.text(mobile)
                    st.caption(f"{len(mobile)} characters")

                # Copy buttons
                st.code(desktop, language=None)

with tab2:
    st.subheader("Bulk Rewrite from CSV/Excel")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            # Column mapping
            st.markdown("**Map your columns:**")
            col1, col2 = st.columns(2)

            with col1:
                url_col = st.selectbox("URL Column", ["(None)"] + list(df.columns))
                meta_col = st.selectbox("Meta Description Column", list(df.columns))

            with col2:
                h1_col = st.selectbox("H1 Column (optional)", ["(None)"] + list(df.columns))
                title_col = st.selectbox("Title Column (optional)", ["(None)"] + list(df.columns))

            # Limit for testing
            max_rows = st.number_input("Max rows to process (0 = all)", 0, len(df), 0)

            if st.button("Rewrite All", type="primary", disabled=not api_key):
                client = OpenAI(api_key=api_key)

                results = []
                process_df = df.head(max_rows) if max_rows > 0 else df

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, row in process_df.iterrows():
                    status_text.text(f"Processing row {idx + 1} of {len(process_df)}...")

                    url = row[url_col] if url_col != "(None)" else ""
                    current_meta = str(row.get(meta_col, "")) if pd.notna(row.get(meta_col)) else ""
                    h1 = str(row.get(h1_col, "")) if h1_col != "(None)" and pd.notna(row.get(h1_col)) else ""
                    page_title = str(row.get(title_col, "")) if title_col != "(None)" and pd.notna(row.get(title_col)) else ""

                    if current_meta:
                        rewritten, error = rewrite_meta_description(
                            client, model, url, current_meta, h1, page_title,
                            tone, avoid_exclamations, avoid_hyperbole, max_desktop
                        )

                        if rewritten:
                            desktop, mobile = optimize_length(rewritten, max_desktop, max_mobile)
                            results.append({
                                'url': url,
                                'original_meta': current_meta,
                                'original_length': len(current_meta),
                                'rewritten_desktop': desktop,
                                'desktop_length': len(desktop),
                                'rewritten_mobile': mobile,
                                'mobile_length': len(mobile)
                            })
                        else:
                            results.append({
                                'url': url,
                                'original_meta': current_meta,
                                'error': error or "Unknown error"
                            })
                    else:
                        results.append({
                            'url': url,
                            'original_meta': "",
                            'error': "No meta description provided"
                        })

                    sleep(0.5)  # Rate limiting
                    progress_bar.progress((idx + 1) / len(process_df))

                status_text.text("Complete!")

                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # Summary
                success_count = len([r for r in results if 'rewritten_desktop' in r])
                st.metric("Successfully Rewritten", f"{success_count}/{len(results)}")

                # Download
                col1, col2 = st.columns(2)
                with col1:
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "rewritten_meta_descriptions.csv", "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False)
                    st.download_button("Download Excel", output.getvalue(),
                                       "rewritten_meta_descriptions.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Footer
st.markdown("---")
