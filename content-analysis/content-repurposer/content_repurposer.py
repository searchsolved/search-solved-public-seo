"""
Content Repurposer - Streamlit App
Transform blog posts into multiple content formats using AI.

Author: Lee Foot
Website: https://leefoot.com
"""

import streamlit as st
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
import requests
from io import BytesIO
import zipfile

st.set_page_config(
    page_title="Content Repurposer",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Content Repurposer")
st.markdown("Transform one piece of content into multiple formats: social posts, email, video scripts, and more.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    provider = st.selectbox("AI Provider", ["Anthropic (Claude)", "OpenAI (GPT)"])

    if provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"])
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])

    st.header("Optional: Firecrawl")
    firecrawl_key = st.text_input("Firecrawl API Key", type="password", help="For fetching content from URLs")

    st.header("Brand Voice")
    brand_voice = st.selectbox("Tone", [
        "Professional",
        "Casual/Friendly",
        "Authoritative",
        "Conversational",
        "Witty/Humorous"
    ])


# Output format configurations
OUTPUT_FORMATS = {
    "twitter_thread": {
        "name": "Twitter/X Thread",
        "icon": "🐦",
        "description": "5-10 tweet thread with hooks and engagement",
        "prompt": """Create a Twitter/X thread from this content.

Requirements:
- Start with a strong hook tweet that creates curiosity
- 5-10 tweets total
- Each tweet under 280 characters
- Use line breaks for readability
- Include relevant emojis sparingly
- End with a call-to-action or summary
- Add "🧵" to the first tweet

Format each tweet on its own line, numbered 1/, 2/, etc."""
    },
    "linkedin_post": {
        "name": "LinkedIn Post",
        "icon": "🔗",
        "description": "Professional post with hook and formatting",
        "prompt": """Create a LinkedIn post from this content.

Requirements:
- Strong opening hook (first line is crucial)
- 150-300 words
- Use line breaks for readability (short paragraphs)
- Include a personal insight or takeaway
- End with a question or call-to-action
- Professional but approachable tone
- No hashtags in the main text (add 3-5 at the end)"""
    },
    "email_newsletter": {
        "name": "Email Newsletter",
        "icon": "📧",
        "description": "Newsletter-style email with subject line",
        "prompt": """Create an email newsletter from this content.

Requirements:
- Compelling subject line (under 50 characters)
- Preview text (under 100 characters)
- Personal greeting
- 200-400 words body
- Clear sections with subheadings if needed
- One primary call-to-action
- Conversational, direct tone
- Sign-off

Format:
SUBJECT: [subject line]
PREVIEW: [preview text]
---
[email body]"""
    },
    "youtube_script": {
        "name": "YouTube Script",
        "icon": "🎥",
        "description": "Video script with intro, body, outro",
        "prompt": """Create a YouTube video script from this content.

Requirements:
- Hook in first 5 seconds (grab attention immediately)
- Clear intro stating what viewers will learn
- Main content broken into 3-5 clear sections
- Verbal cues for B-roll or graphics [in brackets]
- Engagement prompts (like, subscribe, comment)
- Strong outro with recap and CTA
- Aim for 5-8 minute read time
- Conversational, spoken language

Format:
[HOOK]
[INTRO]
[SECTION 1: Title]
[SECTION 2: Title]
...
[OUTRO]"""
    },
    "instagram_carousel": {
        "name": "Instagram Carousel",
        "icon": "📸",
        "description": "10-slide carousel with cover and CTA",
        "prompt": """Create an Instagram carousel from this content.

Requirements:
- 8-10 slides
- Slide 1: Attention-grabbing title/hook
- Slides 2-8: Key points (one main idea per slide)
- Slide 9: Summary/recap
- Slide 10: Call-to-action
- Each slide: headline (5-10 words) + supporting text (15-25 words)
- Simple, scannable language

Format each slide:
SLIDE 1:
Headline: [text]
Body: [text]

SLIDE 2:
..."""
    },
    "podcast_outline": {
        "name": "Podcast Episode Outline",
        "icon": "🎙️",
        "description": "Episode structure with talking points",
        "prompt": """Create a podcast episode outline from this content.

Requirements:
- Episode title (compelling, clear)
- Estimated duration
- Cold open/hook (15-30 seconds)
- Introduction with context
- 3-5 main segments with:
  - Segment title
  - Key talking points (bullets)
  - Potential stories/examples
  - Transition to next segment
- Listener takeaways
- Call-to-action/outro

Format as structured outline with timing estimates."""
    },
    "blog_summary": {
        "name": "Blog Summary/Abstract",
        "icon": "📝",
        "description": "Executive summary and key takeaways",
        "prompt": """Create a summary of this content.

Requirements:
- TL;DR (1-2 sentences)
- Executive summary (50-100 words)
- 5-7 key takeaways as bullet points
- Target audience description
- Recommended next steps/actions

Keep it scannable and actionable."""
    },
    "sms_sequence": {
        "name": "SMS/Text Sequence",
        "icon": "📱",
        "description": "2-3 text messages for marketing",
        "prompt": """Create a 3-message SMS marketing sequence from this content.

Requirements:
- Message 1: Hook/value proposition (under 160 chars)
- Message 2: Key benefit/insight (under 160 chars)
- Message 3: Call-to-action (under 160 chars)
- Each message should stand alone but build on previous
- Include link placeholder [LINK]
- Conversational, urgent tone

Format:
SMS 1: [message]
SMS 2: [message]
SMS 3: [message]"""
    },
    "reddit_post": {
        "name": "Reddit Post",
        "icon": "🤖",
        "description": "Authentic Reddit-style post",
        "prompt": """Create a Reddit post from this content.

Requirements:
- Title that works for relevant subreddits
- 200-400 word post
- Authentic, non-promotional tone
- Add value first, no hard selling
- Ask for community input/feedback
- Format with markdown (headers, bullets, bold)
- Suggest 2-3 relevant subreddits

Format:
TITLE: [title]
SUBREDDITS: r/[sub1], r/[sub2]
---
[post body]"""
    },
    "quora_answer": {
        "name": "Quora Answer",
        "icon": "❓",
        "description": "Authoritative Q&A format",
        "prompt": """Create a Quora answer from this content.

Requirements:
- Suggest 2-3 questions this content could answer
- Write an authoritative answer (250-400 words)
- Personal experience or expertise angle
- Clear structure with formatting
- Cite sources/data where relevant
- Helpful, non-promotional tone

Format:
QUESTION OPTIONS:
1. [question]
2. [question]
---
ANSWER:
[answer text]"""
    }
}


def scrape_url(url, api_key):
    """Scrape URL using Firecrawl API."""
    try:
        response = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            return data.get('data', {}).get('markdown', ''), None
        return None, data.get('error', 'Unknown error')
    except Exception as e:
        return None, str(e)


def repurpose_content(client, model, provider, content, output_format, brand_voice, title=""):
    """Repurpose content using AI."""
    format_config = OUTPUT_FORMATS[output_format]

    voice_instruction = {
        "Professional": "Use a professional, polished tone.",
        "Casual/Friendly": "Use a casual, friendly, approachable tone.",
        "Authoritative": "Use an authoritative, expert tone.",
        "Conversational": "Use a conversational, relatable tone.",
        "Witty/Humorous": "Use a witty, clever tone with appropriate humor."
    }.get(brand_voice, "")

    prompt = f"""{format_config['prompt']}

Brand Voice: {voice_instruction}

Original Content Title: {title}

Original Content:
{content[:8000]}

Create the {format_config['name']} now:"""

    try:
        if provider == "Anthropic (Claude)":
            message = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text, None
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are an expert content strategist who repurposes content for different platforms. {voice_instruction}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000
            )
            return completion.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# Main interface
st.subheader("Input Content")

input_method = st.radio("Input Method", ["Paste Content", "Fetch from URL"], horizontal=True)

content = ""
title = ""

if input_method == "Paste Content":
    title = st.text_input("Content Title", placeholder="My Blog Post Title")
    content = st.text_area("Paste your content", height=300, placeholder="Paste your blog post, article, or any content here...")

else:
    url = st.text_input("URL to fetch", placeholder="https://example.com/blog-post")
    if st.button("Fetch Content", disabled=not firecrawl_key or not url):
        with st.spinner("Fetching content..."):
            content, error = scrape_url(url, firecrawl_key)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success(f"Fetched {len(content)} characters")
                st.session_state['fetched_content'] = content
                st.session_state['fetched_url'] = url

    if 'fetched_content' in st.session_state:
        content = st.session_state['fetched_content']
        title = st.session_state.get('fetched_url', '')
        with st.expander("View fetched content"):
            st.markdown(content[:2000] + "..." if len(content) > 2000 else content)

st.markdown("---")

# Format selection
st.subheader("Select Output Formats")

col1, col2, col3 = st.columns(3)

selected_formats = []

for i, (key, config) in enumerate(OUTPUT_FORMATS.items()):
    col = [col1, col2, col3][i % 3]
    with col:
        if st.checkbox(f"{config['icon']} {config['name']}", key=key, help=config['description']):
            selected_formats.append(key)

# Quick select buttons
st.markdown("")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Select All"):
        st.session_state.update({k: True for k in OUTPUT_FORMATS.keys()})
        st.rerun()
with col2:
    if st.button("Social Only"):
        social = ["twitter_thread", "linkedin_post", "instagram_carousel", "reddit_post"]
        st.session_state.update({k: k in social for k in OUTPUT_FORMATS.keys()})
        st.rerun()
with col3:
    if st.button("Long-form Only"):
        longform = ["email_newsletter", "youtube_script", "podcast_outline"]
        st.session_state.update({k: k in longform for k in OUTPUT_FORMATS.keys()})
        st.rerun()
with col4:
    if st.button("Clear All"):
        st.session_state.update({k: False for k in OUTPUT_FORMATS.keys()})
        st.rerun()

st.markdown("---")

# Generate button
if st.button("🚀 Repurpose Content", type="primary", disabled=not api_key or not content or not selected_formats):
    if provider == "Anthropic (Claude)":
        client = Anthropic(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)

    results = {}
    progress = st.progress(0)
    status = st.empty()

    for i, format_key in enumerate(selected_formats):
        config = OUTPUT_FORMATS[format_key]
        status.text(f"Generating {config['name']}...")

        result, error = repurpose_content(client, model, provider, content, format_key, brand_voice, title)

        if result:
            results[format_key] = result
        else:
            results[format_key] = f"Error: {error}"

        progress.progress((i + 1) / len(selected_formats))

    status.text("Complete!")
    st.session_state['results'] = results

# Display results
if 'results' in st.session_state and st.session_state['results']:
    st.markdown("---")
    st.subheader("Generated Content")

    results = st.session_state['results']

    # Tabs for each result
    tabs = st.tabs([f"{OUTPUT_FORMATS[k]['icon']} {OUTPUT_FORMATS[k]['name']}" for k in results.keys()])

    for tab, (key, content) in zip(tabs, results.items()):
        with tab:
            st.markdown(content)
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    "Download",
                    content,
                    f"{key}.txt",
                    "text/plain",
                    key=f"dl_{key}"
                )

    # Download all as ZIP
    st.markdown("---")
    if st.button("📥 Download All as ZIP"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for key, content in results.items():
                zf.writestr(f"{key}.txt", content)
        zip_buffer.seek(0)

        st.download_button(
            "Download ZIP",
            zip_buffer.getvalue(),
            "repurposed_content.zip",
            "application/zip",
            key="zip_download"
        )

# Help section
with st.expander("Tips for Best Results"):
    st.markdown("""
    **Input Content:**
    - Longer, more detailed content produces better repurposed versions
    - Include key statistics, quotes, and examples in your original
    - Clear structure (headings, bullet points) helps the AI understand your content

    **Format Selection:**
    - Start with 2-3 formats to review quality before generating all
    - Social formats work best with content that has clear takeaways
    - Long-form formats (video, podcast) need more detailed source content

    **Editing:**
    - Always review and edit AI-generated content
    - Add personal anecdotes and brand-specific references
    - Check character limits for each platform
    - Customize CTAs for your specific goals
    """)

# Footer
st.markdown("---")
st.markdown("Built by 🌐 [Lee Foot](https://leefoot.com) · [LinkedIn](https://www.linkedin.com/in/lee-foot/) · 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)")
