"""
Higher Resolution Image Finder - Streamlit App

Uses Google Cloud Vision API to find higher resolution versions of product images.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import json
import os
import time

st.set_page_config(
    page_title="Higher Resolution Image Finder",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Higher Resolution Image Finder")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")
st.markdown("Find higher resolution versions of your product images using Google Cloud Vision API.")


def get_image_dimensions(url, timeout=15):
    """Fetch image and get its dimensions."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img.size, response.content
        return (0, 0), None
    except Exception:
        return (0, 0), None


def find_matching_images(image_url, vision_client, include_partial=True, include_similar=True):
    """Find matching images using Google Vision API."""
    try:
        from google.cloud import vision

        image = vision.Image()
        image.source.image_uri = image_url

        response = vision_client.web_detection(image=image)
        web_content = response.web_detection

        matches = {
            'full': [],
            'partial': [],
            'similar': []
        }

        # Full matches
        if web_content.full_matching_images:
            matches['full'] = [img.url for img in web_content.full_matching_images]

        # Partial matches
        if include_partial and web_content.partial_matching_images:
            matches['partial'] = [img.url for img in web_content.partial_matching_images]

        # Visually similar
        if include_similar and web_content.visually_similar_images:
            matches['similar'] = [img.url for img in web_content.visually_similar_images]

        return matches, None
    except Exception as e:
        return None, str(e)


def process_images(df, url_column, vision_client, min_improvement, max_matches,
                   include_partial, include_similar, progress_bar, status_text):
    """Process all images and find higher resolution versions."""
    results = []
    urls = df[url_column].dropna().tolist()

    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / len(urls))
        status_text.text(f"Processing {i + 1}/{len(urls)}: {url[:50]}...")

        # Get original image dimensions
        orig_dims, _ = get_image_dimensions(url)

        if orig_dims == (0, 0):
            continue

        # Find matching images
        matches, error = find_matching_images(
            url, vision_client, include_partial, include_similar
        )

        if error or not matches:
            continue

        # Check each match for higher resolution
        all_matches = matches['full'] + matches['partial'] + matches['similar']

        for match_url in all_matches[:max_matches]:
            if match_url == url:
                continue

            match_dims, _ = get_image_dimensions(match_url, timeout=10)

            if match_dims == (0, 0):
                continue

            # Calculate improvement
            orig_pixels = orig_dims[0] * orig_dims[1]
            match_pixels = match_dims[0] * match_dims[1]

            if match_pixels > orig_pixels * min_improvement:
                improvement = (match_pixels / orig_pixels - 1) * 100

                match_type = 'full'
                if match_url in matches['partial']:
                    match_type = 'partial'
                elif match_url in matches['similar']:
                    match_type = 'similar'

                results.append({
                    'Original URL': url,
                    'Original Width': orig_dims[0],
                    'Original Height': orig_dims[1],
                    'Match URL': match_url,
                    'Match Width': match_dims[0],
                    'Match Height': match_dims[1],
                    'Improvement %': round(improvement, 1),
                    'Match Type': match_type
                })

        time.sleep(0.5)  # Rate limiting

    return pd.DataFrame(results)


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🔑 Google Cloud Vision API")
    api_key_method = st.radio(
        "API Configuration",
        ["Upload Credentials JSON", "Use Environment Variable"],
        help="Choose how to provide your Google Cloud credentials"
    )

    credentials_valid = False
    vision_client = None

    if api_key_method == "Upload Credentials JSON":
        credentials_file = st.file_uploader(
            "Upload credentials.json",
            type=["json"],
            help="Download from Google Cloud Console"
        )
        if credentials_file:
            try:
                import tempfile
                from google.cloud import vision

                # Save credentials temporarily
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    creds = json.load(credentials_file)
                    json.dump(creds, f)
                    temp_path = f.name

                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
                vision_client = vision.ImageAnnotatorClient()
                credentials_valid = True
                st.success("✅ Credentials loaded")
            except Exception as e:
                st.error(f"Invalid credentials: {str(e)}")
    else:
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                from google.cloud import vision
                vision_client = vision.ImageAnnotatorClient()
                credentials_valid = True
                st.success("✅ Using environment credentials")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Set GOOGLE_APPLICATION_CREDENTIALS env variable")

    st.markdown("---")
    st.subheader("🔍 Search Settings")

    min_improvement = st.slider(
        "Minimum Improvement Ratio",
        min_value=1.1,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="Minimum size improvement to include (1.2 = 20% larger)"
    )

    max_matches = st.slider(
        "Max Matches to Check",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of matches to check per image"
    )

    include_partial = st.checkbox("Include Partial Matches", value=True)
    include_similar = st.checkbox("Include Visually Similar", value=True)

# Main content
st.markdown("### 📤 Upload Image URLs")

uploaded_file = st.file_uploader(
    "Upload CSV with image URLs",
    type=["csv"],
    help="CSV should contain a column with image URLs"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Find URL column
    url_columns = [col for col in df.columns if
                   'url' in col.lower() or 'image' in col.lower() or 'address' in col.lower()]

    if url_columns:
        url_column = st.selectbox("Select URL column", url_columns)
    else:
        url_column = st.selectbox("Select URL column", df.columns.tolist())

    urls = df[url_column].dropna().tolist()
    urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]

    st.info(f"Found **{len(urls)}** valid image URLs")

    # Preview
    with st.expander("Preview URLs"):
        st.dataframe(pd.DataFrame({'Image URL': urls[:20]}), use_container_width=True)

    if credentials_valid and vision_client:
        if st.button("🔍 Find Higher Resolution Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Searching for higher resolution images..."):
                results_df = process_images(
                    df, url_column, vision_client, min_improvement, max_matches,
                    include_partial, include_similar, progress_bar, status_text
                )

            progress_bar.empty()
            status_text.empty()

            if not results_df.empty:
                st.success(f"✅ Found {len(results_df)} higher resolution alternatives!")

                # Results tabs
                tab1, tab2 = st.tabs(["📊 Results", "📈 Summary"])

                with tab1:
                    st.dataframe(results_df, use_container_width=True, height=400)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results CSV",
                        data=csv,
                        file_name="higher_resolution_images.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with tab2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Matches", len(results_df))
                    with col2:
                        st.metric("Unique Originals", results_df['Original URL'].nunique())
                    with col3:
                        st.metric("Avg Improvement", f"{results_df['Improvement %'].mean():.1f}%")

                    # Match type distribution
                    st.subheader("Match Types")
                    type_counts = results_df['Match Type'].value_counts()
                    st.bar_chart(type_counts)
            else:
                st.warning("No higher resolution images found matching your criteria.")
    else:
        st.warning("⚠️ Please configure Google Cloud Vision API credentials in the sidebar")

else:
    st.info("👆 Upload a CSV with image URLs to get started")

    with st.expander("ℹ️ Setup Instructions"):
        st.markdown("""
        ### Google Cloud Vision API Setup

        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing
        3. Enable the **Cloud Vision API**
        4. Create a service account and download the JSON key
        5. Upload the JSON key in the sidebar

        ### CSV Format

        Your CSV should have a column containing image URLs:

        | Image URL |
        |-----------|
        | https://example.com/image1.jpg |
        | https://example.com/image2.png |
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
