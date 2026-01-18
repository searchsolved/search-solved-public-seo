"""
Non-White Background Detector - Streamlit App

Detect product images that don't have white backgrounds.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="Non-White Background Detector",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Non-White Background Detector")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Detects images with non-white backgrounds
    - Flags products needing image updates
    - Helps maintain catalog consistency

    **How to use:**
    1. Upload product images or provide URLs
    2. Set background color threshold
    3. Scan for non-compliant images
    4. Export list of images to fix

    **Best for:**
    - Product image audits
    - Marketplace compliance checks
    - Catalog quality control
    """)
st.markdown("Detect product images that don't have white backgrounds.")


def get_corner_pixels(img, margin=5):
    """Get pixel values from corners of image."""
    width, height = img.size
    if img.mode != 'RGB':
        img = img.convert('RGB')

    corners = {
        'Top-Left': img.getpixel((margin, margin)),
        'Top-Right': img.getpixel((width - margin - 1, margin)),
        'Bottom-Left': img.getpixel((margin, height - margin - 1)),
        'Bottom-Right': img.getpixel((width - margin - 1, height - margin - 1))
    }
    return corners


def is_pixel_white(pixel, threshold):
    """Check if a pixel is white (or close to white)."""
    if isinstance(pixel, int):
        return pixel >= threshold
    return all(channel >= threshold for channel in pixel[:3])


def analyze_image(url, threshold, corners_to_check, margin, require_all, timeout):
    """Analyze a single image for white background."""
    result = {
        'url': url,
        'has_nonwhite_bg': False,
        'nonwhite_corners': [],
        'image_size': None,
        'corner_colors': {},
        'error': None
    }

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        result['image_size'] = f"{img.width}x{img.height}"
        result['image_data'] = response.content

        corners = get_corner_pixels(img, margin)
        nonwhite_corners = []

        for corner_name in corners_to_check:
            pixel = corners.get(corner_name)
            result['corner_colors'][corner_name] = pixel
            if pixel and not is_pixel_white(pixel, threshold):
                nonwhite_corners.append(corner_name)

        result['nonwhite_corners'] = nonwhite_corners

        if require_all:
            result['has_nonwhite_bg'] = len(nonwhite_corners) == len(corners_to_check)
        else:
            result['has_nonwhite_bg'] = len(nonwhite_corners) > 0

    except requests.exceptions.Timeout:
        result['error'] = "Timeout"
    except requests.exceptions.RequestException as e:
        result['error'] = f"Request error: {str(e)[:50]}"
    except Exception as e:
        result['error'] = f"Error: {str(e)[:50]}"

    return result


def process_images(urls, threshold, corners_to_check, margin, require_all, timeout, workers, progress_bar, status_text):
    """Process multiple images."""
    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(analyze_image, url, threshold, corners_to_check, margin, require_all, timeout): url
            for url in urls
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            progress_bar.progress(completed / len(urls))
            status_text.text(f"Processed {completed}/{len(urls)} images...")

    return results


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    threshold = st.slider(
        "Whiteness Threshold",
        min_value=200,
        max_value=255,
        value=245,
        help="RGB value threshold for 'white' (245 = nearly white)"
    )

    margin = st.slider(
        "Corner Margin (px)",
        min_value=1,
        max_value=20,
        value=5,
        help="Distance from edge to sample"
    )

    st.markdown("---")
    st.subheader("🔍 Detection Mode")

    corners_to_check = st.multiselect(
        "Corners to Check",
        ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"],
        default=["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    )

    require_all = st.checkbox(
        "Require ALL corners non-white",
        value=False,
        help="If checked, only flag images where all corners are non-white"
    )

    st.markdown("---")
    st.subheader("⚡ Performance")

    workers = st.slider(
        "Parallel Workers",
        min_value=1,
        max_value=10,
        value=5
    )

    timeout = st.slider(
        "Timeout (seconds)",
        min_value=5,
        max_value=30,
        value=10
    )

# Main content
st.markdown("### 📤 Upload Image URLs")

uploaded_file = st.file_uploader(
    "Upload CSV with image URLs",
    type=["csv"],
    help="CSV should contain a column with image URLs"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, dtype=str)
    st.success(f"✅ Loaded {len(df)} rows")

    # Find URL column
    url_columns = [c for c in df.columns if 'url' in c.lower() or 'image' in c.lower()]
    if url_columns:
        url_col = st.selectbox("Image URL Column", url_columns)
    else:
        url_col = st.selectbox("Image URL Column", df.columns.tolist())

    urls = df[url_col].dropna().tolist()
    urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]

    st.info(f"Found **{len(urls)}** valid image URLs")

    # Preview
    with st.expander("Preview URLs"):
        for url in urls[:10]:
            st.markdown(f"- {url[:80]}...")

    if st.button("🔍 Analyze Images", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = process_images(
            urls, threshold, corners_to_check, margin, require_all, timeout, workers,
            progress_bar, status_text
        )

        progress_bar.empty()
        status_text.empty()

        # Calculate stats
        nonwhite_results = [r for r in results if r['has_nonwhite_bg']]
        white_results = [r for r in results if not r['has_nonwhite_bg'] and not r['error']]
        error_results = [r for r in results if r['error']]

        st.success(f"✅ Analyzed {len(results)} images!")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", len(results))
        with col2:
            st.metric("Non-White Background", len(nonwhite_results), delta=None)
        with col3:
            st.metric("White Background", len(white_results))
        with col4:
            st.metric("Errors", len(error_results))

        # Results tabs
        tab1, tab2, tab3 = st.tabs([
            f"⚠️ Non-White ({len(nonwhite_results)})",
            f"✅ White ({len(white_results)})",
            "📊 All Results"
        ])

        with tab1:
            if nonwhite_results:
                st.subheader("Images with Non-White Backgrounds")

                for i, result in enumerate(nonwhite_results[:20]):
                    with st.expander(f"Image {i + 1}: {result['url'][:60]}..."):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            if 'image_data' in result:
                                st.image(result['image_data'], width=200)

                        with col2:
                            st.markdown(f"**Size:** {result['image_size']}")
                            st.markdown(f"**Non-white corners:** {', '.join(result['nonwhite_corners'])}")

                            st.markdown("**Corner colors (RGB):**")
                            for corner, color in result['corner_colors'].items():
                                is_white = corner not in result['nonwhite_corners']
                                icon = "✅" if is_white else "❌"
                                st.markdown(f"- {icon} {corner}: {color}")

                # Download non-white results
                nonwhite_df = pd.DataFrame([{
                    'Image URL': r['url'],
                    'Size': r['image_size'],
                    'Non-White Corners': ', '.join(r['nonwhite_corners'])
                } for r in nonwhite_results])

                csv = nonwhite_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Download Non-White Images CSV",
                    data=csv,
                    file_name="non_white_background_images.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No images with non-white backgrounds found!")

        with tab2:
            if white_results:
                st.subheader("Images with White Backgrounds")

                for i, result in enumerate(white_results[:10]):
                    with st.expander(f"Image {i + 1}: {result['url'][:60]}..."):
                        if 'image_data' in result:
                            st.image(result['image_data'], width=200)
                        st.markdown(f"**Size:** {result['image_size']}")
            else:
                st.info("No images with white backgrounds found")

        with tab3:
            # Create full results dataframe
            all_df = pd.DataFrame([{
                'Image URL': r['url'],
                'Has Non-White BG': r['has_nonwhite_bg'],
                'Non-White Corners': ', '.join(r['nonwhite_corners']) if r['nonwhite_corners'] else '',
                'Image Size': r['image_size'] or '',
                'Error': r['error'] or ''
            } for r in results])

            st.dataframe(all_df, use_container_width=True, height=400)

            csv = all_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 Download All Results CSV",
                data=csv,
                file_name="background_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Show errors
        if error_results:
            with st.expander(f"⚠️ Errors ({len(error_results)})"):
                for err in error_results[:20]:
                    st.markdown(f"- {err['url'][:50]}: {err['error']}")

else:
    st.info("👆 Upload a CSV file with image URLs to get started")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool helps you **identify product images that don't have white backgrounds**.

        **Why it matters:**
        - E-commerce platforms often require white backgrounds
        - Consistent product images improve user experience
        - Identifies images that need editing

        **How it works:**
        1. Samples pixel colors from image corners
        2. Compares against whiteness threshold
        3. Flags images with non-white corners
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
