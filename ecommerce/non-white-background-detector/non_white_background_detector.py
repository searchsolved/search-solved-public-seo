####################################################################################
#                                                                                  #
#  Non-White Background Detector                                                   #
#                                                                                  #
#  Detect product images that don't have white backgrounds for e-commerce QA.      #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://x.com/LeeFootSEO                                              #
####################################################################################

"""
Non-White Background Detector - Streamlit App

Analyzes product images to detect non-white backgrounds. Useful for e-commerce
quality assurance to identify images that need background removal or correction.

Requirements:
    pip install streamlit pandas pillow requests
"""

import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# App Configuration
st.set_page_config(
    page_title="Non-White Background Detector",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Non-White Background Detector")
st.markdown("""
Detect product images that don't have white backgrounds.
Upload a CSV with image URLs to identify images needing background fixes.
""")

# Sidebar configuration
st.sidebar.header("Detection Settings")

whiteness_threshold = st.sidebar.slider(
    "Whiteness Threshold",
    min_value=200,
    max_value=255,
    value=245,
    help="Pixels below this value (0-255) are considered non-white. Higher = stricter."
)

check_corners = st.sidebar.multiselect(
    "Corners to Check",
    options=["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"],
    default=["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"],
    help="Which corners to sample for background color detection"
)

corner_margin = st.sidebar.slider(
    "Corner Margin (pixels)",
    min_value=0,
    max_value=50,
    value=5,
    help="How many pixels from the edge to sample"
)

require_all_corners = st.sidebar.checkbox(
    "Require ALL corners to be non-white",
    value=False,
    help="If checked, only flags images where all selected corners are non-white"
)

request_timeout = st.sidebar.slider(
    "Request Timeout (seconds)",
    min_value=5,
    max_value=60,
    value=10,
    help="Timeout for downloading each image"
)

max_workers = st.sidebar.slider(
    "Parallel Downloads",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of images to download simultaneously"
)


def get_corner_pixels(img, margin=5):
    """Get pixel values from image corners."""
    width, height = img.size

    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    corners = {}

    # Top-Left
    corners['Top-Left'] = img.getpixel((margin, margin))

    # Top-Right
    corners['Top-Right'] = img.getpixel((width - margin - 1, margin))

    # Bottom-Left
    corners['Bottom-Left'] = img.getpixel((margin, height - margin - 1))

    # Bottom-Right
    corners['Bottom-Right'] = img.getpixel((width - margin - 1, height - margin - 1))

    return corners


def is_pixel_white(pixel, threshold):
    """Check if a pixel is considered white."""
    if isinstance(pixel, int):
        # Grayscale
        return pixel >= threshold
    else:
        # RGB - check if all channels are above threshold
        return all(channel >= threshold for channel in pixel[:3])


def analyze_image(url, threshold, corners_to_check, margin, require_all):
    """Analyze a single image for non-white background."""
    result = {
        'url': url,
        'has_nonwhite_bg': False,
        'corner_values': {},
        'nonwhite_corners': [],
        'image_size': None,
        'error': None
    }

    try:
        # Download image
        response = requests.get(url, timeout=request_timeout)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        result['image_size'] = f"{img.width}x{img.height}"

        # Get corner pixels
        corners = get_corner_pixels(img, margin)

        # Check selected corners
        nonwhite_corners = []
        for corner_name in corners_to_check:
            pixel = corners.get(corner_name)
            if pixel:
                result['corner_values'][corner_name] = pixel
                if not is_pixel_white(pixel, threshold):
                    nonwhite_corners.append(corner_name)

        result['nonwhite_corners'] = nonwhite_corners

        # Determine if background is non-white
        if require_all:
            # All selected corners must be non-white
            result['has_nonwhite_bg'] = len(nonwhite_corners) == len(corners_to_check)
        else:
            # Any corner being non-white triggers detection
            result['has_nonwhite_bg'] = len(nonwhite_corners) > 0

    except requests.exceptions.Timeout:
        result['error'] = "Timeout"
    except requests.exceptions.RequestException as e:
        result['error'] = f"Request error: {str(e)[:50]}"
    except Exception as e:
        result['error'] = f"Error: {str(e)[:50]}"

    return result


def process_images(urls, threshold, corners_to_check, margin, require_all):
    """Process multiple images with progress tracking."""
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(
                analyze_image, url, threshold, corners_to_check, margin, require_all
            ): url for url in urls
        }

        completed = 0
        total = len(urls)

        for future in as_completed(future_to_url):
            result = future.result()
            results.append(result)

            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Processing: {completed}/{total} images")

    progress_bar.progress(1.0)
    status_text.text("Processing complete!")

    return results


# File uploader
st.header("Upload Image URLs")
uploaded_file = st.file_uploader(
    "Upload CSV with image URLs",
    type=["csv"],
    help="CSV file should contain a column with image URLs"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, dtype=str)
        st.success(f"Loaded {len(df):,} rows")

        # Column selection
        url_columns = df.columns.tolist()

        # Try to auto-detect image column
        image_col_guess = None
        for col in url_columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['image', 'img', 'url', 'photo', 'picture', 'src']):
                image_col_guess = col
                break

        url_column = st.selectbox(
            "Select Image URL Column",
            options=url_columns,
            index=url_columns.index(image_col_guess) if image_col_guess else 0
        )

        # Preview data
        with st.expander("Preview Data"):
            st.dataframe(df[[url_column]].head(20))

        # Get valid URLs
        urls = df[url_column].dropna().tolist()
        urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]

        st.info(f"Found {len(urls):,} valid image URLs")

        if len(urls) == 0:
            st.error("No valid URLs found. URLs must start with http:// or https://")
            st.stop()

        # Limit warning
        if len(urls) > 500:
            st.warning(f"Large dataset ({len(urls)} images). This may take a while.")

        # Process button
        if st.button("Detect Non-White Backgrounds", type="primary"):
            with st.spinner("Analyzing images..."):
                results = process_images(
                    urls,
                    whiteness_threshold,
                    check_corners,
                    corner_margin,
                    require_all_corners
                )

            # Convert to DataFrame
            results_df = pd.DataFrame([
                {
                    'Image URL': r['url'],
                    'Has Non-White BG': r['has_nonwhite_bg'],
                    'Non-White Corners': ', '.join(r['nonwhite_corners']) if r['nonwhite_corners'] else '',
                    'Image Size': r['image_size'] or '',
                    'Error': r['error'] or ''
                }
                for r in results
            ])

            # Merge back with original data
            df_merged = df.copy()
            df_merged['Has Non-White BG'] = df_merged[url_column].map(
                dict(zip(results_df['Image URL'], results_df['Has Non-White BG']))
            )
            df_merged['Non-White Corners'] = df_merged[url_column].map(
                dict(zip(results_df['Image URL'], results_df['Non-White Corners']))
            )
            df_merged['Detection Error'] = df_merged[url_column].map(
                dict(zip(results_df['Image URL'], results_df['Error']))
            )

            # Display results
            st.header("Detection Results")

            # Summary metrics
            total_images = len(results)
            nonwhite_count = sum(1 for r in results if r['has_nonwhite_bg'])
            white_count = sum(1 for r in results if not r['has_nonwhite_bg'] and not r['error'])
            error_count = sum(1 for r in results if r['error'])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", f"{total_images:,}")
            with col2:
                st.metric("Non-White Background", f"{nonwhite_count:,}",
                         delta=f"{nonwhite_count/total_images*100:.1f}%" if total_images > 0 else "0%")
            with col3:
                st.metric("White Background", f"{white_count:,}")
            with col4:
                st.metric("Errors", f"{error_count:,}")

            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["Non-White Images", "All Results", "Preview"])

            with tab1:
                st.subheader("Images with Non-White Backgrounds")
                nonwhite_df = df_merged[df_merged['Has Non-White BG'] == True]

                if len(nonwhite_df) > 0:
                    st.dataframe(nonwhite_df, use_container_width=True, hide_index=True)

                    # Download non-white only
                    output = BytesIO()
                    nonwhite_df.to_csv(output, index=False, encoding='utf-8-sig')
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Non-White Images (CSV)",
                        data=output,
                        file_name="non_white_background_images.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("No non-white backgrounds detected!")

            with tab2:
                st.subheader("All Results")
                st.dataframe(df_merged, use_container_width=True, hide_index=True)

                # Download all results
                output = BytesIO()
                df_merged.to_csv(output, index=False, encoding='utf-8-sig')
                output.seek(0)

                st.download_button(
                    label="📥 Download All Results (CSV)",
                    data=output,
                    file_name="background_detection_results.csv",
                    mime="text/csv"
                )

            with tab3:
                st.subheader("Image Preview (Non-White Backgrounds)")

                if nonwhite_count > 0:
                    preview_limit = min(12, nonwhite_count)
                    st.caption(f"Showing first {preview_limit} non-white background images")

                    nonwhite_urls = [r['url'] for r in results if r['has_nonwhite_bg']][:preview_limit]

                    # Display in grid
                    cols = st.columns(4)
                    for idx, url in enumerate(nonwhite_urls):
                        with cols[idx % 4]:
                            try:
                                st.image(url, use_container_width=True)
                                st.caption(url.split('/')[-1][:30])
                            except Exception:
                                st.error("Failed to load")
                else:
                    st.info("No non-white background images to preview")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload a CSV file with image URLs to get started.")

    st.markdown("""
    ### How it works:

    1. Upload a CSV containing product image URLs
    2. The tool downloads each image and checks corner pixels
    3. Images with non-white corners are flagged
    4. Download results for further action

    ### Detection Settings:

    - **Whiteness Threshold**: Pixels must be above this value (0-255) to be considered "white"
    - **Corners to Check**: Which image corners to sample
    - **Corner Margin**: How far from the edge to sample (avoids border artifacts)
    - **Require ALL Corners**: Only flag if all selected corners are non-white

    ### Use Cases:

    - **E-commerce QA**: Find product images needing background removal
    - **Marketplace compliance**: Ensure images meet white background requirements
    - **Bulk image auditing**: Quickly identify problematic images
    - **Photography workflow**: Flag images for post-processing

    ### Supported Formats:

    - JPEG, PNG, GIF, WebP, and most common image formats
    - Images must be accessible via HTTP/HTTPS URLs
    """)
