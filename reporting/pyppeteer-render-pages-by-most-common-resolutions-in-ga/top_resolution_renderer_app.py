"""
Top Resolution Screenshot Renderer - Streamlit App

Takes screenshots of a URL at the most popular screen resolutions from GA data.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import asyncio
import zipfile
from io import BytesIO
import base64

st.set_page_config(
    page_title="Resolution Screenshot Renderer",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Resolution Screenshot Renderer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Renders pages at common screen resolutions
    - Uses GA data to prioritize resolutions
    - Captures screenshots for review

    **How to use:**
    1. Upload URLs to render
    2. Configure resolution settings
    3. Capture screenshots
    4. Download rendered images

    **Best for:**
    - Visual QA testing
    - Responsive design audits
    - Cross-device validation
    """)
st.markdown("Take screenshots of any URL at popular screen resolutions from your GA data.")


async def take_screenshot_async(url, width, height):
    """Take a screenshot using Pyppeteer."""
    try:
        from pyppeteer import launch

        browser = await launch(headless=True, args=['--no-sandbox'])
        page = await browser.newPage()
        await page.setViewport({'width': width, 'height': height})
        await page.goto(url, waitUntil='networkidle0', timeout=30000)
        screenshot = await page.screenshot({'fullPage': False})
        await browser.close()
        return screenshot, None
    except Exception as e:
        return None, str(e)


def take_screenshot(url, width, height):
    """Wrapper to run async function."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(take_screenshot_async(url, width, height))
        loop.close()
        return result
    except Exception as e:
        return None, str(e)


def parse_resolutions_from_ga(df):
    """Parse screen resolutions from GA export."""
    resolution_col = None
    for col in df.columns:
        if 'resolution' in col.lower() or 'screen' in col.lower():
            resolution_col = col
            break

    if not resolution_col:
        resolution_col = df.columns[0]

    resolutions = []
    for res in df[resolution_col].dropna():
        try:
            if 'x' in str(res).lower():
                parts = str(res).lower().split('x')
                width = int(parts[0].strip())
                height = int(parts[1].strip())
                if width > 0 and height > 0:
                    resolutions.append((width, height))
        except:
            continue

    return list(set(resolutions))[:10]  # Top 10 unique


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    url = st.text_input(
        "URL to Screenshot",
        value="https://www.example.com",
        help="Enter the full URL to capture"
    )

    max_resolutions = st.slider(
        "Max Resolutions",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of resolutions to capture"
    )

    st.markdown("---")
    st.markdown("### 📖 GA Export Guide")
    st.markdown("""
    1. Go to Audience → Technology → Browser & OS
    2. Set primary dimension to 'Screen Resolution'
    3. Export as CSV
    """)

# Main content
st.markdown("### 📤 Upload GA Screen Resolution Data")

uploaded_file = st.file_uploader(
    "Upload GA CSV",
    type=["csv"],
    help="Export from: Analytics > Audience > Technology > Browser & OS > Screen Resolution"
)

# Or use default resolutions
st.markdown("**Or use common resolutions:**")
use_defaults = st.checkbox("Use default popular resolutions", value=True)

default_resolutions = [
    (1920, 1080),
    (1366, 768),
    (1536, 864),
    (1280, 720),
    (1440, 900),
    (1600, 900),
    (360, 640),
    (375, 667),
    (414, 896),
    (768, 1024)
]

if uploaded_file:
    df = pd.read_csv(uploaded_file, skiprows=6)  # Skip GA header rows
    resolutions = parse_resolutions_from_ga(df)[:max_resolutions]

    st.success(f"✅ Found {len(resolutions)} resolutions from GA data")

    # Preview
    with st.expander("Preview Resolutions"):
        res_df = pd.DataFrame(resolutions, columns=['Width', 'Height'])
        st.dataframe(res_df, use_container_width=True)

elif use_defaults:
    resolutions = default_resolutions[:max_resolutions]
    st.info(f"Using {len(resolutions)} default resolutions")

    # Preview defaults
    with st.expander("Preview Default Resolutions"):
        res_df = pd.DataFrame(resolutions, columns=['Width', 'Height'])
        st.dataframe(res_df, use_container_width=True)
else:
    resolutions = []
    st.warning("Upload a GA file or enable default resolutions")

# Custom resolution input
st.markdown("### ➕ Add Custom Resolution")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    custom_width = st.number_input("Width", min_value=320, max_value=3840, value=1920)
with col2:
    custom_height = st.number_input("Height", min_value=200, max_value=2160, value=1080)
with col3:
    if st.button("Add"):
        resolutions.append((custom_width, custom_height))
        st.rerun()

if resolutions and url:
    st.markdown("---")

    if st.button("📸 Take Screenshots", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        screenshots = []
        errors = []

        for i, (width, height) in enumerate(resolutions):
            progress_bar.progress((i + 1) / len(resolutions))
            status_text.text(f"Capturing {width}x{height}...")

            screenshot, error = take_screenshot(url, width, height)

            if screenshot:
                screenshots.append({
                    'width': width,
                    'height': height,
                    'data': screenshot
                })
            else:
                errors.append({
                    'resolution': f"{width}x{height}",
                    'error': error
                })

        progress_bar.empty()
        status_text.empty()

        if screenshots:
            st.success(f"✅ Captured {len(screenshots)} screenshots!")

            # Display screenshots
            st.markdown("### 🖼️ Screenshots")

            for i, shot in enumerate(screenshots):
                with st.expander(f"{shot['width']}x{shot['height']}", expanded=(i == 0)):
                    # Display image
                    st.image(shot['data'], caption=f"{shot['width']}x{shot['height']}")

                    # Individual download
                    st.download_button(
                        f"📥 Download {shot['width']}x{shot['height']}.png",
                        data=shot['data'],
                        file_name=f"screenshot_{shot['width']}x{shot['height']}.png",
                        mime="image/png"
                    )

            # Download all as ZIP
            st.markdown("---")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for shot in screenshots:
                    filename = f"screenshot_{shot['width']}x{shot['height']}.png"
                    zip_file.writestr(filename, shot['data'])

            zip_buffer.seek(0)

            st.download_button(
                "📥 Download All Screenshots (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="screenshots.zip",
                mime="application/zip",
                use_container_width=True
            )

        if errors:
            st.warning(f"⚠️ {len(errors)} screenshots failed")
            with st.expander("View Errors"):
                for err in errors:
                    st.error(f"**{err['resolution']}**: {err['error']}")

else:
    if not url:
        st.info("👆 Enter a URL to screenshot")
    if not resolutions:
        st.info("👆 Upload GA data or enable default resolutions")

# Note about dependencies
with st.expander("ℹ️ Requirements"):
    st.markdown("""
    This tool requires **Pyppeteer** for taking screenshots:

    ```bash
    pip install pyppeteer
    ```

    On first run, Pyppeteer will download Chromium (~100MB).

    **Note:** If running on Streamlit Cloud, you may need to use
    a different screenshot method due to browser limitations.
    """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
