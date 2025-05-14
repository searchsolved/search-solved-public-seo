import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from tempfile import NamedTemporaryFile


def detect_main_subject(img_array):
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # For product photos with white backgrounds (common in e-commerce)
    # Try to find the bounding box of non-white areas

    # Threshold to separate foreground from background
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours of non-white areas
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no significant contours, try alternative method for darker objects
    if not contours or max(cv2.contourArea(c) for c in contours) < img_array.size * 0.01:
        # Different threshold for darker objects
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If still no meaningful contours, try adaptive thresholding
    if not contours or max(cv2.contourArea(c) for c in contours) < img_array.size * 0.01:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If still no good contours, try Otsu's thresholding method
    if not contours or max(cv2.contourArea(c) for c in contours) < img_array.size * 0.01:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found after all attempts, return the center of the image
    if not contours:
        return img_array.shape[1] // 2, img_array.shape[0] // 2

    # Filter out small noise contours (less than 0.5% of image area)
    min_area = img_array.size * 0.005
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not contours:
        return img_array.shape[1] // 2, img_array.shape[0] // 2

    # Method 1: Find the center of the bounding box of all contours combined
    all_points = np.vstack([c.reshape(-1, 2) for c in contours])
    x, y, w, h = cv2.boundingRect(all_points)
    center_x = x + w // 2
    center_y = y + h // 2

    return center_x, center_y


def enhance_detection_visualization(img, original_filename):
    """Create visual debugging output showing detection results."""
    # Convert to RGB array for processing
    img_array = np.array(img)

    # Create copy for visualization
    vis_img = img_array.copy()

    # Create a separate image for showing the thresholding
    height, width = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Visualize the different thresholding methods
    _, binary1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    _, binary2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)
    _, binary4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Stack the binary images side by side
    binary_row1 = np.hstack([binary1, binary2])
    binary_row2 = np.hstack([binary3, binary4])
    binary_vis = np.vstack([binary_row1, binary_row2])

    # Resize to fit a reasonable space
    scale = min(1.0, 400 / binary_vis.shape[1])
    binary_vis = cv2.resize(binary_vis, (0, 0), fx=scale, fy=scale)

    # Detect main subject
    center_x, center_y = detect_main_subject(img_array)

    # Draw crosshair on the original image
    cv2.line(vis_img, (center_x, 0), (center_x, height), (255, 0, 0), 2)
    cv2.line(vis_img, (0, center_y), (width, center_y), (255, 0, 0), 2)

    # Draw a circle at the center point
    cv2.circle(vis_img, (center_x, center_y), 10, (0, 255, 0), -1)

    # Convert back to PIL
    vis_pil = Image.fromarray(vis_img)
    binary_pil = Image.fromarray(binary_vis)

    return vis_pil, binary_pil


def center_product_image(img, target_size=(800, 800), bg_color="#FFFFFF", x_offset=0, y_offset=0,
                         add_padding=True, padding_percent=5):
    """Process product images with proper centering and tight cropping."""
    # Convert PIL Image to numpy array for OpenCV processing
    img_array = np.array(img)
    orig_height, orig_width = img_array.shape[:2]

    # Create a copy for processing
    processed = img_array.copy()

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Threshold to identify background (using fixed threshold)
    bg_threshold = 230
    _, mask = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY)
    
    # Invert mask to get foreground
    mask_inv = cv2.bitwise_not(mask)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)

    # Use the mask to find the bounding box of the foreground
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get its bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add small margin (5% of the object size)
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)

        # Ensure margins don't go out of bounds
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(orig_width - x, w + 2 * margin_x)
        h = min(orig_height - y, h + 2 * margin_y)

        # Crop to the object plus margin
        processed = processed[y:y + h, x:x + w]

    # Convert processed numpy array back to PIL
    processed_pil = Image.fromarray(processed)

    # Apply consistent padding if requested
    if add_padding:
        # Calculate the maximum size based on target dimensions minus padding
        padding_factor = padding_percent / 100
        max_content_width = int(target_size[0] * (1 - 2 * padding_factor))
        max_content_height = int(target_size[1] * (1 - 2 * padding_factor))

        # Calculate scaling factor for the padded image
        proc_width, proc_height = processed_pil.size
        scale = min(max_content_width / proc_width, max_content_height / proc_height)
        new_width = int(proc_width * scale)
        new_height = int(proc_height * scale)
    else:
        # Use the full target size
        proc_width, proc_height = processed_pil.size
        scale = min(target_size[0] / proc_width, target_size[1] / proc_height)
        new_width = int(proc_width * scale)
        new_height = int(proc_height * scale)

    # Resize image while maintaining aspect ratio
    img_resized = processed_pil.resize((new_width, new_height), Image.LANCZOS)

    # Convert hex color to RGB
    bg_color = bg_color.lstrip('#')
    bg_rgb = tuple(int(bg_color[i:i + 2], 16) for i in (0, 2, 4))

    # Create a new image with the target size (always RGB)
    new_img = Image.new("RGB", target_size, bg_rgb)

    # Calculate position to paste the resized image centered
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2

    # Apply manual offsets
    paste_x += x_offset
    paste_y += y_offset

    # Paste the image
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


def save_image(img, output_format="JPEG"):
    """Save the image to a temporary file and return the path."""
    if output_format == "WEBP":
        suffix = '.webp'
    else:
        suffix = '.jpg'
        output_format = "JPEG"  # Ensure correct format name

    temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
    img.save(temp_file.name, format=output_format)
    return temp_file.name


# Streamlit app
st.set_page_config(page_title="eCommerce Image Centering Tool | By Lee Foot")
st.title("eCommerce Image Centering Tool")
st.markdown("""
<div style="margin-bottom: 20px;">
    <h6>
        Built with Streamlit & OpenCV | 
        <a href="https://leefoot.co.uk" target="_blank">By Lee Foot</a> | 
        <a href="https://x.com/LeeFootSEO/" target="_blank">Twitter/X</a> | 
        <a href="mailto:hello@leefoot.co.uk">Hire Me</a>
    </h6>
</div>
""", unsafe_allow_html=True)
st.write("Upload images to center the main subject for consistent product displays")

# Sidebar settings
st.sidebar.header("Settings")

# Padding options
st.sidebar.subheader("Padding")
add_padding = st.sidebar.checkbox("Add whitespace padding", value=True,
                                help="Adds consistent padding around products")
padding_percentage = st.sidebar.slider("Padding amount (%)",
                                     0, 30, 5, 5,
                                     help="Percentage of canvas to use as padding")

# Manual adjustment options
st.sidebar.subheader("Manual Adjustment")
enable_manual_adjustment = st.sidebar.checkbox("Enable manual adjustment", value=False)
manual_x_offset = 0
manual_y_offset = 0

if enable_manual_adjustment:
    manual_x_offset = st.sidebar.slider("Horizontal adjustment", -100, 100, 0, 5,
                                      help="Adjust position left (-) or right (+)")
    manual_y_offset = st.sidebar.slider("Vertical adjustment", -100, 100, 0, 5,
                                      help="Adjust position up (-) or down (+)")

# Output size options
st.sidebar.subheader("Output Size")
output_size = st.sidebar.selectbox(
    "Dimensions",
    [
        "600×600 (Small)",
        "800×800 (Medium)",
        "1000×1000 (Large)",
        "Custom"
    ],
    index=0
)

if output_size == "Custom":
    custom_width = st.sidebar.number_input("Width", min_value=100, max_value=2000, value=600,
                                         step=50)
    custom_height = st.sidebar.number_input("Height", min_value=100, max_value=2000, value=600,
                                          step=50)
    target_size = (custom_width, custom_height)
else:
    size_map = {
        "800×800 (Medium)": (800, 800),
        "1000×1000 (Large)": (1000, 1000),
        "600×600 (Small)": (600, 600)
    }
    target_size = size_map[output_size]

# Background color
padding_color = st.sidebar.color_picker(
    "Background Color",
    "#FFFFFF",
    help="Choose the color for padding around resized images"
)

# File uploader
uploaded_files = st.file_uploader("Upload product images", type=["jpg", "jpeg", "png", "webp"],
                                accept_multiple_files=True)

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} images...")

    processed_images = []

    for uploaded_file in uploaded_files:
        # Read image
        img = Image.open(uploaded_file).convert("RGB")

        # Process image with simplified parameters
        centered_img = center_product_image(img, target_size, padding_color,
                                          manual_x_offset, manual_y_offset,
                                          add_padding, padding_percentage)

        # Save processed image
        output_path = save_image(centered_img)
        processed_images.append((centered_img, output_path, uploaded_file.name))

    # Display results with debugging
    st.subheader("Results")

    st.checkbox("Show detection visualization", key="show_detection")
    show_detection = st.session_state.show_detection

    cols = st.columns(min(3, len(processed_images)))

    for i, (img, path, original_name) in enumerate(processed_images):
        with cols[i % len(cols)]:
            if show_detection:
                # Show the detection visualization
                detection_vis, binary_vis = enhance_detection_visualization(Image.open(uploaded_files[i]),
                                                                          original_name)
                st.image(detection_vis, caption=f"Detection: {original_name}", use_container_width=True)
                st.image(binary_vis, caption="Threshold visualization", use_container_width=True)

            st.image(img, caption=f"Centered: {original_name}", use_container_width=True)

            # Add format options
            format_options = st.radio(f"Format for {original_name}", ["JPEG", "WEBP"], horizontal=True)

            # Save in selected format
            download_path = save_image(img, output_format=format_options)

            # Determine correct mime type and file extension
            mime_type = "image/jpeg" if format_options == "JPEG" else "image/webp"
            file_ext = "jpg" if format_options == "JPEG" else "webp"

            # Create download button
            with open(download_path, "rb") as file:
                btn = st.download_button(
                    label=f"Download {original_name}",
                    data=file,
                    file_name=f"centered_{original_name.split('.')[0]}.{file_ext}",
                    mime=mime_type
                )

            # Clean up temporary file
            try:
                os.unlink(download_path)
            except:
                pass

    # Batch download option
    if len(processed_images) > 1:
        st.subheader("Batch Download")

        # Create batch download option with format selection
        import zipfile

        # Format options for batch download
        batch_format = st.radio("Batch download format:", ["JPEG", "WEBP"], horizontal=True)

        zip_file = NamedTemporaryFile(delete=False, suffix='.zip')
        file_ext = "jpg" if batch_format == "JPEG" else "webp"

        with zipfile.ZipFile(zip_file.name, 'w') as zipf:
            for img, _, original_name in processed_images:
                # Save in selected format
                temp_img_path = save_image(img, output_format=batch_format)
                base_name = original_name.split('.')[0]
                zipf.write(temp_img_path, arcname=f"centered_{base_name}.{file_ext}")
                # Clean up temp image
                try:
                    os.unlink(temp_img_path)
                except:
                    pass

        with open(zip_file.name, "rb") as file:
            st.download_button(
                label=f"Download All Centered Images ({batch_format})",
                data=file,
                file_name=f"ecommerce_centered_images_{batch_format.lower()}.zip",
                mime="application/zip"
            )

        # Clean up temporary files
        for _, path, _ in processed_images:
            try:
                os.unlink(path)
            except:
                pass

else:
    st.info("Please upload images to begin processing")

    # Example image
    st.subheader("How it works")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Original Product Image**")
        st.image("https://via.placeholder.com/400x400.png?text=Original+Product+Image", use_container_width=True)
    with cols[1]:
        st.markdown("**Centered Product Image**")
        st.image("https://via.placeholder.com/400x400.png?text=Centered+Product+Image", use_container_width=True)

    st.markdown("""
    1. Upload your product images
    2. The app detects the main subject in each image
    3. Images are centered around the main subject
    4. Download individually or in batch
    5. Upload to WooCommerce for consistent product displays
    """)

# Footer
st.markdown("---")
st.markdown(
    "eCommerce Image Centering Tool | By [Lee Foot](https://leefoot.co.uk) | [Twitter/X](https://x.com/LeeFootSEO/) | [Hire Me](mailto:hello@leefoot.co.uk)")
