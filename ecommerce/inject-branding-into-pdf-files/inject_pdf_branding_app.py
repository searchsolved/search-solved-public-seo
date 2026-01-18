"""
PDF Branding Injector - Streamlit App

Adds branded headers/footers to PDF files with customizable colors and fonts.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import zipfile
from io import BytesIO
import os

try:
    from PyPDF2 import PdfReader, PdfWriter, PageObject
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas
except ImportError:
    st.error("Please install required packages: pip install PyPDF2 reportlab")
    st.stop()

st.set_page_config(
    page_title="PDF Branding Injector",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Branding Injector")
st.markdown("Add branded headers to your PDF files with custom styling.")


def calculate_text_size(page_width, base_width=612, min_size=10, max_size=14, default_size=12):
    """Calculate appropriate text size based on page width."""
    if abs(page_width - base_width) < 10:
        return default_size
    scale_factor = page_width / base_width
    return max(min_size, min(max_size, default_size * scale_factor))


def create_header_canvas(width, height, space_height, phone_number, website,
                         text_font, default_text_size, text_color, background_color):
    """Create a header canvas with branding information."""
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))

    scaled_text_size = calculate_text_size(width, default_size=default_text_size)

    # Draw the background color for the header
    can.setFillColor(background_color)
    can.rect(0, height - space_height, width, space_height, fill=1, stroke=0)

    # Set text properties
    can.setFont(text_font, scaled_text_size)
    can.setFillColor(text_color)

    # Combine phone number and website and position them
    combined_text = f"{website} - {phone_number}"
    text_width = pdfmetrics.stringWidth(combined_text, text_font, scaled_text_size)
    combined_text_x = (width - text_width) / 2

    baseline_adjustment = scaled_text_size * 0.2
    combined_text_y = height - space_height / 2 - scaled_text_size / 2 + baseline_adjustment

    can.drawString(combined_text_x, combined_text_y, combined_text)

    can.save()
    packet.seek(0)
    return packet


def merge_pdf_pages(existing_pdf, new_header, space_height):
    """Merge existing PDF pages with a new header."""
    output = PdfWriter()
    new_pdf = PdfReader(new_header)

    # Process the first page to add the header
    first_page = existing_pdf.pages[0]
    new_page_width = float(first_page.mediabox.right)
    new_page_height = float(first_page.mediabox.top) + space_height

    new_first_page = PageObject.create_blank_page(
        width=first_page.mediabox.right,
        height=first_page.mediabox.top + space_height
    )
    new_first_page.merge_page(new_pdf.pages[0])
    new_first_page.merge_page(first_page, (0, space_height))
    output.add_page(new_first_page)

    # Add remaining pages as-is
    for page_num in range(1, len(existing_pdf.pages)):
        output.add_page(existing_pdf.pages[page_num])

    return output


def add_branding_to_pdf(pdf_bytes, phone_number, website, space_height,
                         text_font, text_font_size, text_color, background_color):
    """Add branding header to a PDF file."""
    try:
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        first_page = pdf_reader.pages[0]
        page_width = float(first_page.mediabox.right)
        page_height = float(first_page.mediabox.top) + space_height

        header_canvas = create_header_canvas(
            page_width, page_height, space_height,
            phone_number, website, text_font, text_font_size,
            text_color, background_color
        )

        output_pdf = merge_pdf_pages(pdf_reader, header_canvas, space_height)

        output_bytes = BytesIO()
        output_pdf.write(output_bytes)
        output_bytes.seek(0)

        return output_bytes.getvalue(), None
    except Exception as e:
        return None, str(e)


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Branding Settings")

    website = st.text_input(
        "Website URL",
        value="www.example.com",
        help="Your website URL to display in the header"
    )

    phone_number = st.text_input(
        "Phone Number",
        value="01234 567 890",
        help="Phone number to display in the header"
    )

    st.markdown("---")
    st.subheader("🎨 Styling")

    col1, col2 = st.columns(2)
    with col1:
        bg_color = st.color_picker(
            "Background Color",
            value="#D91800",
            help="Header background color"
        )
    with col2:
        text_color = st.color_picker(
            "Text Color",
            value="#FFFFFF",
            help="Header text color"
        )

    space_height = st.slider(
        "Header Height (px)",
        min_value=20,
        max_value=60,
        value=30,
        help="Height of the branded header"
    )

    text_font_size = st.slider(
        "Font Size",
        min_value=8,
        max_value=20,
        value=14,
        help="Size of the header text"
    )

    text_font = st.selectbox(
        "Font",
        ["Helvetica", "Helvetica-Bold", "Times-Roman", "Times-Bold", "Courier"],
        help="Font for header text"
    )

# Main content area
st.markdown("### 📤 Upload PDFs")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files to add branding"
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} PDF(s) ready for processing")

    # Preview section
    with st.expander("📋 Preview Uploaded Files"):
        for f in uploaded_files:
            st.markdown(f"- **{f.name}** ({f.size / 1024:.1f} KB)")

    # Header preview
    st.markdown("### 🔍 Header Preview")
    preview_col1, preview_col2 = st.columns([2, 1])

    with preview_col1:
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; color: {text_color};
                        padding: 10px; text-align: center; font-size: {text_font_size}px;
                        font-family: {text_font}; border-radius: 4px;">
                {website} - {phone_number}
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_pdfs = []
        errors = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text(f"Processing {i + 1}/{len(uploaded_files)}: {uploaded_file.name}")

            pdf_bytes = uploaded_file.read()
            result, error = add_branding_to_pdf(
                pdf_bytes,
                phone_number,
                website,
                space_height,
                text_font,
                text_font_size,
                HexColor(text_color),
                HexColor(bg_color)
            )

            if result:
                processed_pdfs.append((uploaded_file.name, result))
            else:
                errors.append((uploaded_file.name, error))

        progress_bar.empty()
        status_text.empty()

        if processed_pdfs:
            st.success(f"✅ Successfully processed {len(processed_pdfs)} PDF(s)")

            if len(processed_pdfs) == 1:
                # Single file download
                filename = f"branded_{processed_pdfs[0][0]}"
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=processed_pdfs[0][1],
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                # Multiple files - create ZIP
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, pdf_data in processed_pdfs:
                        zip_file.writestr(f"branded_{filename}", pdf_data)

                zip_buffer.seek(0)

                st.download_button(
                    label=f"📥 Download All ({len(processed_pdfs)} PDFs) as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="branded_pdfs.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        if errors:
            st.warning(f"⚠️ {len(errors)} file(s) failed to process")
            with st.expander("View Errors"):
                for filename, error in errors:
                    st.error(f"**{filename}**: {error}")

else:
    st.info("👆 Upload one or more PDF files to add branded headers")

    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        1. **Upload** your PDF files
        2. **Customize** the header with your branding:
           - Website URL and phone number
           - Background and text colors
           - Header height and font size
        3. **Preview** how your header will look
        4. **Process** and download your branded PDFs

        The tool adds a colored header bar to the top of each PDF's first page
        containing your website and contact information.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
