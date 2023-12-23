# Inject branding into PDFs by Lee Foot 23-12-2023. 
# More like this @ https://leefoot.co.uk

import os
from datetime import datetime
from io import BytesIO
import warnings
import traceback

from PyPDF2 import PdfReader, PdfWriter, PageObject
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from tqdm import tqdm

# Configuration variables

# Path to the custom font file (optional)
FONT_PATH = "/python_scripts/pdf_branding/custom_fonts/Roboto-Bold.ttf"

# Directory for input PDF files
INPUT_DIR = "/python_scripts/pdf_branding/input"

# Directory for output PDF files
OUTPUT_DIR = "/python_scripts/pdf_branding/output"

# Phone number to display in PDF header
PHONE_NUMBER = "01234 567 890"

# Website URL to display in PDF header
WEBSITE = "www.acme.com"

# Suffix to append to modified PDF filenames
SUFFIX = "_acme_"

# Height of header space in PDF
SPACE_HEIGHT = 30

# Font name for header text
TEXT_FONT = "Roboto-Bold"

# Font size for header text
TEXT_FONT_SIZE = 14

# Colour of the header text
TEXT_COLOR = HexColor("#FFFFFF")

# Background colour of the header
BACKGROUND_COLOR = HexColor("#D91800")

# Flag to control appending of suffix to filenames (set to True to enable)
APPEND_SUFFIX = False

# Flag to control appending of the current date to filenames (set to True to enable)
APPEND_DATE = False


# -----------------
# Utility Functions
# -----------------

def calculate_text_size(page_width, base_width=612, min_size=10, max_size=14, default_size=12):
    """
    Calculate an appropriate text size based on the width of a PDF page.

    Args:
        page_width (float): The width of the PDF page.
        base_width (float, optional): The base width to compare against, default is 612 (standard letter page width).
        min_size (int, optional): The minimum font size, default is 10.
        max_size (int, optional): The maximum font size, default is 14.
        default_size (int, optional): The default font size, default is 12.

    Returns:
        int: Calculated font size.
    """
    if abs(page_width - base_width) < 10:  # If the width is close to base width, use default size
        return default_size
    scale_factor = page_width / base_width
    return max(min_size, min(max_size, default_size * scale_factor))


def create_header_canvas(width, height, space_height, phone_number, website,
                         text_font, default_text_size, text_color,
                         background_color):
    """
    Create a header canvas with phone number and website information.

    Args:
        width (float): Width of the canvas.
        height (float): Height of the canvas.
        space_height (float): Height of the header space.
        phone_number (str): Phone number to display.
        website (str): Website URL to display.
        text_font (str): Font name for the text.
        default_text_size (int): Default text size.
        text_color (HexColor): Color of the text.
        background_color (HexColor): Background color of the header.

    Returns:
        BytesIO: A byte stream containing the rendered header.
    """
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))

    # Calculate scaled text size
    scaled_text_size = calculate_text_size(width, default_size=default_text_size)

    # Set text properties with scaled size
    can.setFont(text_font, scaled_text_size)
    can.setFillColor(text_color)

    # Draw the background color for the header
    can.setFillColor(background_color)
    can.rect(0, height - space_height, width, space_height, fill=1, stroke=0)

    can.setFillColor(text_color)

    # Combine phone number and website and position them
    combined_text = f"{website} - {phone_number}"
    text_width = pdfmetrics.stringWidth(combined_text, text_font, scaled_text_size)
    combined_text_x = (width - text_width) / 2  # Center the text horizontally

    # Adjust the y position to center the text vertically within space_height
    baseline_adjustment = scaled_text_size * 0.2  # Heuristic adjustment factor
    combined_text_y = height - space_height / 2 - scaled_text_size / 2 + baseline_adjustment

    can.drawString(combined_text_x, combined_text_y, combined_text)

    can.save()
    packet.seek(0)
    return packet


def merge_pdf_pages(existing_pdf, new_header, space_height):
    """
    Merge existing PDF pages with a new header.

    Args:
        existing_pdf (PdfReader): The original PDF file reader.
        new_header (BytesIO): Byte stream of the new header.
        space_height (float): Height of the header space.

    Returns:
        PdfWriter: PDF writer object with merged pages.
    """
    output = PdfWriter()
    new_pdf = PdfReader(new_header)

    # Process the first page separately to add the header
    first_page = existing_pdf.pages[0]
    new_first_page = PageObject.create_blank_page(width=first_page.mediabox.right,
                                                  height=first_page.mediabox.top + space_height)
    new_first_page.merge_page(new_pdf.pages[0])
    new_first_page.merge_page(first_page, (0, space_height))
    output.add_page(new_first_page)

    # For the rest of the pages, just add them as they are
    for page_num, page in enumerate(existing_pdf.pages):
        if page_num != 0:  # Skip the first page as it's already processed
            output.add_page(page)

    return output


# -----------------
# PDF Processing Functions
# -----------------

def add_section_to_pdf(input_pdf_path, output_dir, phone_number, website, suffix,
                       space_height, text_font, text_font_size, text_color, background_color):
    """
    Add a custom header section to a single PDF file.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory where the modified PDF will be saved.
        phone_number (str): Phone number to be included in the header.
        website (str): Website URL to be included in the header.
        suffix (str): Suffix to append to the output file's name.
        space_height (float): Height of the header space.
        text_font (str): Font name for the text in the header.
        text_font_size (int): Font size for the text in the header.
        text_color (HexColor): Color of the text in the header.
        background_color (HexColor): Background color of the header.
    """
    filename = os.path.basename(input_pdf_path)
    base_name, ext = os.path.splitext(filename)

    # Append suffix and date if enabled
    if APPEND_SUFFIX:
        base_name += suffix
    if APPEND_DATE:
        date_str = datetime.now().strftime("%Y%m%d")
        base_name += f"_{date_str}"

    output_filename = f"{base_name}{ext}"
    output_pdf_path = os.path.join(output_dir, output_filename)

    existing_pdf = PdfReader(input_pdf_path)
    first_page = existing_pdf.pages[0]
    new_page_width = float(first_page.mediabox.right)
    new_page_height = float(first_page.mediabox.top) + space_height

    header_canvas = create_header_canvas(new_page_width, new_page_height, space_height, phone_number, website,
                                         text_font, text_font_size, text_color, background_color)

    output_pdf = merge_pdf_pages(existing_pdf, header_canvas, space_height)

    with open(output_pdf_path, "wb") as f:
        output_pdf.write(f)


def process_directory(input_dir, output_dir, phone_number, website, suffix,
                      space_height, text_font, text_font_size, text_color, background_color):
    """
    Process all PDF files in the specified directory and its subdirectories, adding a custom header to each.

    Args:
        input_dir (str): Directory containing the original PDF files.
        output_dir (str): Directory where the modified PDF files will be saved.
        phone_number (str): Phone number to be included in the header of each PDF.
        website (str): Website URL to be included in the header of each PDF.
        suffix (str): Suffix to append to each output file's name.
        space_height (float): Height of the header space.
        text_font (str): Font name for the text in the header.
        text_font_size (int): Font size for the text in the header.
        text_color (HexColor): Color of the text in the header.
        background_color (HexColor): Background color of the header.
    """
    pdf_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append((root, file))

    with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
        for root, file in pdf_files:
            try:
                input_pdf_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_pdf_dir = os.path.join(output_dir, relative_path)

                if not os.path.exists(output_pdf_dir):
                    os.makedirs(output_pdf_dir)

                add_section_to_pdf(input_pdf_path, output_pdf_dir, phone_number, website, suffix,
                                   space_height, text_font, text_font_size, text_color, background_color)

            except Exception as e:
                warnings.warn(f"Error processing {file}: {e}")
                traceback.print_exc()  # Print full traceback for debugging

            finally:
                pbar.update(1)


# -----------------
# Main Script Execution
# -----------------

# Register custom font
try:
    pdfmetrics.registerFont(TTFont(TEXT_FONT, FONT_PATH))
except Exception as e:
    warnings.warn(f"Font file not found at {FONT_PATH}. Default font will be used. Error: {e}")

process_directory(INPUT_DIR, OUTPUT_DIR, PHONE_NUMBER, WEBSITE, SUFFIX,
                  SPACE_HEIGHT, TEXT_FONT, TEXT_FONT_SIZE, TEXT_COLOR, BACKGROUND_COLOR)
