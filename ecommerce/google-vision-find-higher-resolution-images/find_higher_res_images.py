####################################################################################
# Website  : https://leefoot.co.uk/                                                #
# Contact  : https://leefoot.co.uk/hire-me/                                        #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://twitter.com/LeeFootSEO                                        #
####################################################################################

# Standard libraries
import json
import logging
import os
import hashlib
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

# External libraries
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout
from PIL import Image, UnidentifiedImageError
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from google.cloud import vision
from pyppeteer import launch

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
GOOGLE_APPLICATION_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_PATH',
                                                "/python_scripts/cloud_vision_api.json")
INPUT_FILE_PATH = '/python_scripts/google_vision/internal_images.csv'
OUTPUT_FILE_PATH = "/python_scripts/google_vision/higher_resolution_images_all.csv"
REQUEST_TIMEOUT = 20  # seconds for request timeout
MIN_RESOLUTION = (1000, 1000)  # Skip source images larger than
MAX_IMAGES = 5  # Maximum number of image suggestions in the output
SKIPPED_FILE_TYPES = ['.svg', '.gif']  # File types to skip

# Initialize a session for persistent requests
session = requests.Session()  # Initialize a session for persistent requests

# Global variables to track processed images
processed_hashes = set()
processed_results = {}
duplicates = {}

stats = {'skipped_format': 0, 'skipped_high_res': 0, 'skipped_duplicate': 0, 'processed': 0}


# ====== Helper Functions ====== #

def initialize_client():
    """
    Initializes the Google Vision client and returns the client and image objects.
    """
    logging.info("Initializing Google Vision client...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS_PATH
    return vision.ImageAnnotatorClient(), vision.Image()


def get_image_hash(image_data):
    """
    Generate a hash for the given image data.

    :param image_data: The data of the image.
    :return: The MD5 hash of the image data.
    """
    return hashlib.md5(image_data).hexdigest()


def is_image_processed(image_hash):
    """
    Check if an image hash is in the set of processed hashes.

    :param image_hash: The hash of the image.
    :return: True if the image has been processed, False otherwise.
    """
    return image_hash in processed_hashes


def mark_image_processed(image_hash, data, is_duplicate=False):
    """
    Mark an image hash as processed by adding it to the set and storing its data.

    :param image_hash: The hash of the image.
    :param data: The data to store for the image.
    :param is_duplicate: Indicates whether the image is a duplicate.
    """
    if not is_duplicate:
        processed_hashes.add(image_hash)
    processed_results[image_hash] = data


def handle_403_error(url):
    """
    Handle 403 error by attempting to fetch the image with Pyppeteer.

    :param url: The URL of the image.
    :return: The dimensions and bytes of the image if successful, (0, 0) and None otherwise.
    """
    logging.debug(f"Handling 403 error with Pyppeteer for URL {url}")
    logging.warning(f"Received 403 error for URL {url}, attempting fallback with Pyppeteer.")
    try:
        image_content = asyncio.get_event_loop().run_until_complete(fetch_image_with_pyppeteer(url))
        dimensions, image_bytes = extract_image_dimensions(image_content, url, fallback=True)
        return dimensions, image_bytes
    except Exception as e:
        logging.error(f"Pyppeteer also failed for URL {url}: {e}")
        return (0, 0), None  # Return (0, 0) to indicate failure


def extract_image_dimensions(image_content, url, fallback=False):
    """
    Extract image dimensions from the image content.

    :param image_content: The content of the image.
    :param url: The URL of the image.
    :param fallback: Indicates whether this is a fallback attempt.
    :return: The dimensions and bytes of the image.
    """
    im = Image.open(io.BytesIO(image_content) if fallback else image_content)
    dimensions = im.size
    image_bytes = im.tobytes()
    logging.debug(f"{'Fallback successful, ' if fallback else ''}Fetched image dimensions for URL {url}: {dimensions}")
    return dimensions, image_bytes


def check_file_type(url):
    """
    Check and skip unwanted file types.

    :param url: The URL of the image.
    :return: True if the file type should be skipped, False otherwise.
    """
    if any(url.endswith(ext) for ext in SKIPPED_FILE_TYPES):
        logging.info(f"Skipping URL due to file type: {url}")
        stats['skipped_format'] += 1
        return True
    return False


# ====== Core Processing Functions ====== #

def fetch_image_with_requests(url, user_agent):
    """
    Fetch the image using the requests library.

    :param url: The URL of the image.
    :param user_agent: The user agent to use for the request.
    :return: The response from the request.
    """
    logging.debug(f"Fetching image with requests for URL {url}")
    headers = {'User-Agent': user_agent.random}
    response = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers, stream=True)
    return response


async def fetch_image_with_pyppeteer(url):
    """
    Fetch the image using Pyppeteer (headless browser).

    :param url: The URL of the image.
    :return: The content of the image.
    """
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto(url)
    image_content = await page.screenshot()  # Screenshot for images, or page.content() for HTML
    await browser.close()
    return image_content


def fetch_image_with_selenium(url):
    """
    Fetch the image using Selenium (headless browser).

    :param url: The URL of the image.
    :return: The dimensions and bytes of the image.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        image_element = driver.find_element(By.TAG_NAME, 'img')  # Change this line
        image_url = image_element.get_attribute('src')

        if image_url.startswith('data:image'):
            # Convert base64 image to bytes
            base64_encoded_data = image_url.split(',')[1]
            image_bytes = base64.b64decode(base64_encoded_data)
        else:
            # Fetch image normally if it's not a data URL
            image_response = requests.get(image_url, stream=True)
            image_bytes = image_response.raw.read()

        # Convert bytes to PIL Image to get dimensions
        image = Image.open(io.BytesIO(image_bytes))
        dimensions = image.size

        logging.debug(f"Fetched image dimensions with Selenium for URL {url}: {dimensions}")
        return dimensions, image_bytes

    finally:
        driver.quit()


def fetch_image_dimensions(url, user_agent):
    """
    Fetch the dimensions of the image.

    :param url: The URL of the image.
    :param user_agent: The user agent to use for the request.
    :return: The dimensions and bytes of the image.
    """
    try:
        response = fetch_image_with_requests(url, user_agent)

        if response.status_code == 403:
            logging.info(f"Fetching image with Pyppeteer for URL {url}")
            dimensions, image_bytes = handle_403_error(url)

            # Check if Pyppeteer also returned 403, then use Selenium
            if dimensions == (0, 0):
                logging.info(f"Fetching image with Selenium as second fallback for URL {url}")
                dimensions, image_bytes = fetch_image_with_selenium(url)

            return dimensions, image_bytes

        logging.info(f"Fetching image with requests for URL {url}")
        dimensions, image_bytes = extract_image_dimensions(response.raw, url)
        return dimensions, image_bytes
    except (RequestException, UnidentifiedImageError, Timeout) as e:
        logging.error(f"Exception in fetch_image_dimensions for URL {url}: {e}")
        return (0, 0), None


def fetch_matching_images(url, client, image):
    """
    Fetches matching images for a given URL using the Google Vision API.

    :param url: The URL of the image.
    :param client: The Google Vision API client.
    :param image: The Google Vision API image object.
    :return: A list of matching images.
    """
    try:
        logging.info(f"Fetching matching images for URL: {url}")
        image.source.image_uri = url
        web_response = client.web_detection(image=image)
        web_content = web_response.web_detection
        json_string = type(web_content).to_json(web_content)
        data = json.loads(json_string)
        return data.get('fullMatchingImages', [])
    except Exception as e:
        logging.error(f"Error fetching matching images for URL {url}: {e}")
        return []


def process_matching_images(url, matching_images, original_image_dimensions, user_agent):
    """
    Process each matching image URL to get dimensions and bytes, and check if the image has been processed.
    If not, the image is marked as processed, and its details are appended to the results list.

    Args:
        url (str): The URL of the original image.
        matching_images (list): List of matching images obtained from the Google Vision API.
        original_image_dimensions (tuple): Dimensions of the original image.
        user_agent (UserAgent): UserAgent object to randomize the user agent string for requests.

    Returns:
        list: A list of tuples. Each tuple contains the original URL, matching image URL,
              dimensions of the matching image, and dimensions of the original image.
    """
    logging.debug(f"Processing matching images for URL {url}")
    results = []
    for matching_image in matching_images:
        matching_image_url = matching_image['url']
        (matching_image_dimensions, matching_image_bytes) = fetch_image_dimensions(matching_image_url, user_agent)
        logging.debug(f"Matching image dimensions for URL {matching_image_url}: {matching_image_dimensions}")
        matching_image_hash = get_image_hash(matching_image_bytes)

        if not is_image_processed(matching_image_hash):
            mark_image_processed(matching_image_hash, [
                (url, matching_image_url, *matching_image_dimensions, *original_image_dimensions)])
            results.append((url, matching_image_url, *matching_image_dimensions, *original_image_dimensions))
            stats['processed'] += 1
    return results


def fetch_and_process_image(url, client, image, user_agent):
    """
    Process an image URL by first validating the file type and checking if the image has been processed.
    If not, it fetches matching images, processes them, and returns the results.

    Args:
        url (str): The URL of the image to process.
        client (vision.ImageAnnotatorClient): Google Vision client.
        image (vision.Image): Google Vision image object.
        user_agent (UserAgent): UserAgent object to randomize the user agent string for requests.

    Returns:
        list: A list of processed image data or an empty list if no processing was done.
    """
    try:
        if validate_image_file_type(url):
            return []

        original_image_dimensions, original_image_bytes = fetch_and_validate_image(url, user_agent)
        if original_image_dimensions is None:
            return []

        original_image_hash = get_image_hash(original_image_bytes)
        if is_image_processed(original_image_hash):
            return handle_processed_image(url, original_image_hash)

        matching_images = fetch_matching_images(url, client, image)
        return process_image(url, matching_images, original_image_dimensions, original_image_hash, user_agent)

    except Exception as e:
        logging.error(f"Unexpected error processing URL {url}: {e}")
        return []


def validate_image_file_type(url):
    """
    Check if the file type of the URL is valid (not in the SKIPPED_FILE_TYPES list).

    Args:
        url (str): The URL of the image to check.

    Returns:
        bool: True if the file type is valid, False otherwise.
    """
    if check_file_type(url):
        stats['skipped_format'] += 1
        return True
    return False


def fetch_and_validate_image(url, user_agent):
    """
    Fetch the image from the given URL and validate if the image meets the resolution criteria.
    If the image is already high-resolution, it's skipped. Otherwise, it's marked for processing.

    Args:
        url (str): The URL of the image to fetch and validate.
        user_agent (UserAgent): UserAgent object to randomize the user agent string for requests.

    Returns:
        tuple or None: Tuple containing the dimensions and bytes of the image if valid,
                       None otherwise.
    """
    original_image_dimensions, original_image_bytes = fetch_image_dimensions(url, user_agent)
    logging.debug(f"Original image dimensions for URL {url}: {original_image_dimensions}")

    if original_image_dimensions[0] >= MIN_RESOLUTION[0] and original_image_dimensions[1] >= MIN_RESOLUTION[1]:
        log_and_update_stats(f"Image dimensions are {original_image_dimensions}, Already High Res! Skipping!",
                             'skipped_high_res')
        return None, None

    if original_image_dimensions[0] < MIN_RESOLUTION[0] or original_image_dimensions[1] < MIN_RESOLUTION[1]:
        log_and_update_stats(f"Processing low-resolution image: {url}", 'processed_low_res')

    return original_image_dimensions, original_image_bytes


def handle_processed_image(url, original_image_hash):
    """
    Handle a processed image by marking it as a duplicate and recording it.

    Args:
        url (str): The URL of the image that has been processed.
        original_image_hash (str): The hash of the image that has been processed.

    Returns:
        list: An empty list indicating no further processing is required for this image.
    """
    logging.info(f"Original image {url} has already been processed. Recording duplicate.")
    stats['skipped_duplicate'] += 1
    duplicates[original_image_hash] = url
    return []


def process_image(url, matching_images, original_image_dimensions, original_image_hash, user_agent):
    """
    Process the image by calling process_matching_images, mark the image as processed, and return the results.

    Args:
        url (str): The URL of the image to process.
        matching_images (list): List of matching images obtained from the Google Vision API.
        original_image_dimensions (tuple): Dimensions of the original image.
        original_image_hash (str): The hash of the original image.
        user_agent (UserAgent): UserAgent object to randomize the user agent string for requests.

    Returns:
        list: A list of processed image data.
    """
    results = process_matching_images(url, matching_images, original_image_dimensions, user_agent)
    if results:
        mark_image_processed(original_image_hash, results)
    else:
        mark_image_processed(original_image_hash, [], is_duplicate=False)
    stats['processed'] += len(results)
    return results


def log_and_update_stats(message, stat_key):
    """
    Log a message and update the corresponding stat in the stats dictionary.

    Args:
        message (str): The message to log.
        stat_key (str): The key in the stats dictionary to update.
    """
    logging.info(message)
    stats[stat_key] = stats.get(stat_key, 0) + 1


def process_images(df, client, image, user_agent):
    """
    Process each image URL in the dataframe concurrently.

    Args:
        df (DataFrame): The dataframe containing image URLs to process.
        client (vision.ImageAnnotatorClient): Google Vision client.
        image (vision.Image): Google Vision image object.
        user_agent (UserAgent): UserAgent object to randomize the user agent string for requests.

    Returns:
        list: A list of results from processing the images.
    """
    results = []
    logging.info(f"Processing {len(df['Address'])} images concurrently...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Create a future-to-url mapping
        future_to_url = {executor.submit(fetch_and_process_image, url, client, image, user_agent): url for url in
                         df['Address']}

        # Iterate through the completed futures
        for i, future in enumerate(as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                result = future.result()
                results.extend(result)
                logging.info(f"Processed {i + 1}/{len(future_to_url)} images.")
            except Exception as e:
                logging.error(f"Error fetching/processing URL {url}: {e}")

    return results


# ====== DataFrame Handling Functions ====== #

def create_initial_dataframe(results):
    """
    Create an initial pandas DataFrame from the results.

    Args:
        results (list): A list of tuples containing image information.

    Returns:
        DataFrame: A pandas DataFrame with columns for original_url, matching_imgs,
                   width_matching_imgs, height_matching_imgs, width_source_img, and height_source_img.
    """
    df = pd.DataFrame(results, columns=['original_url', 'matching_imgs', 'width_matching_imgs',
                                        'height_matching_imgs', 'width_source_img', 'height_source_img'])
    return df


def calculate_differences(df):
    """
    Calculate the differences in width and height between the matching image and the source image.

    Args:
        df (DataFrame): The DataFrame containing information about the images.

    Returns:
        DataFrame: The input DataFrame with two new columns, 'width_diff' and 'height_diff', representing the
                   differences in width and height, respectively.
    """
    df['width_diff'] = df['width_matching_imgs'] - df['width_source_img']
    df['height_diff'] = df['height_matching_imgs'] - df['height_source_img']
    return df


def filter_and_sort_results(df):
    """
    Filter out unwanted results based on width and height differences, remove duplicates,
    and sort the DataFrame by width and height differences in descending order.

    Args:
        df (DataFrame): The DataFrame containing image comparison results.

    Returns:
        DataFrame: The filtered and sorted DataFrame.
    """
    df = df.loc[~((df['width_diff'] <= 0) | (df['height_diff'] <= 0))]
    df.drop_duplicates(subset=["width_diff", "height_diff"], keep="first", inplace=True)
    df.sort_values(["width_diff", "height_diff"], ascending=[False, False], inplace=True)
    return df


def select_top_results(df):
    """
    Select the top results for each original URL based on the MAX_IMAGES constant.

    Args:
        df (DataFrame): The DataFrame containing the filtered and sorted image comparison results.

    Returns:
        DataFrame: A DataFrame containing the top results for each original URL.
    """
    df = df.groupby(['original_url']).head(MAX_IMAGES)
    return df[['original_url', 'matching_imgs', 'width_diff', 'height_diff',
               'width_matching_imgs', 'height_matching_imgs', 'width_source_img', 'height_source_img']]


def create_dataframes(results):
    """
    Creates and organizes the final DataFrame based on the processed image results.

    Args:
        results (list): A list of processed image results.

    Returns:
        DataFrame: The final organized DataFrame ready for output.
    """
    logging.info("Creating and organizing final dataframe...")
    df = create_initial_dataframe(results)
    df = calculate_differences(df)
    df = filter_and_sort_results(df)
    df = select_top_results(df)
    return df


# ====== Main Function ====== #

def main():
    """
    The main function of the script. It initializes the necessary clients, reads the input data,
    processes each image URL in the input, handles duplicate results, and writes the final output
    to a CSV file. It also logs the progress and final statistics of the processing.
    """
    logging.info("Starting the image processing script...")

    # Initialize Google Vision client and UserAgent for randomized user agent strings
    client, image = initialize_client()
    user_agent = UserAgent()

    # Read input data
    df = pd.read_csv(INPUT_FILE_PATH)

    # Process images and fetch results
    results = process_images(df, client, image, user_agent)

    # Handle duplicate results by assigning the results from original images to their duplicates
    for hash_value, duplicate_url in duplicates.items():
        if hash_value in processed_results:
            duplicate_results = processed_results[hash_value]
            for result in duplicate_results:
                # Update the URL to the duplicate's URL
                modified_result = (duplicate_url, *result[1:])
                results.append(modified_result)

    # Create final DataFrame from results, and write to CSV
    final_df = create_dataframes(results)
    final_df.to_csv(OUTPUT_FILE_PATH)

    # Log completion and statistics
    logging.info(f"Script completed. Output saved to {OUTPUT_FILE_PATH}")
    logging.info(
        f"Stats: Skipped due to format: {stats['skipped_format']}, Skipped due to high resolution: {stats['skipped_high_res']}, Skipped due to duplicate: {stats['skipped_duplicate']}, Processed: {stats['processed']}"
    )


if __name__ == "__main__":
    main()
