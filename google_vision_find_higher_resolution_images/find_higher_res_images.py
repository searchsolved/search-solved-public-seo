import os
import pandas as pd
from PIL import Image
import PIL
from PIL import Image, UnidentifiedImageError

import io
import requests
import concurrent.futures
import logging
from typing import List
from urllib.parse import urlparse
from google.cloud import vision
from tqdm import tqdm
import urllib3.exceptions
import google.api_core.exceptions
from urllib3.exceptions import MaxRetryError
import imghdr


# read in the datafile of image urls
df = pd.read_csv('/python_scripts/google_vision/input_file/wc_images.csv')
df = df[:1000]
header = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/87.0.4280.88 Mobile Safari/537.36'
}

# read in the json secrets file and instantiate google vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/python_scripts/cloud_vision_api.json"
client = vision.ImageAnnotatorClient()

# set up logging
logging.basicConfig(filename='image_scraper.log', level=logging.ERROR)


def get_image_size(url: str) -> tuple:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, stream=True, timeout=20)
        if r.status_code == 200:
            # use imghdr to determine the type of the image
            img_type = imghdr.what(None, h=r.content)
            if img_type is None:
                raise ValueError('Unknown image type')

            # parse the image format's header to get the size
            with Image.open(io.BytesIO(r.content)) as img:
                return img.size
        else:
            logging.error(f'Error retrieving image size for {url}: HTTP status code {r.status_code}')
            return _get_image_size_content_length(url)
    except Exception as e:
        logging.error(f'Error retrieving image size for {url}: {e}')
        return (0, 0)

def _get_image_size_content_length(url: str) -> tuple:
    try:
        r = requests.head(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        if 'content-length' in r.headers:
            # return the content length as the size
            return (int(r.headers['content-length']), int(r.headers['content-length']))
    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f'Error retrieving image size for {url}: {e}')
    return (0, 0)


# set up a dictionary to cache the results of Google Cloud Vision API calls
url_cache = {}

# define a function to retrieve image URLs from Google Vision API
def get_image_urls(url: str) -> List[str]:
    try:
        image = vision.Image(source=vision.ImageSource(image_uri=url))
        response = client.web_detection(image=image).web_detection
        urls = [img.url for img in response.full_matching_images]
        return urls
    except (google.api_core.exceptions.GoogleAPIError, urllib3.exceptions.MaxRetryError) as e:
        logging.error(f'Error retrieving image URLs for {url}: {e}')
        return []

# define a function to process a single URL
def process_url(url: str) -> pd.DataFrame:
    result = {'original_url': url, 'matching_imgs': [], 'width_diff': [], 'height_diff': []}
    try:
        og_size = get_image_size(url)
        og_width, og_height = Image.open(requests.get(url, headers=header, timeout=20, stream=True).raw).size
        img_urls = get_image_urls(url)
        with tqdm(desc=f"Processing {len(img_urls)} image URLs for {url}", total=len(img_urls)) as pbar:
            for img_url in img_urls:
                try:
                    img_size = get_image_size(img_url)
                    if img_size > og_size:
                        img_width, img_height = Image.open(requests.get(img_url, headers=header, timeout=20, stream=True).raw).size
                        result['matching_imgs'].append(img_url)
                        result['width_diff'].append(img_width - og_width)
                        result['height_diff'].append(img_height - og_height)
                except (requests.exceptions.SSLError, urllib3.exceptions.MaxRetryError, urllib3.exceptions.SSLError, urllib3.exceptions.ReadTimeoutError, PIL.UnidentifiedImageError) as e:
                    logging.error(f'Error processing image URL {img_url} for {url}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error processing image URL {img_url} for {url}: {e}')
                pbar.update(1)
    except (google.api_core.exceptions.GoogleAPIError, urllib3.exceptions.MaxRetryError, urllib3.exceptions.ReadTimeoutError) as e:
        logging.error(f'Error processing {url}: {e}')
    except Exception as e:
        logging.error(f'Unexpected error processing {url}: {e}')
    return pd.DataFrame(result)

# process the URLs in parallel using a thread pool
results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_url, url) for url in df['Images']]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing URLs"):
        results.append(future.result())

# concatenate the results into a single DataFrame
df = pd.concat(results)
df.to_csv("/python_scripts/higher_resolution_images.csv")
