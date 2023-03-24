import logging
import os
from typing import List
from tqdm import tqdm


import pandas as pd
import requests
from google.cloud import vision
from PIL import Image

header = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Mobile Safari/537.36'
}


# read in the json secrets file and instantiate google vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/python_scripts/cloud_vision_api.json"
client = vision.ImageAnnotatorClient()


# define a function to retrieve image URLs from Google Vision API
def get_image_urls(url: str) -> List[str]:
    try:
        image = vision.Image(source=vision.ImageSource(image_uri=url))
        response = client.web_detection(image=image).web_detection
        urls = [img.url for img in response.full_matching_images]
        return urls
    except Exception as e:
        logging.error(f'Error retrieving image URLs for {url}: {e}')
        return []


# define a function to get the size of an image from its URL
def get_image_size(url: str) -> tuple:
    try:
        im = Image.open(requests.get(url, headers=header, timeout=20, stream=True, verify=False).raw)
        return im.size
    except Exception as e:
        logging.error(f'Error opening {url}: {e}')
        return (0, 0)


# define a function to process a single URL
def process_url(url: str) -> pd.DataFrame:
    result = {'original_url': url, 'matching_imgs': [], 'width_diff': [], 'height_diff': []}
    try:
        og_image = Image.open(requests.get(url, headers=header, timeout=20, stream=True, verify=True).raw)
        og_width, og_height = og_image.size

        img_urls = []
        image = vision.Image(source=vision.ImageSource(image_uri=url))
        response = client.web_detection(image=image).web_detection
        for img in tqdm(response.full_matching_images, desc=f"Processing {url}", unit="image"):
            try:
                matching_image = Image.open(requests.get(img.url, headers=header, timeout=20, stream=True, verify=True).raw)
            except requests.exceptions.SSLError as e:
                logging.warning(f"SSL error processing {img.url}: {e}")
                continue
            matching_width, matching_height = matching_image.size
            img_urls.append(img.url)
            result['matching_imgs'].append(img.url)
            result['width_diff'].append(matching_width - og_width)
            result['height_diff'].append(matching_height - og_height)

        result['width_matching_imgs'] = [Image.open(requests.get(url, headers=header, timeout=20, stream=True, verify=True).raw).size[0] for url in img_urls]
        result['height_matching_imgs'] = [Image.open(requests.get(url, headers=header, timeout=20, stream=True, verify=True).raw).size[1] for url in img_urls]
    except Exception as e:
        logging.error(f'Error processing {url}: {e}')
        return pd.DataFrame(result)

    return pd.DataFrame(result)


# read in the datafile of image urls
df = pd.read_csv('/python_scripts/google_vision/input_file/wc_images_small.csv')

# process URLs in parallel using all available CPU cores
df_results = pd.concat([process_url(url) for url in tqdm(df['Images'], desc="Processing URLs")], ignore_index=True)

# merge the data with the original DataFrame
df = pd.merge(df, df_results, on='original_url', how='outer')

# drop image recommendations that are the same width/height or smaller
df = df.loc[(df['width_diff'] > 0) & (df['height_diff'] > 0)]

# drop duplicates if both width and height and duplicated
df.drop_duplicates(subset=["original_url", "width_diff", "height_diff"], keep="first", inplace=True)

# sort on highest values
df.sort_values(["width_diff", "height_diff"], ascending=[False, False], inplace=True)

# keep the top 3 highest values
df = df.groupby(['original_url']).head(3)

# reorder the columns
df = df[['original_url', 'matching_imgs', 'width_diff', 'height_diff', 'width_matching_imgs', 'height_matching_imgs']]

# save the results to a CSV file
df.to_csv("/python_scripts/higher_resolution_images_all.csv", index=False)
