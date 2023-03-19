import json
import os
import pandas as pd
from PIL import Image
import requests
from google.cloud import vision

header = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Mobile Safari/537.36'}


# read in the json secrets file and instantiate google vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/python_scripts/cloud_vision_api.json"
client = vision.ImageAnnotatorClient()
image = vision.Image()

# read in the datafile of image urls
df = pd.read_csv('/python_scripts/google_vision/input_file/internal_images_mini.csv')

# store the data
full_images = []
matching_img_width = []
matching_img_height = []
og_url_df1 = []

# store the og data for a second dataframe to merge in

og_width = []
og_height = []
og_url_df2 = []

for url in df['Address']:
    try:
        image.source.image_uri = url
        web_response = client.web_detection(image=image)
        web_content = web_response.web_detection
        json_string = type(web_content).to_json(web_content)
        d = json.loads(json_string)
        response = d['fullMatchingImages']
        og_url_df2.append(url)

        # get the image width
        try:
            im = Image.open(requests.get(url, timeout=20, headers=header, stream=True).raw)
            width, height = im.size
            og_height.append(height)
            og_width.append(width)
        except Exception:
            print("exception at 45")
            og_height.append(0)
            og_width.append(0)


        for d in response:
            print(d['url'])
            full_images.append((d['url']))
            og_url_df1.append(url)
            try:
                im = Image.open(requests.get(d['url'], timeout=20, headers=header, stream=True).raw)
                width, height = im.size
                matching_img_height.append(height)
                matching_img_width.append(width)
            except Exception:
                print("exception at 59")
                matching_img_height.append(0)
                matching_img_width.append(0)
    except TypeError:
        print("exception at 63")
        pass

df = pd.DataFrame(None)
df['original_url'] = og_url_df1
df['matching_imgs'] = full_images
df['width_matching_imgs'] = matching_img_width
df['height_matching_imgs'] = matching_img_height

# make second dataframe to blend the data back in vlookup style
df2 = pd.DataFrame(None)
df2['original_url'] = og_url_df2
df2['width_source_img'] = og_width
df2['height_source_img'] = og_height

# merge the data
df = pd.merge(df, df2, on="original_url", how="left")

# calculate size difference
df['width_diff'] = df['width_matching_imgs'] - df['width_source_img']
df['height_diff'] = df['height_matching_imgs'] - df['height_source_img']

# drop and image recommendations that are the same width / height or smaller
df = df.loc[~((df['width_diff'] <= 0) | (df['height_diff'] <= 0))]

# re-order the column
df = df[['original_url', 'matching_imgs', 'width_diff', 'height_diff', 'width_matching_imgs', 'height_matching_imgs', 'width_source_img', 'height_source_img']]

# drop duplicates if both width and height and duplicated
df.drop_duplicates(subset=["width_diff", "width_diff"], keep="first", inplace=True)

# sort on highest values
df.sort_values(["width_diff", "width_diff"], ascending=[False, False], inplace=True)

# keep the top 3 highest values
df = df.groupby(['original_url']).head(3)


df.to_csv("/python_scripts/higher_resolution_images.csv")
