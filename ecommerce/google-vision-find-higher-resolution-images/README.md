Automatically Source High-Resolution Images for eCommerce Sites

AutoImageRes leverages the power of Google Vision API to automate the sourcing of high-resolution images for medium to large eCommerce websites. This Python script efficiently upgrades standard product images, providing a seamless transition to higher quality visuals without the manual hassle.

## Overview
This tool is designed to streamline the process of enhancing product images on eCommerce platforms. By integrating with the Google Vision API, it identifies and fetches matching high-resolution images, making your product displays more appealing and professional.

Key Features:
- **Automatic Sourcing:** Fetch high-resolution images on autopilot from an image export via a Screaming Frog crawl.
- **Output Clarity:** Outputs include the new size and pixel difference versus the original file, simplifying prioritization.
- **User-Friendly:** Simple and robust, with clear, actionable results.

## Quick Start Guide

1. **Crawl Your Site:** Use Screaming Frog to crawl your site.
2. **Export Images:** From Screaming Frog, select 'Images' under the 'Internal' tab to export all images. Remember to export as a `.csv`, not an `.xlsx` file.
3. **Cloud Vision API Setup:** Set up the Cloud Vision API and download the JSON file. (Instructions [here](https://cloud.google.com/vision/docs/quickstart-client-libraries))
4. **Update Constants:** Modify the Constants section in the script to point to your Google Vision JSON file and the source image `.csv` file.
5. **Specify Output:** Choose a location and filename for the save file.
6. **Run the Script:** Execute the script and watch the magic happen!

## Benefits of Higher Resolution Images
- **Increased Conversions:** Enhance your marketing efforts with high-quality images, leading to better conversion rates.
- **Google Image Search Traffic:** Drive additional traffic through high-resolution images in Google Image Search.
- **Reduce Returns:** Detailed product images can decrease return rates by providing clearer product representation.
- **Enhanced User Experience:** High-quality images project professionalism and trust, potentially increasing site engagement.

## User Configurable Options
- `GOOGLE_APPLICATION_CREDENTIALS_PATH`: File path for Google Cloud service account key.
- `INPUT_FILE_PATH`: Directory path to the input CSV file.
- `OUTPUT_FILE_PATH`: Directory path for the output CSV file.
- `REQUEST_TIMEOUT`: Maximum duration for HTTP request completion.
- `MIN_RESOLUTION`: Skip source images equal to or over a set resolution.
- `MAX_IMAGES`: Maximum number of image suggestions.
- `SKIPPED_FILE_TYPES`: List of image file extensions to be ignored.

## Image Scraping Process
The script employs a cautious approach to downloading images, given varying server thresholds for scraping. It utilizes rotating user agents and three methods to retrieve high-resolution images:

### Attempt 1: Requests
The fastest method using the Requests library. However, it's the most prone to being blocked.

### Attempt 2: PyPuppeteer
If Requests fail, PyPuppeteer is the next in line for image retrieval.

### Attempt 3: Selenium
As a last resort, Selenium, which mimics human browser interaction, is employed to secure the image.

**Advanced Usage:** For a more comprehensive result set, consider using proxy servers. This is left as an exercise for advanced users.

## Note
A basic knowledge of Python is necessary to effectively use this script, including how to install and run scripts.

## Full Code and Article
Access the full script and a detailed write-up [here](https://leefoot.co.uk/portfolio/automate-higher-resolution-images-ecom/).
