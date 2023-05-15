# eBay Related Search Scraper

This Python code implements a web scraping tool to retrieve related search keywords from eBay. It provides a user-friendly web interface using the Streamlit library. The scraper allows you to search for related keywords on eBay and visualize the results in an interactive tree format.

## Usage

1. **Install the required dependencies** listed in the `requirements.txt` file by running the following command:

```pip install -r requirements.txt```

2. **Run the Python script** using the following command:

```streamlit run ebay-related-searches.py```

3. The Streamlit app will launch in your default web browser.

4. **Enter the keyword** you want to search for on eBay in the text input field.

5. **Select the country code top-level domain (ccTLD)** from the dropdown menu to specify the country.

6. **Click the "Submit" button** to start the scraping process.

7. **Wait for the scraper** to retrieve the related search keywords from eBay. The progress will be displayed.

8. Once the scraping is complete, an **interactive tree visualization** of the related search keywords will be displayed.

9. To **download the scraped data** as a CSV file, click the "Download your report!" button.

![eBay Related Searches](https://imgur.com/VjahHDW)
