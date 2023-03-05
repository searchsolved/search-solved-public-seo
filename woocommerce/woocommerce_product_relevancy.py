"""by Lee Foot 5th March 2023
This script retrieves categories and products from a WooCommerce store, sorts the products based on how closely their names match the category name, and updates the product order based on the sort order. It uses the Woocommerce API, fuzzywuzzy library for string matching, and colorama for terminal output formatting.
To use this script, the user needs to replace the placeholders "YOUR WEBSITE", "YOUR CONSUMER KEY", and "YOUR CONSUMER SECRET" in the wcapi variable with their actual website URL and API credentials. The credentials can be found by logging into the WooCommerce store and going to the "Settings" page, selecting the "Advanced" tab, and then selecting the "REST API" option. From there, the user can generate API keys for their store.
It is important to note that this script has a maximum number of retries and retry delay settings, which can be adjusted by changing the MAX_RETRIES and RETRY_DELAY constants. Additionally, the user must have the required libraries installed (woocommerce, fuzzywuzzy, colorama, and requests) before running the script."""

from woocommerce import API
from fuzzywuzzy import fuzz
from colorama import Fore, Style
import requests

import time

MAX_RETRIES = 6
RETRY_DELAY = 1

wcapi = API(
    url="YOUR WEBSITE",
    consumer_key="YOUR CONSUMER KEY",
    consumer_secret="YOUR CONSUMER SECRET",
    version="wc/v3",
    timeout=(120, 120)  # timeout for sending / receiving in seconds
)

categories = []
page = 1

while True:
    for attempt in range(MAX_RETRIES):
        try:
            response = wcapi.get("products/categories", params={"per_page": 100, "page": page})
            if response.ok and response.json():
                categories += response.json()
                page += 1
                break
        except requests.exceptions.ReadTimeout as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Read timeout error. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Maximum number of retries ({MAX_RETRIES}) exceeded. Aborting.")
                raise e

    if not categories:
        break

    print(f"Found {len(categories)} categories in the store")

    for category in categories:
        print(f"Processing category: {category['name']}")
        category_name = category["name"]
        category_id = category["id"]

        for attempt in range(MAX_RETRIES):
            try:
                products = wcapi.get(f"products?category={category_id}").json()
                break
            except requests.exceptions.ReadTimeout as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Read timeout error. Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Maximum number of retries ({MAX_RETRIES}) exceeded. Aborting.")
                    raise e

        num_products = len(products)
        print(f"Found {num_products} products in the category")

        if num_products > 1:
            products_with_score = {}
            for product in products:
                score = fuzz.token_set_ratio(product["name"], category_name)
                products_with_score[product["name"]] = (score, product["id"])

            sorted_products = {k: v for k, v in
                               sorted(products_with_score.items(), key=lambda item: (-item[1][0], item[0]))}
            print(f"Sorted Products: {sorted_products}")

            for i, (product_name, relevance_id) in enumerate(sorted_products.items()):
                product_id = relevance_id[1]

                search_query = f"products/{product_id}?search={category_name}"

                for attempt in range(MAX_RETRIES):
                    try:
                        product_search_result = wcapi.get(search_query).json()
                        break
                    except requests.exceptions.ReadTimeout as e:
                        if attempt < MAX_RETRIES - 1:
                            print(f"Read timeout error. Retrying in {RETRY_DELAY} seconds...")
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"Maximum number of retries ({MAX_RETRIES}) exceeded. Aborting.")
                            raise e

                if product_search_result:
                    existing_order = product_search_result.get("menu_order", -1)
                    if existing_order != i:
                        data = {"menu_order": i}

                        for attempt in range(MAX_RETRIES):
                            try:
                                wcapi.put(f"products/{product_id}", data).json()
                                print(
                                    f"{Fore.GREEN}Posting {product_name} to WooCommerce. ID: {product_id}, Menu Order: {i}{Style.RESET_ALL}")
                                break
                            except requests.exceptions.ReadTimeout as e:
                                if attempt < MAX_RETRIES - 1:
                                    print(f"Read timeout error. Retrying in {RETRY_DELAY} seconds...")
                                    time.sleep(RETRY_DELAY)
                                else:
                                    print(f"Maximum number of retries ({MAX_RETRIES}) exceeded. Aborting.")
                                    raise e

                    else:
                        print(
                            f"{Fore.YELLOW}Skipping update for {product_name}. ID: {product_id}, Menu Order: {existing_order}. Already optimized!{Style.RESET_ALL}")

                else:
                    print(f"{Fore.RED}No product found with ID: {product_id}{Style.RESET_ALL}")
