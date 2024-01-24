from woocommerce import API
from fuzzywuzzy import fuzz

# Set up the WooCommerce API connection
wcapi = API(
    url="http://example.com",
    consumer_key="YOUR CONSUMER KEY",
    consumer_secret="YOUR CONSUMER SECRET",
    version="wc/v3",
    timeout=(120, 120)  # timeout for sending / receiving in seconds
)
# Fetch all the product categories using pagination
categories = []
page = 1
while True:
    response = wcapi.get("products/categories", params={"per_page": 100, "page": page})
    if not response.ok or not response.json():
        break
    categories += response.json()
    print(f"Fetched {len(categories)} categories, fetching next page...")
    page += 1

print(f"Found {len(categories)} categories in the store")
# Fetch all the product categories using pagination
categories = []
page = 1
while True:
    response = wcapi.get("products/categories", params={"per_page": 100, "page": page})
    if not response.ok or not response.json():
        break
    categories += response.json()
    page += 1

print(f"Found {len(categories)} categories in the store")

# Loop through each category
for category in categories:
    print(f"Processing category: {category['name']}")
    category_name = category["name"]
    category_id = category["id"]

    # Fetch all the products in the category
    products = wcapi.get(f"products?category={category_id}").json()
    num_products = len(products)
    print(f"Found {num_products} products in the category")

    if num_products > 0:
        # Calculate the relevance score between each product name and the category name
        products_with_score = {}
        for product in products:
            score = fuzz.token_sort_ratio(product["name"], category_name)
            products_with_score[product["name"]] = (score, product["id"])
            # print(f"Product: {product['name']}, ID: {product['id']}, Score: {score}")

        # Sort the products by relevance score in descending order, and then by product name alphabetically
        sorted_products = {k: v for k, v in
                           sorted(products_with_score.items(), key=lambda item: (-item[1][0], item[0]))}
        print(f"Sorted Products: {sorted_products}")

        # Create a list of product updates
        product_updates = []
        for i, (product_name, relevance_id) in enumerate(sorted_products.items()):
            product_id = relevance_id[1]

            # Add the product update to the list
            product_updates.append({"id": product_id, "menu_order": i})

        # Batch update the products
        response = wcapi.post("products/batch", {"update": product_updates})
        if response.ok:
            print("Batch update successful")
        else:
            print("Batch update failed")
            print(response.json())
    else:
        print("No products found in the category")
