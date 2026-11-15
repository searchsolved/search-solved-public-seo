"""
Demo input for the Review Tagger and the Review Sentiment Extractor.

Writes demo_reviews.csv with a Review column and an order_id column. The
reviews are made up, for a fictional running and outdoor shop. Each one is
put together from a product, an opening and one or two remarks about a
topic such as delivery, sizing or comfort.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    export OPENAI_API_KEY=your-key
    python ../review_tagger_cli.py --input demo_reviews.csv --output tagged_reviews.csv
"""

import csv
import random

random.seed(20261114)

PRODUCTS = ["running shoes", "trail shoes", "walking boots", "running jacket", "running socks", "running tights"]

OPENINGS = ["Bought these {p} last month.", "Ordered the {p} for a trip.", "My second pair of {p} from here.",
            "Got the {p} as a present.", "Needed new {p} before a race.", "First time buying {p} online."]

REMARKS = {
    "delivery": [("Arrived the next day, well packed.", 1), ("Delivery took over a week and nobody replied to my email.", -1),
                 ("Came two days early.", 1), ("The courier left the parcel in the rain.", -1)],
    "sizing": [("True to size and a good fit.", 1), ("They run small, I had to send them back for a bigger size.", -1),
               ("Size guide was spot on.", 1), ("Much tighter than the size suggests.", -1)],
    "comfort": [("Comfortable from the first run, no rubbing.", 1), ("Rubbed badly on the first outing.", -1),
                ("Really cushioned and light.", 1), ("The seams dig in after an hour.", -1)],
    "durability": [("Still look new after two hundred miles.", 1), ("The stitching came apart within a month.", -1),
                   ("Solid build, feels like it will last.", 1), ("Started falling apart after a few weeks.", -1)],
    "value": [("Great price compared with other shops.", 1), ("Overpriced for what you get.", -1),
              ("Worth every penny.", 1), ("Found the same thing cheaper elsewhere a week later.", -1)],
    "waterproofing": [("Kept me dry through a full day of rain.", 1), ("Leaked at the seams in the first shower.", -1),
                      ("Properly waterproof, stayed dry all day.", 1), ("Water resistant at best, not waterproof.", -1)],
    "customer service": [("The returns process was quick and easy.", 1), ("Could not get anyone on the phone.", -1),
                         ("Staff sorted out an exchange the same day.", 1), ("Still waiting for my refund.", -1)],
}

rows = []
topics = list(REMARKS)
for i in range(40):
    product = random.choice(PRODUCTS)
    review = random.choice(OPENINGS).format(p=product)
    for topic in random.sample(topics, random.choice([1, 1, 2])):
        review += " " + random.choice(REMARKS[topic])[0]
    rows.append((f"R{1001 + i}", review))

with open("demo_reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["order_id", "Review"])
    writer.writerows(rows)

print(f"Wrote demo_reviews.csv: {len(rows)} reviews")
