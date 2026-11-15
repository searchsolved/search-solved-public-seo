"""
Demo input for the Category Title Suggester.

Writes demo_pages.csv (a crawl export with Address and Title 1) and
demo_gsc.csv (Search Console rows with page, query and clicks). The data is
made up. Each category page earns clicks for its head term, which is already
in the title, and for more specific queries that are not.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    python ../category_title_suggester_cli.py --crawl demo_pages.csv --gsc demo_gsc.csv --brand "your store" --output title_suggestions.csv
"""

import csv
import random

random.seed(20261114)

SITE = "https://www.yourstore.example"

# category path, head term (in the title), other queries the page earns clicks for
CATEGORIES = [
    ("running/mens-running-shoes", "Mens Running Shoes",
     ["mens road running shoes", "mens trail running shoes", "mens running trainers", "cushioned running shoes mens",
      "wide fit running shoes mens", "mens running shoes sale", "best running shoes for men", "neutral running shoes mens"]),
    ("running/womens-running-shoes", "Womens Running Shoes",
     ["womens trail running shoes", "ladies running trainers", "womens running shoes sale", "stability running shoes womens",
      "lightweight running shoes womens", "womens road running shoes"]),
    ("running/running-jackets", "Running Jackets",
     ["waterproof running jacket", "lightweight running jacket", "mens running jacket", "womens running jacket",
      "reflective running jacket", "packable running jacket", "winter running jacket"]),
    ("running/running-socks", "Running Socks",
     ["trail running socks", "compression running socks", "merino running socks", "anti blister running socks",
      "cushioned running socks"]),
    ("walking/walking-boots", "Walking Boots",
     ["waterproof walking boots", "mens walking boots", "womens walking boots", "leather walking boots",
      "lightweight walking boots", "wide fit walking boots", "walking boots sale"]),
    ("walking/rucksacks", "Rucksacks",
     ["hiking rucksack", "30 litre rucksack", "day pack", "waterproof rucksack", "rucksack with rain cover"]),
    ("accessories/head-torches", "Head Torches",
     ["rechargeable head torch", "running head torch", "bright head torch", "waterproof head torch"]),
    ("accessories/hydration", "Hydration Packs",
     ["running vest", "hydration vest", "soft flask", "running water bottle", "hydration pack for running"]),
]

pages = [{"Address": f"{SITE}/category/{path}", "Title 1": f"{head} | Your Store"} for path, head, _ in CATEGORIES]
pages.append({"Address": f"{SITE}/blog/running-jacket-guide", "Title 1": "How to Choose a Running Jacket | Your Store"})

with open("demo_pages.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Address", "Title 1"])
    writer.writeheader()
    writer.writerows(pages)

gsc = []
for path, head, queries in CATEGORIES:
    page = f"{SITE}/category/{path}"
    gsc.append({"page": page, "query": head.lower(), "clicks": random.randint(250, 900)})
    gsc.append({"page": page, "query": f"your store {head.lower()}", "clicks": random.randint(20, 80)})
    for query in queries:
        gsc.append({"page": page, "query": query, "clicks": random.randint(3, 160)})
gsc.append({"page": f"{SITE}/blog/running-jacket-guide", "query": "how to choose a running jacket", "clicks": 41})
random.shuffle(gsc)

with open("demo_gsc.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["page", "query", "clicks"])
    writer.writeheader()
    writer.writerows(gsc)

print(f"Wrote demo_pages.csv ({len(pages)} pages) and demo_gsc.csv ({len(gsc)} rows)")
