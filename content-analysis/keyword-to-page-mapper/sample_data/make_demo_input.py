"""
Demo input for the Keyword-to-Page Mapper.

Writes demo_pages.csv (a crawl export with Address and H1-1) and
demo_keywords.csv (keyword and volume). The data is made up, for a fictional
running and outdoor shop. Most keywords have a page that suits them; some
have none, and those should come out as content gaps.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    python ../keyword_to_page_mapper_cli.py --pages demo_pages.csv --keywords demo_keywords.csv --output keyword_mapping.csv
"""

import csv
import random

random.seed(20261114)

SITE = "https://www.yourstore.example"

PAGES = [
    ("running/mens-running-shoes", "Mens Running Shoes"),
    ("running/womens-running-shoes", "Womens Running Shoes"),
    ("running/trail-running-shoes", "Trail Running Shoes"),
    ("running/running-jackets", "Running Jackets"),
    ("running/running-socks", "Running Socks"),
    ("running/running-shorts", "Running Shorts"),
    ("walking/walking-boots", "Walking Boots"),
    ("walking/waterproof-jackets", "Waterproof Jackets"),
    ("walking/rucksacks", "Rucksacks"),
    ("accessories/head-torches", "Head Torches"),
    ("accessories/hydration-packs", "Hydration Packs"),
    ("accessories/running-watches", "GPS Running Watches"),
    ("guides/how-to-choose-running-shoes", "How to Choose Running Shoes"),
    ("guides/couch-to-5k", "Couch to 5k Training Plan"),
]

KEYWORDS = [
    # keywords a page already suits
    "running shoes for men", "mens running trainers", "ladies running shoes", "womens running trainers",
    "trail shoes for running", "off road running shoes", "waterproof running jacket", "lightweight running jacket",
    "socks for running", "compression running socks", "mens running shorts", "hiking boots",
    "waterproof walking boots", "rain jackets", "hiking backpack", "day pack rucksack",
    "running head torch", "rechargeable headlamp", "running hydration vest", "gps watch for running",
    "which running shoes should i buy", "beginner 5k training plan",
    # keywords with no suitable page
    "trekking poles", "running sunglasses", "foam roller", "energy gels", "kids walking boots",
    "camping stove", "running buggy", "ice bath tub", "yoga mat", "cycling shorts",
]

with open("demo_pages.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Address", "H1-1"])
    writer.writerows([(f"{SITE}/{path}", h1) for path, h1 in PAGES])

with open("demo_keywords.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["keyword", "volume"])
    writer.writerows([(kw, random.choice([90, 140, 210, 320, 480, 720, 1000, 1600, 2400, 3600])) for kw in KEYWORDS])

print(f"Wrote demo_pages.csv ({len(PAGES)} pages) and demo_keywords.csv ({len(KEYWORDS)} keywords)")
