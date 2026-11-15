"""
Demo input for the Competitor Content Gap Finder.

Writes source_products.csv (your product titles) and two competitor files,
competitor_a.csv and competitor_b.csv, with the columns the script expects
by default (mpn, h1, url). The data is made up. Each product has a short
title on your site; competitors that stock the same MPN describe it with
more detail.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    python ../competitor_gap_finder.py --source source_products.csv --compare "competitor_*.csv" --output content_gaps.csv
"""

import csv
import random

random.seed(20261114)

# brand, model, product type, details a fuller title might include
PRODUCTS = [
    ("Northpeak", "Stride 4", "Running Shoes", ["Mens", "Road", "Neutral", "Cushioned", "Blue"]),
    ("Northpeak", "Stride 4 GTX", "Running Shoes", ["Womens", "Waterproof", "Trail", "Black"]),
    ("Northpeak", "Ridge Mid", "Walking Boots", ["Mens", "Waterproof", "Leather", "Brown"]),
    ("Northpeak", "Ridge Low", "Walking Shoes", ["Womens", "Waterproof", "Lightweight", "Grey"]),
    ("Calder", "Aero Shell", "Running Jacket", ["Mens", "Waterproof", "Lightweight", "Packable", "Hooded"]),
    ("Calder", "Aero Wind", "Running Jacket", ["Womens", "Windproof", "Reflective", "Packable"]),
    ("Calder", "Thermal Tight", "Running Tights", ["Mens", "Winter", "Brushed", "Reflective"]),
    ("Calder", "Tempo Short", "Running Shorts", ["Mens", "5 Inch", "Lined", "Zip Pocket"]),
    ("Fellside", "Trail Crew", "Running Socks", ["Cushioned", "Merino", "Twin Pack"]),
    ("Fellside", "Compress Pro", "Running Socks", ["Compression", "Knee High", "Unisex"]),
    ("Fellside", "Hydro Vest 5L", "Running Vest", ["Hydration", "Two Soft Flasks", "Unisex"]),
    ("Fellside", "Beam 400", "Head Torch", ["Rechargeable", "400 Lumens", "Waterproof"]),
    ("Northpeak", "Summit 30", "Rucksack", ["30 Litre", "Rain Cover", "Hiking"]),
    ("Northpeak", "Glide 2", "Running Shoes", ["Womens", "Road", "Stability", "Pink"]),
    ("Calder", "Base Layer LS", "Base Layer", ["Mens", "Long Sleeve", "Merino", "Thermal"]),
    ("Calder", "Storm Cap", "Running Cap", ["Waterproof", "Reflective", "Adjustable"]),
    ("Fellside", "Grip Glove", "Running Gloves", ["Touchscreen", "Windproof", "Reflective"]),
    ("Fellside", "Flask 500", "Soft Flask", ["500ml", "Collapsible", "BPA Free"]),
    ("Northpeak", "Path Gaiter", "Trail Gaiters", ["Waterproof", "Unisex", "Adjustable"]),
    ("Calder", "Race Belt", "Running Belt", ["Adjustable", "Gel Loops", "Phone Pocket"]),
    ("Fellside", "Recover Roller", "Foam Roller", ["High Density", "33cm", "Textured"]),
    ("Northpeak", "Dash Kids", "Running Shoes", ["Junior", "Velcro", "Lightweight"]),
    ("Calder", "Loop Band", "Headband", ["Thermal", "Reflective", "One Size"]),
    ("Fellside", "Blister Kit", "Blister Plasters", ["Hydrocolloid", "Pack of 10"]),
]

FIELDS = ["mpn", "h1", "url"]


def mpn(i):
    return f"{PRODUCTS[i][0][:2].upper()}-{1000 + i * 37}"


def slug(text):
    return text.lower().replace(" ", "-")


def write(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {filename}: {len(rows)} products")


# Your site: brand, model and product type only.
source = []
for i, (brand, model, kind, _) in enumerate(PRODUCTS):
    title = f"{brand} {model} {kind}"
    source.append({"mpn": mpn(i), "h1": title, "url": f"https://www.yourstore.example/p/{slug(title)}"})
write("source_products.csv", source)

# Competitors: a share of the same products, with some of the extra details.
for name, share in [("competitor_a", 0.8), ("competitor_b", 0.6)]:
    rows = []
    for i, (brand, model, kind, details) in enumerate(PRODUCTS):
        if random.random() > share:
            continue
        extra = random.sample(details, random.randint(2, len(details)))
        title = f"{brand} {model} {' '.join(extra)} {kind}"
        rows.append({"mpn": mpn(i), "h1": title,
                     "url": f"https://www.{name.replace('_', '-')}.example/{slug(title)}"})
    write(f"{name}.csv", rows)
