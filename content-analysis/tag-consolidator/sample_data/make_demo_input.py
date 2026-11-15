"""
Demo input for the Tag Consolidator.

Writes demo_tags.csv with a Primary Tag column and a Secondary Tag column.
The data is made up, for a fictional running and outdoor shop: each product
group has a long list of detailed tags, many of which say nearly the same
thing.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    export OPENAI_API_KEY=your-key
    python ../tag_consolidator_cli.py --input demo_tags.csv --output consolidated_tags.csv
"""

import csv

TAGS = {
    "Running Shoes": [
        "max cushioning", "soft cushioning", "plush midsole", "responsive foam", "carbon plate", "racing flat",
        "stability support", "overpronation control", "medial post", "wide fit", "extra wide", "narrow fit",
        "road running", "tarmac", "trail grip", "deep lugs", "rock plate",
    ],
    "Jackets": [
        "waterproof membrane", "taped seams", "water resistant", "shower proof", "windproof", "wind shell",
        "packable", "stuffs into pocket", "reflective trim", "high visibility", "insulated", "fleece lined",
        "hooded", "adjustable hood", "pit zips", "breathable",
    ],
    "Socks": [
        "merino wool", "wool blend", "synthetic", "anti blister", "double layer", "seamless toe",
        "compression", "graduated compression", "ankle length", "crew length", "knee high",
        "cushioned sole", "padded heel", "twin pack", "multipack",
    ],
}

rows = [(primary, tag) for primary, tags in TAGS.items() for tag in tags]

with open("demo_tags.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Primary Tag", "Secondary Tag"])
    writer.writerows(rows)

print(f"Wrote demo_tags.csv: {len(rows)} rows")
