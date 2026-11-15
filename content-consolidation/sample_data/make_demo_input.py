"""
Demo input for the Content Consolidation Analyser.

Writes demo_serp_rows.csv: one row per query and ranking URL, in the column
layout the script expects by default (search.q, result.organic_results.link).
The data is made up. It imitates search results for a running and outdoor
shop: groups of similar queries share many of the same ranking URLs, and
unrelated queries share none.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    python ../content_consolidation_analyzer.py --input demo_serp_rows.csv --output consolidation_results.csv
"""

import csv
import random

random.seed(20261114)

YOU = "https://www.yourstore.example"
COMPETITORS = ["competitor-a", "competitor-b", "competitor-c", "bigretailer", "reviewsite", "runnersmag", "outdoorforum"]


def pool(slugs):
    """Ranking URLs that a group of related queries keeps returning."""
    urls = []
    for i, slug in enumerate(slugs):
        domain = COMPETITORS[i % len(COMPETITORS)]
        urls.append(f"https://www.{domain}.example/{slug}")
    return urls


def other(query, n):
    """Results that only this query returns."""
    slug = query.replace(" ", "-")
    return [f"https://www.site{random.randint(100, 999)}.example/{slug}-{i}" for i in range(n)]


# Each group: queries, URLs they tend to share, and how many of those each query's results contain.
GROUPS = [
    {
        "queries": ["mens running shoes", "mens running trainers", "best mens running shoes",
                    "running shoes for men", "mens road running shoes", "buy mens running shoes"],
        "shared": [f"{YOU}/running/mens", f"{YOU}/blog/best-running-shoes"] + pool(
            ["mens-running-shoes", "running/men", "best-running-shoes-men", "mens/running-shoes",
             "reviews/mens-running-shoes", "gear/best-running-shoes", "threads/mens-running-shoes"]),
        "take": (7, 9),
    },
    {
        "queries": ["waterproof walking boots", "waterproof hiking boots", "best waterproof walking boots",
                    "mens waterproof walking boots"],
        "shared": [f"{YOU}/walking/boots/waterproof"] + pool(
            ["waterproof-walking-boots", "boots/waterproof", "best-waterproof-boots", "walking-boots",
             "reviews/waterproof-boots", "gear/walking-boots"]),
        "take": (5, 6),
    },
    {
        "queries": ["running jackets", "waterproof running jacket", "lightweight running jacket",
                    "running jackets for winter", "best running jacket"],
        "shared": [f"{YOU}/running/jackets", f"{YOU}/blog/running-jacket-guide"] + pool(
            ["running-jackets", "jackets/running", "best-running-jackets", "running/jackets",
             "reviews/running-jackets"]),
        "take": (4, 6),
    },
    {
        "queries": ["trail running socks", "running socks", "compression running socks"],
        "shared": [f"{YOU}/running/socks"] + pool(
            ["running-socks", "socks/running", "best-running-socks", "trail-running-socks"]),
        "take": (4, 5),
    },
]

UNRELATED = ["running shoe size guide", "how to clean trainers", "couch to 5k plan", "what is pronation",
             "marathon training schedule", "how to lace running shoes", "gps running watch",
             "energy gels for running", "foam roller exercises", "head torch for running",
             "running belt", "blister plasters"]

rows = []
for group in GROUPS:
    for query in group["queries"]:
        n = random.randint(*group["take"])
        results = random.sample(group["shared"], n) + other(query, 10 - n)
        random.shuffle(results)
        rows += [(query, url) for url in results]
for query in UNRELATED:
    rows += [(query, url) for url in other(query, 10)]

with open("demo_serp_rows.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["search.q", "result.organic_results.link"])
    writer.writerows(rows)

print(f"Wrote demo_serp_rows.csv: {len(rows)} rows, {len({q for q, _ in rows})} queries")
