"""
Demo input for the Meta Description Grader and the Meta Description Rewriter.

Writes demo_meta_descriptions.csv with url, title, h1 and meta_description
columns. The pages are made up, for a fictional running and outdoor shop.
The descriptions are written to vary in quality: some are hype, some are
too short or too long, some are keyword lists and some are sound.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py

Then upload demo_meta_descriptions.csv on the bulk tab of either app and map
the URL and Meta Description columns (the Rewriter also takes H1 and Title).
"""

import csv

SITE = "https://www.yourstore.example"

PAGES = [
    ("/running/mens-running-shoes", "Mens Running Shoes | Your Store", "Mens Running Shoes",
     "BEST mens running shoes!!! Shop NOW for AMAZING deals you won't believe. Don't miss out!!!"),
    ("/running/womens-running-shoes", "Womens Running Shoes | Your Store", "Womens Running Shoes",
     "Womens running shoes."),
    ("/running/trail-running-shoes", "Trail Running Shoes | Your Store", "Trail Running Shoes",
     "Trail running shoes, trail shoes, off road running shoes, trail trainers, running shoes for trails, "
     "cheap trail running shoes UK."),
    ("/walking/walking-boots", "Walking Boots | Your Store", "Walking Boots",
     "Welcome to our walking boots page. Here you will find information about the walking boots that are "
     "sold by our company on this website."),
    ("/walking/waterproof-jackets", "Waterproof Jackets | Your Store", "Waterproof Jackets",
     "Stay dry on the hill with waterproof jackets tested in British weather. Compare breathable shells "
     "from £60, with free returns for 60 days."),
    ("/accessories/head-torches", "Head Torches | Your Store", "Head Torches",
     "Our head torches are available in a wide range of brightness levels and battery types, and they can "
     "be used for running at night, for camping, for walking the dog and for many other activities that "
     "take place in the dark throughout the year."),
    ("/guides/how-to-choose-running-shoes", "How to Choose Running Shoes | Your Store",
     "How to Choose Running Shoes",
     "Find the right running shoes in five steps: measure your feet, check your gait, pick the cushioning "
     "and test the fit before you buy."),
    ("/guides/couch-to-5k", "Couch to 5k Training Plan | Your Store", "Couch to 5k Training Plan",
     "Couch to 5k training plan"),
    ("/running/running-socks", "Running Socks | Your Store", "Running Socks",
     "Running socks are sold here. Blisters can be prevented by socks. Many socks are stocked."),
    ("/accessories/running-watches", "GPS Running Watches | Your Store", "GPS Running Watches",
     "GPS running watches that track pace, distance and heart rate. Order before 3pm for next day delivery."),
]

with open("demo_meta_descriptions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "title", "h1", "meta_description"])
    for path, title, h1, meta in PAGES:
        writer.writerow([SITE + path, title, h1, meta])

print(f"Wrote demo_meta_descriptions.csv: {len(PAGES)} pages")
