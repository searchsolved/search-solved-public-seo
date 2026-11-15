"""
Demo input for the Content Duplication Finder.

Writes demo_crawl_with_copy.csv with the three columns the app needs
(Address, H1-1, Copy 1). The data is made up. It imitates a running and
outdoor shop with three kinds of page: store pages built from one template
with the town swapped, product guides that were copied and lightly edited,
and pages with copy of their own.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    Then upload demo_crawl_with_copy.csv to the app.
"""

import csv
import random

random.seed(20261114)

SITE = "https://www.yourstore.example"
rows = []

# 1. Store pages: one template, a few words changed per town.
STORE = ("Visit our {town} running shop for expert gait analysis, a full range of road and trail shoes, "
         "and friendly advice from runners who know the local routes. Our {town} team can fit you for "
         "shoes in around twenty minutes, and we stock clothing, socks, nutrition and head torches all "
         "year round. Parking is available {parking}, and the shop is open seven days a week. "
         "Book a free fitting online or just drop in. {extra}")
TOWNS = [("Leeds", "behind the store", "We also host a free social run every Tuesday evening."),
         ("York", "on the street outside", "We also host a free social run every Wednesday evening."),
         ("Harrogate", "in the retail park", "Ask about our beginners group, which meets on Saturdays."),
         ("Sheffield", "behind the store", "We also host a free social run every Thursday evening."),
         ("Skipton", "in the town car park nearby", "Our Skipton shop is the closest to the Dales fell races.")]
for town, parking, extra in TOWNS:
    rows.append((f"{SITE}/stores/{town.lower()}", f"{town} Running Shop",
                 STORE.format(town=town, parking=parking, extra=extra)))

# 2. Guides that were copied and lightly edited.
GUIDE = ("Choosing {item} comes down to fit, terrain and how far you run. Start with the surface: road "
         "running needs cushioning and a smooth sole, while trails need grip and protection. Then think "
         "about distance, because comfort matters more the longer you are out. {middle} Try several pairs, "
         "run in them if you can, and replace them when the cushioning feels flat or the tread has worn.")
GUIDES = [("guides/how-to-choose-running-shoes", "How to Choose Running Shoes", "running shoes",
           "A neutral shoe suits most runners, and a stability shoe helps if your foot rolls inwards."),
          ("guides/how-to-choose-trail-shoes", "How to Choose Trail Running Shoes", "trail running shoes",
           "Deeper lugs suit mud, and a rock plate helps on stony ground."),
          ("guides/running-shoe-buying-guide", "Running Shoe Buying Guide", "running shoes",
           "A neutral shoe suits most runners, and a stability shoe helps if your foot rolls in.")]
for path, h1, item, middle in GUIDES:
    rows.append((f"{SITE}/{path}", h1, GUIDE.format(item=item, middle=middle)))

# 3. Pages with their own copy.
UNIQUE = [
    ("running/mens", "Mens Running Shoes",
     "Our mens running shoes cover everything from daily trainers to carbon plated racers. Filter by brand, "
     "width and support type to narrow the range, or sort by price to find this season's best value pairs."),
    ("walking/boots", "Walking Boots",
     "Leather and fabric walking boots for hills, coast paths and winter dog walks. Every pair is waterproof "
     "tested, and most are available in half sizes and wide fittings."),
    ("blog/couch-to-5k", "Couch to 5k: A Nine Week Plan",
     "This plan takes a complete beginner to thirty minutes of continuous running in nine weeks. Each week "
     "has three sessions that mix walking and running, with rest days between them."),
    ("help/returns", "Returns and Exchanges",
     "You can return unworn items within thirty days for a refund or exchange. Print a prepaid label from "
     "your account, attach it to the parcel and drop it at any collection point."),
    ("help/delivery", "Delivery Information",
     "Standard delivery takes two to three working days and is free on orders over fifty pounds. Next day "
     "delivery is available until eight in the evening, Monday to Friday."),
    ("blog/marathon-kit-list", "Marathon Kit List",
     "A checklist for race day: shoes you have trained in, a tested breakfast, gels, a throwaway layer for "
     "the start pen, and safety pins for your number."),
    ("about", "About Us", None),
]
for path, h1, copy in UNIQUE:
    rows.append((f"{SITE}/{path}", h1, copy))

random.shuffle(rows)
with open("demo_crawl_with_copy.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Address", "H1-1", "Copy 1"])
    writer.writerows(rows)
print(f"Wrote demo_crawl_with_copy.csv: {len(rows)} pages")
