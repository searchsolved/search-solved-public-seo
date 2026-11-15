"""
Demo input for the Entity Extractor (spaCy version).

Writes demo_articles.csv with a url column and a content column. The pages
are made up, for a fictional running blog, but the people, places, events
and organisations in them are real, so the model has something to find.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    python ../entity_extractor_cli.py --input demo_articles.csv --id-col url --output entity_counts.csv
"""

import csv

SITE = "https://www.yourstore.example/blog"

ARTICLES = [
    ("marathon-majors-guide",
     "The six original World Marathon Majors are Tokyo, Boston, London, Berlin, Chicago and New York. "
     "The Boston Marathon is the oldest annual marathon in the world and was first run in 1897. "
     "The Berlin Marathon is known for fast times, and the London Marathon finishes on The Mall. "
     "Runners who finish all six receive the Six Star Medal from Abbott World Marathon Majors."),
    ("fastest-marathon-runners",
     "Eliud Kipchoge of Kenya ran a marathon distance in under two hours in Vienna in 2019. "
     "Paula Radcliffe set a womens marathon record at the London Marathon in 2003. "
     "Kipchoge has won the Berlin Marathon several times, and Radcliffe won the New York City Marathon three times. "
     "Both athletes were sponsored by Nike."),
    ("parkrun-beginners",
     "parkrun began in Bushy Park in London in 2004 and is now held every Saturday in parks around the world. "
     "Paul Sinton-Hewitt founded the event. Many new runners follow the NHS Couch to 5K plan before their first parkrun. "
     "You can record your run on Strava or with a Garmin watch."),
    ("running-watch-comparison",
     "Garmin, Polar, Coros and Suunto all make GPS running watches. Apple and Samsung sell smartwatches that track runs too. "
     "Garmin is based in Kansas, Polar and Suunto are from Finland, and Coros is based in California. "
     "Most watches upload to Strava, and Garmin has its own platform called Garmin Connect."),
    ("fell-running-in-the-lake-district",
     "The Lake District in Cumbria is the home of English fell running. The Bob Graham Round covers 42 peaks, "
     "including Scafell Pike, Helvellyn and Skiddaw, and starts from the Moot Hall in Keswick. "
     "Kilian Jornet set a record for the round in 2018, and Jasmin Paris has held the womens record. "
     "The Fell Runners Association publishes the race calendar."),
    ("running-shoe-brands",
     "Nike, Adidas, Asics, Brooks, Saucony, New Balance and Hoka are among the best known running shoe brands. "
     "Asics was founded in Kobe in Japan, Adidas in Germany and Nike in Oregon. "
     "Hoka began in France and is now owned by Deckers Brands. On, a Swiss company, counts Roger Federer among its investors."),
]

with open("demo_articles.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "content"])
    for slug, text in ARTICLES:
        writer.writerow([f"{SITE}/{slug}", text])

print(f"Wrote demo_articles.csv: {len(ARTICLES)} pages")
