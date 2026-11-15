"""
Demo input for the Bulk H1 Translator.

Writes demo_h1s.csv with Address, H1-1 and Language columns, the names a
Screaming Frog export uses. The pages are made up, for the regional sites
of a fictional running and outdoor shop: two H1s each in French, German,
Spanish and Italian, one that is already in English and one page with no
H1, so the output shows how the tool handles both.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py
    python ../bulk_h1_translator_cli.py --input demo_h1s.csv --output translated_h1s.csv
"""

import csv

ROWS = [
    ("https://www.yourstore.example/fr/chaussures-running-homme", "Chaussures de running pour homme", "French"),
    ("https://www.yourstore.example/fr/vestes-impermeables-randonnee", "Vestes imperméables pour la randonnée", "French"),
    ("https://www.yourstore.example/de/laufschuhe-damen", "Laufschuhe für Damen", "German"),
    ("https://www.yourstore.example/de/trainingsplan-halbmarathon", "Trainingsplan für den ersten Halbmarathon", "German"),
    ("https://www.yourstore.example/es/zapatillas-running-mujer", "Zapatillas de running para mujer", "Spanish"),
    ("https://www.yourstore.example/es/como-elegir-mochila-senderismo", "Cómo elegir una mochila de senderismo", "Spanish"),
    ("https://www.yourstore.example/it/scarpe-trail-running", "Scarpe da trail running", "Italian"),
    ("https://www.yourstore.example/it/giacche-antivento-corsa", "Giacche antivento da corsa", "Italian"),
    ("https://www.yourstore.example/fr/mens-running-shoes", "Mens Running Shoes", "French"),
    ("https://www.yourstore.example/de/kontakt", "", "German"),
]

with open("demo_h1s.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Address", "H1-1", "Language"])
    writer.writerows(ROWS)

print(f"Wrote demo_h1s.csv: {len(ROWS)} rows")
