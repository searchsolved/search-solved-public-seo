"""
Demo input for the Content Hub Classifier.

Writes demo_articles.csv with a url column and a content column. The
articles are made up, for a fictional running blog, and fall into a few
broad subjects (shoes, training, food and drink, injuries) so there is
something for the tool to group.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_input.py

Then choose "Batch Upload (CSV)" in the app, upload demo_articles.csv and
choose the content column.
"""

import csv

SITE = "https://www.yourstore.example/blog"

ARTICLES = [
    ("how-to-choose-running-shoes",
     "Choosing running shoes starts with where you run. Road shoes have smooth soles and plenty of cushioning, "
     "while trail shoes have deeper lugs for grip on mud. Measure both feet in the afternoon, when they are at "
     "their largest, and leave a thumb's width in front of your longest toe. A gait check on a treadmill shows "
     "whether a neutral or a stability shoe suits you."),
    ("when-to-replace-running-shoes",
     "Most running shoes last between 300 and 500 miles. The midsole foam compresses long before the upper "
     "wears out, so look for creases in the foam, uneven wear on the outsole and new aches after easy runs. "
     "Rotating two pairs gives the foam time to recover and makes both pairs last longer."),
    ("trail-shoes-vs-road-shoes",
     "Trail running shoes differ from road shoes in three ways: deeper lugs for grip, a rock plate to protect "
     "the foot, and a tougher upper. On hard-packed paths a road shoe is fine. On wet grass, mud and loose "
     "stone, a trail shoe gives far more grip and confidence on descents."),
    ("couch-to-5k-week-by-week",
     "A couch to 5k plan takes a new runner from walking to running for 30 minutes in nine weeks. Week one "
     "alternates 60 seconds of running with 90 seconds of walking. Each week the running intervals get longer. "
     "Three sessions a week with a rest day between them is enough, and repeating a week is normal."),
    ("first-10k-training-plan",
     "Training for a first 10k takes about eight weeks for someone who can already run 5k. The plan has three "
     "runs a week: one easy run, one run with faster intervals and one long run that grows by a kilometre each "
     "week. Keep the easy runs slow enough to hold a conversation."),
    ("marathon-long-runs-explained",
     "The long run is the most important session in marathon training. It builds endurance and teaches the body "
     "to burn fat for fuel. Most plans build to a longest run of 20 miles three weeks before race day, then "
     "taper. Run it at an easy pace and practise taking on food and drink."),
    ("what-to-eat-before-a-run",
     "Eat a meal rich in carbohydrate two to three hours before a run: porridge, toast or a bagel work well. "
     "Avoid a lot of fat and fibre, which sit heavily in the stomach. For an early run, a banana 30 minutes "
     "beforehand is enough. Drink water through the morning instead of a large amount at once."),
    ("energy-gels-and-hydration-on-long-runs",
     "On runs longer than 90 minutes the body needs extra carbohydrate. Take an energy gel every 30 to 45 "
     "minutes with a few mouthfuls of water. In hot weather add electrolyte tablets to replace the salt lost "
     "in sweat. Practise in training so there are no surprises on race day."),
    ("shin-splints-causes-and-recovery",
     "Shin splints cause pain along the inner edge of the shin bone and usually follow a sudden jump in "
     "mileage. Rest from running for one to two weeks, ice the area and keep fit with cycling or swimming. "
     "Return gradually, check that your shoes are not worn out and add calf strengthening exercises."),
    ("runners-knee-what-to-do",
     "Runner's knee is a dull ache around the kneecap that gets worse on stairs and downhill. Weak hips and "
     "quads are common causes. Cut back on mileage, avoid steep descents and strengthen the glutes and quads "
     "with squats and side leg raises. See a physiotherapist if it lasts more than a few weeks."),
]

with open("demo_articles.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "content"])
    for slug, text in ARTICLES:
        writer.writerow([f"{SITE}/{slug}", text])

print(f"Wrote demo_articles.csv: {len(ARTICLES)} articles")
