"""
Demo input for the Content Extractor.

Writes a small demo site to demo_site/ (10 HTML pages) and demo_urls.csv
with a url column pointing at those pages on http://localhost:8765. The
articles are the made-up running blog articles from the Content Hub
Classifier sample. Each page has the parts a real page has, so the output
shows what the tool removes and what it keeps: a script, a style block, a
header with navigation, the article, a sidebar of related links and a
footer.

Author: Lee Foot
Website: https://www.leefoot.com

Usage:
    python make_demo_site.py
    python -m http.server 8765 --directory demo_site
    python ../content_extractor_cli.py --input demo_urls.csv --output extracted_content.csv
"""

import csv
import html
import os

SOURCE = os.path.join("..", "..", "content-hub-classification", "sample_data", "demo_articles.csv")
BASE = "http://localhost:8765"

TITLES = {
    "how-to-choose-running-shoes": "How to choose running shoes",
    "when-to-replace-running-shoes": "When to replace your running shoes",
    "trail-shoes-vs-road-shoes": "Trail shoes vs road shoes",
    "couch-to-5k-week-by-week": "Couch to 5k, week by week",
    "first-10k-training-plan": "Your first 10k training plan",
    "marathon-long-runs-explained": "Marathon long runs explained",
    "what-to-eat-before-a-run": "What to eat before a run",
    "energy-gels-and-hydration-on-long-runs": "Energy gels and hydration on long runs",
    "shin-splints-causes-and-recovery": "Shin splints: causes and recovery",
    "runners-knee-what-to-do": "Runner's knee: what to do",
}

PAGE = """<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>{title} | Your Store Blog</title>
<style>
  body {{ font-family: sans-serif; max-width: 60em; margin: 0 auto; }}
  aside {{ border-top: 1px solid #ccc; }}
</style>
<script>
  window.dataLayer = window.dataLayer || [];
  window.dataLayer.push({{ pageType: "article", slug: "{slug}" }});
</script>
</head>
<body>
<header>
  <p>Your Store Blog</p>
  <nav>
    <a href="/">Home</a>
    <a href="/shoes">Shoes</a>
    <a href="/training">Training</a>
    <a href="/nutrition">Nutrition</a>
    <a href="/injuries">Injuries</a>
  </nav>
</header>
<main>
  <article>
    <h1>{title}</h1>
{paragraphs}
  </article>
  <aside>
    <h2>Related guides</h2>
    <ul>
{related}
    </ul>
  </aside>
</main>
<footer>
  <p>Your Store Ltd. Free delivery on orders over 50 pounds. Sign up to the newsletter for training tips.</p>
</footer>
</body>
</html>
"""


def paragraphs(text):
    """Two sentences per paragraph, so each page has several."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    sentences = [s if s.endswith(".") else s + "." for s in sentences]
    groups = [" ".join(sentences[i:i + 2]) for i in range(0, len(sentences), 2)]
    return "\n".join(f"    <p>{html.escape(g)}</p>" for g in groups)


with open(SOURCE, encoding="utf-8", newline="") as f:
    articles = [(row["url"].rsplit("/", 1)[1], row["content"]) for row in csv.DictReader(f)]

os.makedirs("demo_site", exist_ok=True)
slugs = [slug for slug, _ in articles]
for i, (slug, text) in enumerate(articles):
    # Link each page to the next three, wrapping round at the end of the list.
    others = [slugs[(i + n) % len(slugs)] for n in (1, 2, 3)]
    related = "\n".join(f'      <li><a href="/{s}.html">{html.escape(TITLES[s])}</a></li>' for s in others)
    page = PAGE.format(title=html.escape(TITLES[slug]), slug=slug, paragraphs=paragraphs(text), related=related)
    with open(os.path.join("demo_site", f"{slug}.html"), "w", encoding="utf-8") as out:
        out.write(page)

with open("demo_urls.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url"])
    for slug in slugs:
        writer.writerow([f"{BASE}/{slug}.html"])

print(f"Wrote demo_site/ ({len(slugs)} pages) and demo_urls.csv")
