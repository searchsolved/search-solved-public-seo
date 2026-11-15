# Sample data: Page Intent Classifier

The input is a list of ten public pages on python.org, chosen because the site has clearly different page types (download, documentation, donation, jobs, events). The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `demo_urls.txt` | Input: 10 URLs, one per line |
| `page_intent.csv` | Output: url, intent, action, category, error |

To reproduce:

    python ../page_intent_classifier_cli.py --api-key your-key --urls demo_urls.txt --output page_intent.csv --delay 2

All ten pages were fetched and classified. One result shows a limit of the tool: the donations page came back with the category "donate", which is not one of the eight categories the prompt asks for (signup, purchase, browse, learn, contact, compare, download, other). The list is requested in the prompt and not enforced in code, so check the category column for values outside it.

Output produced on 2026-11-15 with gpt-4o-mini (called through OpenRouter), in 10 API calls with a 2 second pause between pages, on branch fix/content-tools-wave0. The pages may have changed since.
