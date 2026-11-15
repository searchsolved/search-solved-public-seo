# Content Block Extractor

**By Lee Foot** [leefoot.com](https://www.leefoot.com)

Uses Claude to identify the major content blocks on a set of pages (hero sections, feature grids, carousels and so on), gives an XPath for each, and counts how often each XPath appears across the pages. Useful for template analysis and for setting up a scraper.

There are two ways to run it. Both use the same code (`content_blocks_core.py`) and produce the same output.

| | File | Best for |
|---|---|---|
| Streamlit app | `extract_content_blocks_app.py` | A handful of pages, in a browser |
| Command line | `extract_content_blocks.py` | Long URL lists, with progress saved as it goes |

## How it works

1. Fetches each page.
2. Removes `script`, `style`, `noscript`, `meta`, `link`, `header`, `footer` and `nav` elements. Blocks inside the header, footer or nav are therefore never returned.
3. Sends the remaining HTML to Claude Haiku 5.5 (`claude-haiku-5-5`) at medium effort, cut at 200,000 characters if the page is longer. The command line script can use `claude-sonnet-5-5` instead.
4. Reads the JSON array of blocks from the reply. Entries without an XPath are dropped.
5. Counts how many rows share each XPath and gives each XPath one name (the first one seen).

The XPaths are written by the model and are not tested against the page. Check them before relying on them.

## Requirements

- Python 3.9 or later
- An Anthropic API key
- `pip install -r requirements.txt`

## Streamlit app

```bash
streamlit run extract_content_blocks_app.py
```

Enter your Anthropic API key in the sidebar, then enter one URL, paste several, or upload a CSV and pick the URL column. The app shows the blocks, a chart of the most common XPaths, and a reference table with one row per XPath. URLs that fail are listed with the error.

## Command line

```bash
export ANTHROPIC_API_KEY=your-key
python extract_content_blocks.py --input urls.txt --output-dir output
```

`urls.txt` holds one URL per line.

| Option | Default | What it does |
|---|---|---|
| `--input` | required | Text file with one URL per line |
| `--output-dir` | `output` | Folder for the CSV files |
| `--limit` | all | Process only the first N URLs. Use `--limit 2` for a test run |
| `--save-every` | 50 | Save a batch file every N rows |
| `--delay` | 1 | Seconds to wait before each page fetch |
| `--model` | `claude-haiku-5-5` | Claude model: `claude-haiku-5-5` or `claude-sonnet-5-5` |
| `--max-html-chars` | 200000 | Cut the filtered HTML at this many characters |

The key is read from the `ANTHROPIC_API_KEY` environment variable. It is never stored in the script.

## Output

One row per block per page.

| Column | Description |
|---|---|
| `url` | Source URL |
| `name` | Name of the content block |
| `xpath` | XPath for the element |
| `notes` | Brief description |

Files:

- `content_blocks.csv` (app) or `content_blocks_<timestamp>_<id>.csv` batch files (command line): the four columns above.
- `combined_output_with_frequency.csv` (both): the same rows with a `frequency` column, one name per XPath, sorted by frequency. `frequency` is the number of rows that share the XPath across the pages in this run.

The command line script combines only the rows from the current run. Batch files left in the folder by an earlier run are not counted.

## Cost

One API call per page. The cost depends on page size. Claude Haiku 5.5 is priced at $0.10 per million input tokens and $0.50 per million output tokens when the prompt is 100,000 tokens or fewer, and $0.50 and $2.50 above that. The model's thinking is billed as output tokens. Run `--limit 2` first to see token counts for your own pages.

## Errors

A page that fails to download, or a reply with no usable JSON, is reported and skipped; the run continues. The command line script lists the failed URLs at the end.

## Notes

- A one second delay runs before each page fetch. Raise `--delay` for sites that need it, and respect robots.txt and each site's terms.
- Pages that build their content with JavaScript will return little, because the HTML is fetched without a browser.

## License

MIT License.
