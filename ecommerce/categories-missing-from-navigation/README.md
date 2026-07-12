# Categories Missing From Navigation

## Features

Find sitemap URLs that are not linked from your site navigation. Useful for spotting category pages with no navigation links after a redesign, or orphaned sections of large e-commerce sites.

- Streamlit web interface + CLI version
- Fetches XML sitemaps, including sitemap index files
- Extracts links from any navigation element via a CSS selector
- Resolves relative links to absolute URLs before comparing
- Optional URL-contains filter (for example `/category/`)
- Polite fetching: real user agent and a delay between requests
- Export missing URLs to CSV

## Usage

### Streamlit App

```
streamlit run categories_missing_from_navigation_app.py
```

Enter your sitemap URL (for example `https://www.example.com/sitemap.xml`), the CSS selector for your navigation element (for example `nav`, `#main-nav` or `.header-menu`), and optionally a URL filter.

### CLI

```
python categories_missing_from_navigation_cli.py \
    --sitemap https://www.example.com/sitemap.xml \
    --nav-selector "#main-nav" \
    --page https://www.example.com/ \
    --filter /category/ \
    --output missing_from_navigation.csv
```

| Argument | Description |
| --- | --- |
| `--sitemap` | XML sitemap URL (required; sitemap index files are supported) |
| `--nav-selector` | CSS selector for the navigation element (default: `nav`) |
| `--page` | Page whose navigation will be checked (default: homepage of the sitemap domain) |
| `--filter` | Only report missing URLs containing this string |
| `--delay` | Delay in seconds between fetches (default: 2) |
| `--output` | Output CSV path (default: `missing_from_navigation.csv`) |

## Notes

- The tool reads raw HTML, so navigation rendered with JavaScript will not be detected.
- URL comparison is exact, so make sure your sitemap and navigation use consistent URL formats (trailing slashes, www, protocol).
- To find your navigation selector, right-click the navigation in your browser, choose Inspect, and note the element's tag, id or class.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
