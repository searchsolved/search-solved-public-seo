# Striking Distance Creator V2

Find keywords ranking in positions 4-20 and automatically check if they appear in the page title, H1, and body copy. Identify quick wins for on-page optimization.

## Features

- Upload keyword reports from Ahrefs, SEMrush, etc.
- Filter by position range and search volume
- Automatic content extraction from live URLs
- Checks keyword presence in title, H1, and copy
- Groups keywords by URL with top 5 opportunities
- Exports actionable recommendations

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the app:**
   ```bash
   streamlit run striking_distance_v2.py
   ```

2. **Upload a keyword file** (CSV from Ahrefs/SEMrush)

3. **Map columns:**
   - Keyword column
   - URL column
   - Volume column
   - Position column

4. **Adjust filters:**
   - Min/Max position (default: 4-20)
   - Minimum search volume

5. **Download results**

## Input File Format

Your CSV should have columns for:
- Keyword
- URL (ranking page)
- Search volume
- Position

## Output

CSV file with columns:
- Current URL
- Title, H1 (extracted from live page)
- Striking Distance Volume (total)
- KWs in Striking Dist. (count)
- KW1-KW5 (top 5 keywords by volume)
- KW1-KW5 Vol (search volume)
- KW1-KW5 in Title (True/False)
- KW1-KW5 in H1 (True/False)
- KW1-KW5 in Copy (True/False)

## How It Works

1. **Filters keywords** in position range with minimum volume
2. **Groups by URL** keeping top 5 keywords per page
3. **Extracts content** from each URL (title, H1, body)
4. **Checks presence** of each keyword in page elements
5. **Removes fully optimized** pages (where keyword is in all elements)

## Interpreting Results

- **False in Title**: Add keyword to page title
- **False in H1**: Add keyword to H1 heading
- **False in Copy**: Add keyword to body content

Focus on keywords with high volume and missing from title/H1.

## Limitations

- Content extraction requires accessible URLs
- Some sites may block automated requests
- Run locally if you need to whitelist your IP

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)