# Automatic Category Suggester Script - Updated 01/03/2024
### Twitter: @LeeFootSEO
### Web: https://leefoot.co.uk

This script merges two crawl files from Screaming Frog to suggest new landing pages to align invetory to search demand.
## Preparatory Steps

Before running this script, please follow these steps to prepare your data:

### Conducting a Crawl with Screaming Frog
1. Conduct a crawl of your website using Screaming Frog to gather necessary data.

### Preparing the Internal HTML Report
2. Export the "Internal HTML" report from Screaming Frog.
3. Open the exported file in Excel (or any compatible spreadsheet software) and add a column named "Page Type" to the data.
4. Manually tag each product page with 'Product Page' in the "Page Type" column.
5. Manually tag each category page with 'Category Page' in the "Page Type" column.

### Exporting and Preparing Inlinks Data
6. Select all product URLs in the crawl.
7. Right-click on the selection and choose 'Export Inlinks'.
8. Save the exported "Inlinks" file.

Place both the modified "Internal HTML" file and the "Inlinks" file in a designated folder. Update the file paths in the `main` function of this script to point to the location where you've stored these files.

## Installation

Ensure that the necessary third-party libraries (pandas, tqdm, torch, sentence-transformers, nltk) are installed in your environment. You can install these dependencies via pip:

```bash
pip install pandas tqdm torch sentence-transformers nltk
```

## Usage

Run the script after completing the data preparation steps. The script will process the provided data, performing operations such as filtering, n-gram generation, exact and partial match calculation, and fuzzy matching. The final results will be saved to a specified CSV file.

```python
python automatic_category_suggester.py
```

Replace `automatic_category_suggester.py` with the actual name of your Python script.

## Contributing

Contributions to improve the script are welcome. Please feel free to fork the repository and submit pull requests.

### Don't want to mess around with Python? Contact me about my managed service and let me take care of it! https://leefoot.co.uk/services/managed-service/
