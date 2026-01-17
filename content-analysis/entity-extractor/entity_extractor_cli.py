#!/usr/bin/env python3
"""
Entity Extractor - CLI Version

Extract named entities from text using SpaCy NLP.

Usage:
    python entity_extractor_cli.py --input content.csv --output entities.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
from bs4 import BeautifulSoup
import sys

try:
    import spacy
except ImportError:
    print("Error: SpaCy is not installed. Install with: pip install spacy")
    print("Then download a model: python -m spacy download en_core_web_sm")
    sys.exit(1)


def clean_html(html_content):
    if pd.isna(html_content):
        return ""
    soup = BeautifulSoup(str(html_content), 'html.parser')
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'ol', 'ul', 'table']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)


def extract_entities(text, nlp_model, allowed_types):
    if not text or pd.isna(text):
        return []
    max_chars = 100000
    if len(text) > max_chars:
        text = text[:max_chars]
    doc = nlp_model(text)
    entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in allowed_types and ent.text.strip()]
    return entities


def main():
    parser = argparse.ArgumentParser(description='Extract named entities from text using SpaCy')
    parser.add_argument('--input', required=True, help='Input CSV with content')
    parser.add_argument('--output', default='entities.csv', help='Output CSV path')
    parser.add_argument('--content-col', default='content', help='Content column name')
    parser.add_argument('--id-col', help='Optional ID column (e.g., URL)')
    parser.add_argument('--model', default='en_core_web_sm', help='SpaCy model name')
    parser.add_argument('--entity-types', nargs='+', default=['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT'],
                        help='Entity types to extract')

    args = parser.parse_args()

    print(f"Loading SpaCy model: {args.model}")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Error: Model '{args.model}' not found. Download with: python -m spacy download {args.model}")
        sys.exit(1)

    print(f"Loading content from: {args.input}")
    df = pd.read_csv(args.input)

    # Find content column
    content_col = None
    for col in df.columns:
        if col.lower() == args.content_col.lower():
            content_col = col
            break
    if not content_col:
        content_col = df.columns[0]

    all_results = []
    total = len(df)

    for idx, row in df.iterrows():
        content = row[content_col]
        identifier = row[args.id_col] if args.id_col and args.id_col in df.columns else idx

        cleaned = clean_html(content)
        entities = extract_entities(cleaned, nlp, args.entity_types)

        for entity, label in entities:
            all_results.append({
                'Source': identifier,
                'Entity': entity,
                'Label': label
            })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{total} rows...")

    if all_results:
        df_results = pd.DataFrame(all_results)

        # Aggregate counts
        entity_counts = df_results.groupby(['Entity', 'Label']).size().reset_index(name='Count')
        entity_counts = entity_counts.sort_values('Count', ascending=False)
        entity_counts.to_csv(args.output, index=False, encoding='utf-8-sig')

        print(f"\nResults saved to: {args.output}")
        print(f"  Total entities found: {len(df_results)}")
        print(f"  Unique entities: {len(entity_counts)}")
    else:
        print("No entities found in the content.")
        sys.exit(1)


if __name__ == '__main__':
    main()
