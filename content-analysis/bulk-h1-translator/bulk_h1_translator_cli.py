#!/usr/bin/env python3
"""
Bulk H1 Translator - CLI Version

Translate H1 headings to English in bulk using any OpenAI-compatible API.
Defaults to a local Ollama endpoint; set --base-url and OPENAI_API_KEY
to use OpenAI or another hosted provider.

Usage:
    # Local LLM (Ollama, default endpoint)
    python bulk_h1_translator_cli.py --input crawl.csv

    # OpenAI
    export OPENAI_API_KEY=your-key
    python bulk_h1_translator_cli.py --input crawl.csv \
        --base-url https://api.openai.com/v1 --model gpt-4o-mini

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)


# JSON schema response format
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation_response",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "translated_h1": {"type": "string"}
            },
            "required": ["translated_h1"]
        }
    }
}

SYSTEM_PROMPT = ("You are a translator that always responds with a valid JSON "
                 "object containing only the translated text.")


def create_translation_prompt(h1, language):
    """Build the translation prompt for a single row."""
    prompt = f"""Translate the following text from {language} to English.
    Maintain the original meaning and tone as closely as possible.
    If the text is already in English or empty, return it unchanged.

    H1 to translate: '{h1}'"""
    return prompt


def translate_h1(client, model, prompt, retries=3):
    """Call the API for a single prompt, retrying on failure."""
    last_error = "Error: Translation failed"
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500,
                response_format=RESPONSE_FORMAT
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            last_error = "Error: Invalid JSON response"
        except Exception as e:
            print(f"Error processing prompt: {e}")
            last_error = "Error: Translation failed"
        if attempt < retries - 1:
            time.sleep(1)
    return {"translated_h1": last_error}


def main():
    parser = argparse.ArgumentParser(
        description='Translate H1 headings to English using any '
                    'OpenAI-compatible API'
    )
    parser.add_argument('--input', required=True,
                        help='Input CSV with H1 and Language columns')
    parser.add_argument('--output', default='translated_h1s.csv',
                        help='Output CSV path (default: translated_h1s.csv)')
    parser.add_argument('--h1-column', default='H1-1',
                        help='Name of the H1 column (default: H1-1, '
                             'the Screaming Frog export name)')
    parser.add_argument('--language-column', default='Language',
                        help='Name of the language column (default: Language)')
    parser.add_argument('--base-url', default='http://localhost:11434/v1',
                        help='OpenAI-compatible endpoint '
                             '(default: http://localhost:11434/v1 for Ollama; use http://localhost:1234/v1 for LM Studio)')
    parser.add_argument('--model', default='local-model',
                        help='Model name (default: local-model; Ollama uses '
                             'whichever model is loaded, for OpenAI use e.g. '
                             'gpt-4o-mini)')
    parser.add_argument('--retries', type=int, default=3,
                        help='Attempts per H1 before recording an error '
                             '(default: 3)')

    args = parser.parse_args()

    # API key from the environment; optional for local endpoints
    api_key = os.environ.get('OPENAI_API_KEY', 'lm-studio')

    # Initialise the client
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    # Load data
    print(f"Loading: {args.input}")
    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding='latin-1')

    # Ensure required columns exist
    for col in [args.h1_column, args.language_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file")

    print(f"  Loaded {len(df):,} rows")
    print(f"  Endpoint: {args.base_url}")
    print(f"  Model: {args.model}")

    # Build a translation prompt for each row
    def build_prompt(row):
        h1 = row[args.h1_column] if pd.notna(row[args.h1_column]) else ''
        language = row[args.language_column]
        return create_translation_prompt(h1, language)

    df['Translation Prompt'] = df.apply(build_prompt, axis=1)

    # Translate each prompt with a progress bar
    tqdm.pandas(desc="Translating")
    df['translation_result'] = df['Translation Prompt'].progress_apply(
        lambda prompt: translate_h1(client, args.model, prompt,
                                    retries=args.retries)
    )

    # Extract the translated texts from the JSON responses
    df['translated_h1'] = df['translation_result'].apply(
        lambda x: x.get('translated_h1', '')
    )

    # Remove the temporary columns used for processing
    df = df.drop(columns=['Translation Prompt', 'translation_result'])

    # Save the results, including all original columns
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    error_count = df['translated_h1'].astype(str).str.startswith('Error:').sum()

    print(f"\nResults saved to: {args.output}")
    print(f"  Rows translated: {len(df) - error_count}")
    if error_count:
        print(f"  Errors: {error_count}")


if __name__ == '__main__':
    main()
