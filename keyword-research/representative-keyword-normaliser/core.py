"""
Representative Keyword Normaliser - Core Logic
Suggest a cleaner, more descriptive representative keyword for each keyword in a list,
using any OpenAI-compatible endpoint (local Ollama by default, or the OpenAI API).

Useful as a pre-processing step before SERP clustering or semantic clustering.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import json
import logging
import re

import pandas as pd
from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_fixed

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_API_KEY = "not-needed"  # Local servers such as Ollama do not require a key

# OpenAI models offered by every tool in this repository, for use with
# https://api.openai.com/v1. The local default model above is unchanged.
OPENAI_MODELS = ["gpt-6-luna", "gpt-6.1-terra"]

SYSTEM_PROMPT = (
    "You are a keyword research assistant. "
    "Provide only the requested keyword in the specified JSON format."
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "keyword_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"}
            },
            "required": ["keyword"],
            "additionalProperties": False
        }
    }
}


class UnableToParseJSONError(Exception):
    pass


def reasoning_params(model, effort="low"):
    """Extra request parameters for OpenAI's gpt-6 reasoning models.

    gpt-6 models take reasoning_effort and do not accept temperature. Local
    and other OpenAI-compatible servers may reject reasoning_effort, so it is
    only sent when the model name starts with "gpt-6".
    """
    if str(model).strip().lower().startswith("gpt-6"):
        return {"reasoning_effort": effort}
    return {}


def build_client(base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY):
    """Create an OpenAI-compatible client for the given endpoint."""
    return OpenAI(base_url=base_url, api_key=api_key)


def clean_keyword(keyword):
    """Strip characters outside printable ASCII, Arabic and Devanagari ranges."""
    return re.sub(r'[^\x20-\x7E\u0600-\u06FF\u0900-\u097F]+', '', str(keyword))


def parse_json_response(response_content):
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON. Raw response: {response_content}")
        raise UnableToParseJSONError("Unable to parse JSON response")


# Errors that will not go away on a second try: bad key, unknown model, rejected request.
PERMANENT_ERRORS = (AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2),
       retry=retry_if_not_exception_type(PERMANENT_ERRORS), reraise=True)
def request_keyword(keyword, client, model=DEFAULT_MODEL):
    """One request to the model. Raises on failure so the retry decorator can try again."""
    cleaned_keyword = clean_keyword(keyword)
    prompt = (
        f"Suggest an accurate and descriptive representative keyword for: {cleaned_keyword}. "
        "Pretend you are doing a Google search, what keyword would best describe the page "
        "based on the existing keyword? OUTPUT IN THE SAME LANGUAGE AS THE SOURCE."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=RESPONSE_FORMAT,
        **reasoning_params(model),
    )

    response_content = completion.choices[0].message.content
    logging.info(f"Raw API response for '{keyword}': {response_content}")
    response_json = parse_json_response(response_content)
    return response_json['keyword']


def process_keyword(keyword, client, model=DEFAULT_MODEL):
    """Ask the model for a representative keyword.

    Invalid replies and temporary API errors are retried (3 attempts, 2 seconds apart).
    If every attempt fails, or the error is permanent (bad key, unknown model), the row
    gets an "Error: ..." value and the run carries on with the next keyword.
    """
    try:
        return request_keyword(keyword, client, model)
    except Exception as e:
        logging.error(f"Error processing '{keyword}': {str(e)}")
        return f"Error: {str(e)[:100]}..."


def process_dataframe(df, client, model=DEFAULT_MODEL, column=None, progress_callback=None):
    """Add a 'Suggested Keyword' column to a DataFrame of keywords.

    column: name of the keyword column. Defaults to the first column.
    progress_callback: optional callable(done, total) for progress reporting.
    """
    keyword_column = column if column else df.columns[0]
    if keyword_column not in df.columns:
        raise ValueError(f"Column '{keyword_column}' not found in input file")

    suggestions = []
    total = len(df)
    for i, keyword in enumerate(df[keyword_column]):
        suggestions.append(process_keyword(keyword, client, model=model))
        if progress_callback:
            progress_callback(i + 1, total)

    df = df.copy()
    df['Suggested Keyword'] = suggestions
    return df


def process_csv(input_file, output_file, client, model=DEFAULT_MODEL, column=None,
                progress_callback=None):
    """Read keywords from a CSV, append suggestions and write the result to a CSV."""
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df = process_dataframe(df, client, model=model, column=column,
                           progress_callback=progress_callback)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"Results successfully written to {output_file}")
    return df
