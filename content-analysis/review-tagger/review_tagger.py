"""
Review Tagger - Core Module

Uses OpenAI's GPT models to assign a one or two-word descriptive tag to each
review in a dataset. Reviews are sent in batches to keep token usage efficient
and the model returns a JSON mapping of row IDs to tags.

Author: Lee Foot
Website: https://leefoot.com
"""

import json
import time

import pandas as pd
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 25
DEFAULT_COLUMN = "Review"
MAX_RETRIES = 3
RETRY_DELAY = 5

SYSTEM_PROMPT = (
    "You are an expert review analyst. For each review you receive, return a "
    "single one or two-word tag that captures the primary topic of the review. "
    "Examples:\n"
    '- "Delivery was late" = "Delivery"\n'
    '- "Great build quality" = "Build Quality"\n'
    '- "Sizing was off" = "Sizing"\n'
    '- "Easy to install" = "Installation"\n'
    '- "Good value for money" = "Value"\n\n'
    "Reviews will be sent with an identifying ID which MUST be returned with "
    "the assigned tag. If a review covers multiple topics, choose the most "
    "dominant one. Output valid JSON only, in this format:\n\n"
    '{"<ID>": "<Tag>", "<ID>": "<Tag>", ...}\n\n'
    "Each tag must be one or two words maximum."
)


def tag_reviews(
    df,
    api_key,
    review_column=DEFAULT_COLUMN,
    model=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    progress_callback=None,
):
    """Tag each review in a dataframe with a one or two-word descriptive label.

    Sends reviews to the OpenAI API in batches, parses the JSON response, and
    returns the original dataframe with a new 'Tag' column.

    Args:
        df: Input dataframe containing a review text column.
        api_key: OpenAI API key.
        review_column: Name of the column containing review text.
        model: OpenAI model name.
        batch_size: Number of reviews to send per API call.
        progress_callback: Optional callable(processed_batches, total_batches)
            invoked after each batch is processed.

    Returns:
        A copy of the dataframe with a 'Tag' column populated.
    """
    if review_column not in df.columns:
        raise ValueError(f"Column '{review_column}' not found in input data")

    client = OpenAI(api_key=api_key)

    df = df.copy()
    df["Tag"] = pd.NA

    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for idx, row in batch_df.iterrows():
            review_data = {"ID": str(idx), "Review": str(row[review_column])}
            messages.append({"role": "user", "content": json.dumps(review_data)})

        tags = _call_with_retries(client, messages, model)

        for id_key, tag_value in tags.items():
            try:
                row_idx = int(id_key)
                if row_idx in df.index:
                    df.at[row_idx, "Tag"] = tag_value
            except (ValueError, KeyError):
                continue

        if progress_callback is not None:
            progress_callback(batch_idx + 1, total_batches)

    return df


def _call_with_retries(client, messages, model):
    """Call the OpenAI API with retry logic. Returns a dict of {ID: Tag}."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=60,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            return parsed

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Max retries reached: {e}")
                return {}

    return {}
