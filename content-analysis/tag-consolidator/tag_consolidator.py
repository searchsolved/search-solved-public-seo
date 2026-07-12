"""
Tag Consolidator - Core Module

Uses OpenAI's GPT models to consolidate granular secondary tags into broader,
generic categories. Tags are grouped by their primary tag and each group is
sent to the API in a single call, so related tags are consolidated together.

Expected input: a CSV with a primary tag column and a secondary tag column,
for example tags produced by a review-mining or feedback-classification
exercise ("delivery", "sizing", "build quality" and so on).

Author: Lee Foot
Website: https://www.leefoot.com
"""

import json

import pandas as pd
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PRIMARY_COLUMN = "Primary Tag"
DEFAULT_SECONDARY_COLUMN = "Secondary Tag"
GENERIC_TAG_COLUMN = "Generic Tag"


def create_prompt(tags):
    """Build the consolidation prompt for a list of secondary tags."""
    string_tags = [str(tag) for tag in tags]

    prompt = (
        "Please consolidate the following tags by grouping similar issues together: "
        + ", ".join(string_tags)
        + ". Respond ONLY with a valid JSON object where each key is a broader "
        "generic category name and each value is a list of the original tags "
        "that belong to it. Every tag must appear in exactly one category. "
        'Example format: {"Delivery": ["late delivery", "damaged in transit"], '
        '"Build Quality": ["flimsy material", "poor stitching"]}'
    )
    return prompt


def call_openai_api(client, prompt, model=DEFAULT_MODEL):
    """Call the OpenAI chat completions API and return the response."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        response_format={"type": "json_object"},
    )
    return chat_completion


def parse_response(result):
    """Parse the API response into a {generic category: [tags]} dict.

    Returns an empty dict if the response cannot be parsed.
    """
    response_content = result.choices[0].message.content
    try:
        parsed_json = json.loads(response_content)
    except (json.JSONDecodeError, TypeError):
        return {}

    mapping = {}
    for category, tags in parsed_json.items():
        if isinstance(tags, list):
            mapping[category] = [str(tag) for tag in tags]
        else:
            mapping[category] = [str(tags)]
    return mapping


def consolidate_tags(
    df,
    api_key,
    model=DEFAULT_MODEL,
    primary_column=DEFAULT_PRIMARY_COLUMN,
    secondary_column=DEFAULT_SECONDARY_COLUMN,
    progress_callback=None,
    checkpoint_callback=None,
):
    """Consolidate secondary tags into broader generic categories.

    Groups the dataframe by the primary tag column, sends each group's unique
    secondary tags to the OpenAI API, and writes the returned generic category
    for each tag into a new 'Generic Tag' column.

    Args:
        df: Input dataframe containing the primary and secondary tag columns.
        api_key: OpenAI API key.
        model: OpenAI model name.
        primary_column: Name of the primary tag column.
        secondary_column: Name of the secondary tag column.
        progress_callback: Optional callable(processed_groups, total_groups,
            group_name) invoked after each group is processed.
        checkpoint_callback: Optional callable(partial_df) invoked after each
            group with the rows processed so far, for incremental saving.

    Returns:
        A copy of the dataframe with the 'Generic Tag' column populated.
    """
    if primary_column not in df.columns:
        raise ValueError(f"Column '{primary_column}' not found in input data")
    if secondary_column not in df.columns:
        raise ValueError(f"Column '{secondary_column}' not found in input data")

    client = OpenAI(api_key=api_key)

    df = df.copy()
    df[GENERIC_TAG_COLUMN] = pd.NA

    temp_df = pd.DataFrame(columns=df.columns)

    grouped = df.groupby(primary_column)
    total_groups = len(grouped)

    processed_groups = 0
    for name, group in grouped:
        tags = group[secondary_column].unique()
        prompt = create_prompt(tags)

        try:
            result = call_openai_api(client, prompt, model=model)
            mapping = parse_response(result)
        except Exception:
            mapping = {}

        for category, mapped_tags in mapping.items():
            for tag in mapped_tags:
                df.loc[
                    (df[primary_column] == name)
                    & (df[secondary_column].astype(str) == tag),
                    GENERIC_TAG_COLUMN,
                ] = category

        temp_df = pd.concat([temp_df, df[df[primary_column] == name]])

        if checkpoint_callback is not None:
            checkpoint_callback(temp_df)

        processed_groups += 1
        if progress_callback is not None:
            progress_callback(processed_groups, total_groups, name)

    return df
