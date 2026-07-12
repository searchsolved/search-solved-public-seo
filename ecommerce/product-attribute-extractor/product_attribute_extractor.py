# Author: Lee Foot
# Website: https://leefoot.com
"""
Product Attribute Extractor - Core Module

Uses an OpenAI-compatible LLM to iteratively extract structured product
attributes (brand, colour, size, material, voltage, etc.) from product titles
and descriptions. New attributes are discovered progressively across the
catalogue and reused for consistency.

Author: Lee Foot
Website: https://leefoot.com
"""

import json
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Few-shot examples using generic e-commerce products
FEW_SHOT_EXAMPLES = [
    {
        "product_description": "Bosch Professional 18V Cordless Drill Driver GSR 18V-55",
        "attributes": {
            "product_type": "cordless drill driver",
            "brand": "Bosch Professional",
            "voltage": "18V",
            "product_range": "GSR 18V-55",
        },
    },
    {
        "product_description": "Karcher K4 Power Control Pressure Washer 1800W Yellow",
        "attributes": {
            "product_type": "pressure washer",
            "brand": "Karcher",
            "wattage": "1800W",
            "colour": "yellow",
            "product_range": "K4 Power Control",
        },
    },
    {
        "product_description": "Stanley FatMax 5m Tape Measure 32mm Blade Width",
        "attributes": {
            "product_type": "tape measure",
            "brand": "Stanley",
            "size": "5m",
            "blade_width": "32mm",
            "product_range": "FatMax",
        },
    },
    {
        "product_description": "DeWalt DCS391N 18V XR 165mm Circular Saw Body Only",
        "attributes": {
            "product_type": "circular saw",
            "brand": "DeWalt",
            "voltage": "18V",
            "blade_diameter": "165mm",
            "product_range": "XR",
            "feature": "body only",
        },
    },
]


def _build_examples_string():
    """Format few-shot examples for inclusion in the system prompt."""
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"Product: {ex['product_description']}")
        lines.append(f"Attributes: {json.dumps(ex['attributes'])}")
    return "\n".join(lines)


def build_system_prompt(known_attributes):
    """
    Construct the system prompt incorporating known attributes and few-shot
    examples. The prompt instructs the model to reuse existing attribute names
    where possible before creating new ones.
    """
    examples_str = _build_examples_string()
    known_list = ", ".join(sorted(known_attributes)) if known_attributes else "(none yet)"

    prompt = (
        "You are a product data specialist. Extract all product attributes from the "
        "given product text and return them as a flat JSON object.\n\n"
        "Rules:\n"
        "- Reuse attribute names from this known list where applicable: "
        f"{known_list}\n"
        "- Only create a new attribute name if none of the known attributes fit.\n"
        "- Use lowercase_with_underscores for all attribute keys.\n"
        "- Always include 'product_type' even if you have to infer it.\n"
        "- Return ONLY valid JSON, no commentary.\n\n"
        f"Examples:\n{examples_str}\n\n"
        "Now process the following product text and return attributes as JSON."
    )
    return prompt


def safe_parse_json(json_str):
    """Attempt to parse a JSON string, returning an empty dict on failure."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        logger.error("Failed to parse JSON response: %s", json_str[:200])
        return {}


def validate_and_clean_attributes(attributes):
    """Normalise attribute keys to lowercase with underscores and deduplicate."""
    cleaned = {}
    for key, value in attributes.items():
        normalised_key = key.strip().lower().replace(" ", "_")
        if normalised_key not in cleaned or not cleaned[normalised_key]:
            cleaned[normalised_key] = value
    return cleaned


def extract_attributes(client, model, product_text, known_attributes):
    """
    Send product text to the LLM and extract structured attributes.

    Args:
        client: OpenAI client instance.
        model: Model identifier string.
        product_text: The product title/description to process.
        known_attributes: Set of attribute names discovered so far.

    Returns:
        Dictionary of cleaned attribute key-value pairs.
    """
    system_prompt = build_system_prompt(known_attributes)

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": product_text},
            ],
        )
        raw = safe_parse_json(response.choices[0].message.content)

        if "product_type" not in raw:
            raw["product_type"] = "unknown"

        return validate_and_clean_attributes(raw)

    except Exception as e:
        logger.error("Extraction failed for '%s': %s", product_text[:80], e)
        return {}


def create_client(api_key, base_url="https://api.openai.com/v1"):
    """Create an OpenAI client with the specified base URL."""
    return OpenAI(api_key=api_key, base_url=base_url)


def sort_columns_by_frequency(df):
    """Reorder DataFrame columns so the most populated appear first."""
    column_frequency = df.count().sort_values(ascending=False)
    return df[column_frequency.index]
