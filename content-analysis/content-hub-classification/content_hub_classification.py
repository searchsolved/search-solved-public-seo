####################################################################################
#                                                                                  #
#  Content Hub Classification                                                      #
#                                                                                  #
#  Classify article content into content hub categories using OpenAI GPT.          #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""
Content Hub Classification

Analyzes article content using OpenAI's API to classify it into content hub
categories. Extracts primary topic, key subtopics, and recommended products.

**Important Security Notice:**
Never hardcode your API key in scripts that may be shared or stored in version
control systems. Use environment variables or secure storage solutions.

Setup:
    1. Set OPENAI_API_KEY environment variable
    2. Install requirements: pip install openai pandas

Usage:
    python content_hub_classification.py

Requirements:
    pip install openai pandas
"""

import json
import os
import pandas as pd
from openai import OpenAI

# Configuration
MODEL = "gpt-4o-mini"  # OpenAI model to use
SAVE_PATH = os.path.join(os.getcwd(), 'content_analysis_output.csv')

# Get API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    print("\nTo set it:")
    print("  Linux/Mac: export OPENAI_API_KEY='your-api-key'")
    print("  Windows: set OPENAI_API_KEY=your-api-key")
    print("\nOr in Python before running:")
    print("  import os")
    print("  os.environ['OPENAI_API_KEY'] = 'your-api-key'")
    exit(1)

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)


def analyze_article(article_text):
    """
    Analyzes the given article using OpenAI's API and returns the structured content analysis.

    Parameters:
    - article_text (str): The text of the article to analyze.

    Returns:
    - dict: The JSON response from the API containing the content analysis.
    """
    # Define the messages for the API call with the optimized prompt
    messages = [
        {
            "role": "system",
            "content": (
                "Analyze the following article and provide a structured summary based on the specified JSON schema. "
                "Select the most specific and relevant single content hub category that directly relates to the article's primary topic. "
                "Avoid broad or general categories. Use UK English."
            )
        },
        {
            "role": "user",
            "content": f"Analyze this article:\n\n{article_text}"
        }
    ]

    # Define the strict response format using the updated JSON schema
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "content_analysis_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "content_analysis": {
                        "type": "object",
                        "properties": {
                            "primary_topic": {
                                "type": "string"
                            },
                            "content_hub_category": {
                                "type": "string"
                            },
                            "key_subtopics": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "recommended_products": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": [
                            "primary_topic",
                            "content_hub_category",
                            "key_subtopics",
                            "recommended_products"
                        ],
                        "additionalProperties": False
                    }
                },
                "required": ["content_analysis"],
                "additionalProperties": False
            }
        }
    }

    # Make the API call
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format=response_format
        )

        # Parse and return the response
        response_content = completion.choices[0].message.content
        return json.loads(response_content)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'completion' in locals():
            print(f"Raw response: {completion}")
        return None


def save_analysis_to_dataframe(analysis_json, save_path):
    """
    Converts the JSON analysis to a pandas DataFrame and saves it as a CSV file.

    Parameters:
    - analysis_json (dict): The JSON output from the analyze_article function.
    - save_path (str): The full path where the CSV file will be saved.
    """
    # Extract the 'content_analysis' part of the JSON
    content_analysis = analysis_json.get("content_analysis", {})

    if not content_analysis:
        print("No content analysis data found.")
        return

    # Convert lists to JSON strings to preserve list structure in CSV
    content_analysis['key_subtopics'] = json.dumps(content_analysis.get('key_subtopics', []))
    content_analysis['recommended_products'] = json.dumps(content_analysis.get('recommended_products', []))

    # Create a DataFrame
    df = pd.json_normalize(content_analysis)

    # Save the DataFrame as a CSV file
    try:
        df.to_csv(save_path, index=False)
        print(f"DataFrame successfully saved to {save_path}")
    except Exception as e:
        print(f"Failed to save DataFrame: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Example article - replace with your own content
    article = """
    A Complete Guide to Sensor Cable and Connectors

    A guide to sensor connectors and sensor and actuator cables, discussing their
    uses, working, and the different types.

    What are Sensor Connectors?

    Sensor connectors are primarily used in conjunction with sensor cables, actuator
    cables, and switch cables for a variety of different industrial applications.
    They are quite versatile connectors, typically available in a wide range of
    different materials, sizes and lengths to suit a selection of industrial
    components and applications.

    They are used to connect sensing devices and components such as proximity sensors,
    photoelectric sensors, ultrasonic sensors, and current transducers, providing a
    durable and secure connection.
    """

    print("Analyzing article content...")
    result = analyze_article(article)

    if result:
        # Print the JSON output
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))

        # Save the analysis to a DataFrame and then to a CSV file
        save_analysis_to_dataframe(result, SAVE_PATH)
