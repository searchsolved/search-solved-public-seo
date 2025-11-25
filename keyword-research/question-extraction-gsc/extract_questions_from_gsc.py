####################################################################################
#                                                                                  #
#  Question Extraction from Google Search Console                                  #
#                                                                                  #
#  Extract question-type keywords from GSC data using pattern matching.            #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://x.com/LeeFootSEO                                              #
####################################################################################

"""
Question Extraction from GSC

Extracts question-type keywords from Google Search Console data.
Uses pattern matching to identify queries that represent user questions.

Setup:
    1. Install google-searchconsole: pip install git+https://github.com/joshcarty/google-searchconsole
    2. Create OAuth credentials in Google Cloud Console
    3. Save credentials as client_secrets.json

Usage:
    1. Update DOMAIN variable with your GSC property
    2. Update credentials paths
    3. Run the script

Requirements:
    pip install pandas searchconsole
"""

import os
import string
import pandas as pd

# Try to import searchconsole, provide helpful error if not installed
try:
    import searchconsole
except ImportError:
    print("searchconsole not installed. Install with:")
    print("pip install git+https://github.com/joshcarty/google-searchconsole")
    exit(1)

# Configuration
DOMAIN = "https://example.com/"  # Your GSC property URL
DAYS = 360  # Number of days to look back
OUTPUT_FILE = os.path.join(os.getcwd(), 'questions.csv')

# Credentials paths - update these to your credential locations
CLIENT_SECRETS_PATH = os.path.join(os.getcwd(), 'client_secrets.json')
CREDENTIALS_PATH = os.path.join(os.getcwd(), 'credentials.json')

# Question pattern for matching
# Strict matching - lower risk of false positives
QUESTION_PATTERN = r'\b(?:who|what|when|where|why|how|are|do|did|can|will|is|am|should|may|might|' \
                   r'adjusting|cutting|measuring|weight|height|depth|installing|instalation|best|' \
                   r'types|type|vs|building a|regulations|changing|change|choose|choosing|cleaning|' \
                   r'converting|convert|cost|price|different|measure|measurement|do i|do you|size|' \
                   r'sizes|thickness|dimensions|meaning|definition|terminology|difference|fitting|' \
                   r'slating|tiling|insulating|putting up|draught proofing|fixing|repairing|hanging|' \
                   r'painting|mounting|replacing|resealing|sanding|sealing|trimming|adding|boarding|' \
                   r'laying|is it|making a|mixing|moving|putting|reduce|reducing|replace|rendering|' \
                   r'skimming|options|water proofing|waterproofing|calculating|calculator|alternative|' \
                   r'alternatives|substitute|capping off|planning permission|prevent|preventing|' \
                   r'pros and cons|recoating|re-coating|removing|repair|repointing|retiling|re-tiling|' \
                   r'aligning|welding|using|finishing|preparing|priming)\b'

# Alternative: Loose matching - higher risk of false positives
# QUESTION_PATTERN = r'(?i)(\bwhat\b|\bwho\b|\bwhom\b|\bwhose\b|\bwhere\b|\bwhen\b|\bwhy\b|\bhow\b|\bwhich\b|\bwhether\b|\bif\b|\bdo\b|\bdoes\b|\bdid\b|\bcould\b|\bcan\b|\bwill\b|\bwould\b)'


def extract_questions():
    """
    Main function to extract question keywords from GSC.
    """
    # Check if credentials exist
    if not os.path.exists(CLIENT_SECRETS_PATH):
        print(f"Error: Client secrets not found at {CLIENT_SECRETS_PATH}")
        print("Please download OAuth credentials from Google Cloud Console")
        return None

    # Authenticate with GSC
    print("Authenticating with Google Search Console...")
    account = searchconsole.authenticate(
        client_config=CLIENT_SECRETS_PATH,
        credentials=CREDENTIALS_PATH,
    )

    # Get property
    try:
        webproperty = account[DOMAIN]
    except KeyError:
        print(f"Error: Property {DOMAIN} not found in your GSC account")
        print("Available properties:", list(account))
        return None

    # Query GSC data
    print(f"Fetching data for last {DAYS} days...")
    df = webproperty.query.range('today', days=-DAYS).dimension('query').get().to_dataframe()

    print(f"Retrieved {len(df)} queries from GSC")

    # Create dataframe for questions
    questions = pd.DataFrame(columns=['query'])

    # Filter rows containing question marks
    question_df = df[df['query'].str.contains('\?', na=False)]
    questions = pd.concat([questions, question_df])

    # Filter rows matching question patterns
    df = df[df['query'].str.contains(QUESTION_PATTERN, na=False, regex=True)]

    # Reset index
    df = df.reset_index(drop=True)

    # Clean up queries
    punctuation_pattern = f"[{string.punctuation}]"
    df['query'] = df['query'].str.replace(punctuation_pattern, '', regex=True)
    df = df.sort_values(by="impressions", ascending=False)
    df['query'] = (df['query'].str.split()).str.join(' ')
    df.drop_duplicates(subset=['query'], keep="first", inplace=True)

    print(f"\nFound {len(df)} question-type queries")
    print(df.head(20))

    # Export to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    extract_questions()
