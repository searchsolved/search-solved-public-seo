####################################################################################
#                                                                                  #
#  Entity Extraction App                                                           #
#                                                                                  #
#  Extract entities from SERPs, CSV files, or YouTube transcripts.                 #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

import streamlit as st

st.set_page_config(
    page_title="Entity Extraction App",
    page_icon="mag",
)

st.write("# Entity Extraction App")

st.sidebar.success("Select a page above.")

st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

st.markdown(
    """
    This app uses the [dandelion.eu API](https://dandelion.eu/) to extract entities from:
    - Live SERPs via keyword search
    - CSV/keyword export files
    - YouTube video transcripts

    **API Requirements:**

    - **SERPs page**: Requires [ValueSERP API Key](https://www.valueserp.com/pricing) (PAYG pricing)
    - **All pages**: Requires free [Dandelion.eu API Key](https://dandelion.eu/) (1,000 free requests/day)

    **Select a page from the sidebar** to get started!
"""
)
