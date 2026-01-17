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

    ---

    Made by [@leefootseo](https://bsky.app/profile/leefootseo.bsky.social)
"""
)
