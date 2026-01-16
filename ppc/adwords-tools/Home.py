import streamlit as st

st.set_page_config(
    page_title="AdWords Tool Set",
    page_icon="🌠",
)

st.write("# AdWords Tool Set")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Extract MPNs, Brands, Keyphrases, Automatic Bid Adjustments and More!

    **Tools included:**
    - **MPN Extractor**: Extract MPNs, brands and keyphrases from Google Ads Search Term Reports
    - **Bid Calculator**: Calculate optimal bid adjustments for device, location and time of day

    ---
    *Created by [Lee Foot](https://leefoot.com)*
    """
)
