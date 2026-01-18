####################################################################################
#                                                                                  #
#  Google Algorithm Tracker                                                        #
#                                                                                  #
#  Scrape Google's Search Status page for algorithm updates.                       #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                                   #
# Contact  : https://www.leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Google Algorithm Tracker

Scrapes Google's Search Status page to get a list of algorithm updates.
Classifies updates by type (Core, Spam, Reviews, Helpful Content, etc.).
No API key required.

Features:
- Scrapes official Google Search Status page
- Classifies algorithm updates by type
- Filterable by date range and update type
- Export to CSV/Excel
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from io import BytesIO

st.set_page_config(page_title="Google Algorithm Tracker", page_icon="🔄", layout="wide")

st.title("Google Algorithm Tracker")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Scrapes the official Google Search Status page
    - Lists all algorithm updates with dates
    - Classifies updates by type (Core, Spam, Reviews, etc.)
    - No API key required

    **Data source:**
    [Google Search Status Dashboard](https://status.search.google.com/products/rGHU1u87FJnkP6W2GwMi/history)

    **Update types tracked:**
    - Core Updates
    - Spam Updates
    - Helpful Content Updates
    - Product Reviews Updates
    - Link Spam Updates
    - Page Experience Updates
    """)


def classify_update(summary):
    """Classify the type of algorithm update."""
    summary_lower = summary.lower()

    classifications = [
        ('core update', 'Core Update'),
        ('spam update', 'Spam Update'),
        ('helpful content update', 'Helpful Content Update'),
        ('helpful content system', 'Helpful Content Update'),
        ('product reviews update', 'Product Reviews Update'),
        ('reviews update', 'Reviews Update'),
        ('link spam update', 'Link Spam Update'),
        ('page experience update', 'Page Experience Update'),
        ('site reputation abuse', 'Site Reputation Abuse'),
        ('expired domain abuse', 'Expired Domain Abuse'),
        ('scaled content abuse', 'Scaled Content Abuse'),
    ]

    for keyword, update_type in classifications:
        if keyword in summary_lower:
            return update_type

    return 'Other'


def scrape_algorithm_updates():
    """Scrape Google Search Status page for algorithm updates."""
    url = 'https://status.search.google.com/products/rGHU1u87FJnkP6W2GwMi/history'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')

    all_data = []

    for table in tables:
        rows = table.find_all('tr')

        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                summary = cols[0].text.strip()
                date_text = cols[1].text.strip()

                # Parse date
                try:
                    date_obj = datetime.strptime(date_text, "%d %b %Y")
                    date_formatted = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    date_formatted = None
                    date_obj = None

                update_type = classify_update(summary)

                all_data.append({
                    'Date': date_formatted,
                    'Date_obj': date_obj,
                    'Summary': summary,
                    'Update Type': update_type
                })

    df = pd.DataFrame(all_data)

    # Sort by date descending
    df = df.sort_values('Date', ascending=False)

    return df


# Sidebar settings
st.sidebar.header("Filter Settings")

# Date filter
date_filter_option = st.sidebar.selectbox(
    "Date filter",
    ["All time", "Last 30 days", "Last 90 days", "Last year", "Custom range"]
)

start_date = None
end_date = None

if date_filter_option == "Custom range":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End date", datetime.now())

# Update type filter
update_types = [
    "All",
    "Core Update",
    "Spam Update",
    "Helpful Content Update",
    "Product Reviews Update",
    "Reviews Update",
    "Link Spam Update",
    "Page Experience Update",
    "Other"
]

selected_types = st.sidebar.multiselect(
    "Update types",
    update_types,
    default=["All"]
)

# Main content
st.subheader("Algorithm Updates")

if st.button("Fetch Algorithm Updates", type="primary"):
    with st.spinner("Scraping Google Search Status page..."):
        try:
            df = scrape_algorithm_updates()

            if df.empty:
                st.warning("No algorithm updates found")
            else:
                # Apply date filter
                if date_filter_option == "Last 30 days":
                    cutoff = datetime.now() - timedelta(days=30)
                    df = df[df['Date_obj'] >= cutoff]
                elif date_filter_option == "Last 90 days":
                    cutoff = datetime.now() - timedelta(days=90)
                    df = df[df['Date_obj'] >= cutoff]
                elif date_filter_option == "Last year":
                    cutoff = datetime.now() - timedelta(days=365)
                    df = df[df['Date_obj'] >= cutoff]
                elif date_filter_option == "Custom range" and start_date and end_date:
                    df = df[(df['Date_obj'] >= datetime.combine(start_date, datetime.min.time())) &
                            (df['Date_obj'] <= datetime.combine(end_date, datetime.max.time()))]

                # Apply type filter
                if "All" not in selected_types and selected_types:
                    df = df[df['Update Type'].isin(selected_types)]

                # Drop helper column
                df_display = df.drop(columns=['Date_obj'])

                # Store in session state
                st.session_state['algorithm_df'] = df_display

                st.success(f"Found {len(df_display)} algorithm updates")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display results if available
if 'algorithm_df' in st.session_state:
    df_display = st.session_state['algorithm_df']

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Updates", len(df_display))
    with col2:
        core_updates = len(df_display[df_display['Update Type'] == 'Core Update'])
        st.metric("Core Updates", core_updates)
    with col3:
        spam_updates = len(df_display[df_display['Update Type'] == 'Spam Update'])
        st.metric("Spam Updates", spam_updates)
    with col4:
        if len(df_display) > 0:
            latest = df_display.iloc[0]['Date']
            st.metric("Latest Update", latest)

    # Update type breakdown
    st.subheader("Update Type Breakdown")
    type_counts = df_display['Update Type'].value_counts()
    st.bar_chart(type_counts)

    # Full table
    st.subheader("All Updates")
    st.dataframe(df_display, use_container_width=True)

    # Download options
    st.subheader("Download")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df_display.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="google_algorithm_updates.csv",
            mime="text/csv"
        )

    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, sheet_name='Algorithm Updates', index=False)

            # Type summary sheet
            type_summary = df_display['Update Type'].value_counts().reset_index()
            type_summary.columns = ['Update Type', 'Count']
            type_summary.to_excel(writer, sheet_name='Type Summary', index=False)

        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name="google_algorithm_updates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Click 'Fetch Algorithm Updates' to load data from Google")

    st.subheader("Example Output")
    example_data = {
        "Date": ["2024-03-05", "2024-02-15", "2024-01-10"],
        "Summary": [
            "March 2024 core update",
            "February 2024 spam update",
            "January 2024 helpful content update"
        ],
        "Update Type": ["Core Update", "Spam Update", "Helpful Content Update"]
    }
    st.dataframe(pd.DataFrame(example_data))
