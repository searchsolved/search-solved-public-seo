import pandas as pd
import streamlit as st
st.set_page_config(page_title="Device, Location & Day of Week / Time of Day Bid Optimiser", page_icon="🔢",
                   layout="wide")
import chardet

st.title("Device, Location & Day of Week / Time of Day Bid Optimiser")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")


with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Calculates optimal bid adjustments for Google Ads
    - Analyzes Device, Location, Time of Day, or Day of Week performance
    - Recommends adjustments based on cost per conversion data

    **How to use:**
    1. Go to Google Ads > Reports
    2. Create a report with Device, Location, Time, or Day breakdown
    3. **Required columns:** Cost / conv. and Bid adj.
    4. Export as CSV and upload below
    5. Download the recommended bid adjustments

    **How it works:**
    - Compares each segment's Cost/Conv to the average
    - Calculates percentage adjustment needed
    - Positive adjustment = segment converts cheaper than average
    - Negative adjustment = segment converts more expensive

    **Best for:**
    - Optimizing bids by device type
    - Geographic bid adjustments
    - Day-parting optimization
    - Improving ROAS across segments
    """)

uploaded_file = st.file_uploader(
    "Upload your Device, Time of Day / Day of Week or Location Report", help="""Make sure to include Cost / conv. and Bid Adj. columns!""")

if uploaded_file is not None:

    result = chardet.detect(uploaded_file.getvalue())
    encoding_value = result["encoding"]

    if encoding_value == "UTF-16":
        white_space = True
    else:
        white_space = False

    df = pd.read_csv(
        uploaded_file,
        encoding=encoding_value,
        delim_whitespace=white_space,
        encoding_errors='ignore',
        on_bad_lines='skip',
        skiprows=2,
        dtype='str',
    )

    if len(df) == 0:
        st.caption("Your sheet seems empty!")

else:
    st.stop()

# sanity check for bid adjustment column and Cost / conv. column
if 'Cost / conv.' not in df.columns:
    st.info("Cost / conv. column not found! Please make sure it is exported and try again!")
    st.stop()

if 'Bid adj.' not in df.columns:
    print("Not found!")
    st.info("Bid adj. column not found! Please make sure it is exported and try again!!")
    st.stop()

# count the number of Total: columns and use to slice df to remove them
total = df.apply(lambda row: row.astype(str).str.contains('Total: ').any().sum(), axis=1)
total = total.sum()
campaign_total = df['Cost / conv.'].iloc[len(df) - total]  # get the campaign cost
df = df[:len(df) - total]

# check for strings in the bid adj column
df['Bid adj.'] = df['Bid adj.'].apply(lambda x: x.replace("--", "0"))
df['Bid adj.'] = df['Bid adj.'].apply(lambda x: x.replace("%", ""))

with st.expander("Click to View Uploaded Dataframe."):
    st.write(df)

# convert columns for calculations
campaign_total = float(campaign_total)
df['Cost / conv.'] = df['Cost / conv.'].astype(float)

# calculate the base adjustment is starting from 0
df['Adjustment'] = (((campaign_total / df['Cost / conv.'])-1) * 100).round(0)
df['Adjustment'] = df['Adjustment'].fillna(0)
df['Adjustment'] = df['Adjustment'].astype(str).str.split(".").str[0]
df['Adjustment'] = df['Adjustment'].replace("inf", 0)
st.write(df)
# cast columns back to int for calculations
df['Bid adj.'] = df['Bid adj.'].astype(int)
df['Adjustment'] = df['Adjustment'].astype(int)

# set the New Final Adjustment
df['Temp'] = df['Bid adj.'] * (df['Adjustment'] / 100)
df['New Final Adjustment'] = df['Bid adj.'] + df['Temp']
df['New Final Adjustment'] = df['New Final Adjustment'].astype(str).str.split(".").str[0]

# cast columns back to str for comparison
df['Bid adj.'] = df['Bid adj.'].astype(str)
df['Adjustment'] = df['Adjustment'].astype(str)

# drop rows if no changes
df['equal'] = df['Bid adj.'] == df['New Final Adjustment']  # compare the rows for equality, if no change - drop rows
df = df[~df["equal"].isin([True])]
st.write(df)

# delete the helper cols
del df['Temp']
del df['Adjustment']
del df['equal']

if df.empty:
    st.info("Congratulations, No Bids Require Adjusting!")
    st.stop()

def convert_df(df):  # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="📥 Download Stacked Bid Adjustments",
    data=csv,
    file_name='recommended_bid_adjustments.csv',
    mime='text/csv')
