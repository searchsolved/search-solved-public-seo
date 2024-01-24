####################################################################################
# Website  : https://leefoot.co.uk/                                                #
# Contact  : https://leefoot.co.uk/hire-me/                                        #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://twitter.com/LeeFootSEO                                        #
####################################################################################

import streamlit as st

st.set_page_config(page_title="Google Trends & NueralProphet - Explainable Trends at Scale", page_icon="🔎",
                   layout="wide")  # needs to be the first thing after the streamlit import

import chardet
from stqdm import stqdm

from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from pytrends.request import TrendReq
import pandas as pd
import xlsxwriter
import base64
import os
import time

set_random_seed(0)

st.write(
    "[![this is an image link](https://i.imgur.com/Ex8eeC2.png)](https://www.patreon.com/leefootseo) [Become a Patreon for Early Access, Support & More!](https://www.patreon.com/leefootseo)  |  Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://twitter.com/LeeFootSEO)")

st.title("Google Trends & Facebook Prophet Tool")

# streamlit variables
FORECAST_WEEKS = st.sidebar.text_input('Number of weeks to forecast', value=52)
LANGUAGE = st.sidebar.selectbox(
    "Select the host language to search Google Trends",
    (
        "en-GB",
        "en-US",
        "es",
        "pt-BR",
        "fr",
        "de",
        "it",
        "hi",
        "pl",
        "ro",
        "zh-CN",
        "sv",
        "tr",
        "cy",
        "no",
        "ja",
        "ua",
        "ru"
    ),
)
HISTORIC = st.sidebar.checkbox('Make historic predictions?', value=True)
uploaded_file = st.file_uploader("Upload your .csv list of keywords / Crawl file")
RETRIES = st.sidebar.text_input('Select the number of retries when scraping', value=3)
SLEEP_TIMER = st.sidebar.text_input('Select the sleep delay when scraping', value=5)

RETRIES = int(RETRIES)
FORECAST_WEEKS = int(FORECAST_WEEKS)
SLEEP_TIMER = int(SLEEP_TIMER)

pt = TrendReq(hl=LANGUAGE, timeout=(10, 25), retries=RETRIES, backoff_factor=0.5)

# set counters to slice keyword list in the loop
counter = 1
start = 0

if uploaded_file is not None:

    try:

        result = chardet.detect(uploaded_file.getvalue())
        encoding_value = result["encoding"]
        if encoding_value == "UTF-16":
            white_space = True
        else:
            white_space = False
        df = pd.read_csv(uploaded_file, encoding=encoding_value, delim_whitespace=white_space, on_bad_lines='skip')

        number_of_rows = len(df)
        if number_of_rows == 0:
            st.caption("Your sheet seems empty!")
        with st.expander("↕ View raw data", expanded=False):
            st.write(df)
    except UnicodeDecodeError:
        st.warning("""🚨 The file doesn't seem to load. Check the filetype, file format and Schema""")

else:
    st.info("👆 Upload a .csv or .txt file first.")
    st.stop()

with st.form(key='columns_in_form_2'):
    st.subheader("Please Select the Keyword Column")
    kw_col = st.selectbox('Select the column containing your keywords (Usually the H1 Column):', df.columns)
    submitted = st.form_submit_button('Submit')

if submitted:
    df = df[df[kw_col].notna()]  # drop missing values
    df.drop_duplicates(subset=kw_col, inplace=True)
    # make the graphs
    workbook = xlsxwriter.Workbook('chart_scatter.xlsx')
    bold = workbook.add_format({'bold': 1})

    cell_format1 = workbook.add_format()
    cell_format1.set_num_format('d-m-yyyy')

    ALL_KWS = df[kw_col].to_list()


    with stqdm(total=len(ALL_KWS)) as pbar:
        while counter != len(ALL_KWS) + 1:

            KW = ALL_KWS[start:counter]
            worksheet_name = str(ALL_KWS[start])
            worksheet_name = worksheet_name.replace(" ", "_")

            # strip out special characters
            spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                          "*", "+", ",", "-", ".", "/", ":", ";", "<",
                          "=", ">", "?", "@", "[", "\\", "]", "^",
                          "`", "{", "|", "}", "~", "–"]

            for char in spec_chars:
                worksheet_name = worksheet_name.replace(char, '')
            worksheet_name = worksheet_name[0:31]  # ensures the sheet name doesn't exceed 31 chars (excel limit)
            if worksheet_name == "nan":
                worksheet_name = "nan" + str(counter)
            worksheet = workbook.add_worksheet(worksheet_name)

            headings = ['Date', 'Actual', 'Predicted']  # Add the worksheet data that the charts will refer to.
            worksheet.write_row('A1', headings, bold)
            pt.build_payload(KW)
            df = pt.interest_over_time()
            pbar.set_description("Searching & Predicting: %s" % KW[0])
            pbar.update(1)
            try:
                df = df[df['isPartial'] == False].reset_index()
                data = df.rename(columns={'date': 'ds', KW[0]: 'y'})[['ds', 'y']]

                model = NeuralProphet(daily_seasonality=True)
                metrics = model.fit(data, freq="W")

                future = model.make_future_dataframe(data, periods=FORECAST_WEEKS, n_historic_predictions=HISTORIC)

                data = model.predict(future)
                data = data.rename(columns={'ds': 'date', 'y': 'actual', 'yhat1': 'predicted'})#[['date', 'actual', 'predicted']]
                worksheet.set_column('A:A', 16, cell_format1)
                try:
                    worksheet.write_column('A2', data['date'])
                except:
                    pass
                try:
                    worksheet.write_column('B2', data['actual'])
                except:
                    pass
                try:
                    worksheet.write_column('C2', data['predicted'])
                except:
                    pass

                # --------------------------------- create a new scatter chart ------------------------------
                max_rows = len(data)
                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})

                # Configure the first series.
                chart.add_series({
                    'name': '=' + worksheet_name + '!$B$1',
                    'categories': '=' + worksheet_name + '!$A$2:$A$' + str(max_rows),
                    'values': '=' + worksheet_name + '!$B$2:$B$' + str(max_rows),
                    'line': {'color': 'gray'},
                })

                # Configure second series.
                chart.add_series({
                    'name': '=' + worksheet_name + '!$C$1',
                    'categories': '=' + worksheet_name + '!$A$2:$A$' + str(max_rows),
                    'values': '=' + worksheet_name + '!$C$2:$C$' + str(max_rows),
                    'line': {'dash_type': 'round_dot', 'color': 'black'},
                })

                # Add a chart title and some axis labels.
                chart.set_title({'name': worksheet_name})
                chart.set_x_axis({'name': 'Months', 'date_axis': True})
                chart.set_y_axis({'name': 'Search Demand'})

                # Set an Excel chart style.
                chart.set_style(7)

                # Insert the chart into the worksheet (with an offset).
                worksheet.insert_chart('D2', chart, {'x_offset': 25, 'y_offset': 10, 'x_scale': 3, 'y_scale': 2})
                start += 1
                counter += 1
                time.sleep(SLEEP_TIMER)

            except KeyError:
                start += 1
                counter += 1

        # show completed successfully message
        st.success('Finished!')
        st.markdown("### **🎈 Download your results!**")
        st.write("")

        workbook.close()

        def get_binary_file_downloader_html(bin_file, file_label='File'):
            with open(bin_file, 'rb') as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
            return href

        st.markdown(get_binary_file_downloader_html('chart_scatter.xlsx', 'Your File!'), unsafe_allow_html=True)
