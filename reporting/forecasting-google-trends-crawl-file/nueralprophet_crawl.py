####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

import streamlit as st

st.set_page_config(page_title="[LEGACY] Google Trends Forecasting", page_icon="⚠️",
                   layout="wide")

import chardet
from stqdm import stqdm
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from pytrends.request import TrendReq
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import base64
import os
import time
import requests

set_random_seed(0)

st.title("⚠️ [LEGACY] Google Trends & NeuralProphet Forecasting")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

st.warning("**Legacy Tool:** This tool uses the unofficial Google Trends API (pytrends) which is frequently rate-limited and unreliable. Results may be incomplete or the tool may fail entirely. Use with caution.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Forecasts Google Trends data using NeuralProphet ML
    - Analyzes seasonal patterns and projects future performance
    - Works with single keywords or batch processing

    **Two modes available:**
    - **Single Keyword:** Enter one keyword, get an instant forecast with visualization
    - **Batch Upload:** Upload a CSV of keywords, get an Excel file with forecasts for each

    **How to use:**
    1. Choose your mode (Single Keyword or Batch Upload)
    2. Configure forecast settings in the sidebar
    3. Enter your keyword or upload your file
    4. Click Submit and wait for results
    5. Download your predictions

    **Best for:**
    - Keyword trend forecasting
    - Content calendar planning
    - Seasonal keyword analysis
    - Identifying rising/declining topics
    """)

# Sidebar settings (shared between both modes)
st.sidebar.header("Settings")
FORECAST_WEEKS = st.sidebar.number_input('Weeks to forecast', min_value=1, max_value=104, value=52)
LANGUAGE = st.sidebar.selectbox(
    "Google Trends language/region",
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
HISTORIC = st.sidebar.checkbox('Include historic predictions?', value=True)
RETRIES = st.sidebar.number_input('API retries', min_value=1, max_value=10, value=3)

# Mode selection
mode = st.radio("Choose mode:", ["Single Keyword", "Batch Upload"], horizontal=True)

if mode == "Single Keyword":
    # ==================== SINGLE KEYWORD MODE ====================
    st.subheader("Single Keyword Forecast")

    keyword = st.text_input('Enter your search keyword')

    with st.form(key='single_keyword_form'):
        submitted = st.form_submit_button('Generate Forecast', type='primary')

    if submitted and keyword:
        with st.spinner(f"Fetching Google Trends data for '{keyword}'..."):
            try:
                # Get cookie for pytrends
                session = requests.Session()
                session.get('https://trends.google.com')
                cookies_map = session.cookies.get_dict()
                nid_cookie = cookies_map.get('NID', '')

                pt = TrendReq(hl=LANGUAGE, timeout=(10, 25), retries=RETRIES, backoff_factor=0.5,
                              requests_args={'headers': {'Cookie': f'NID={nid_cookie}'}})

                pt.build_payload([keyword])
                df = pt.interest_over_time()

                if df.empty:
                    st.warning("No data received from Google Trends. The keyword may have insufficient search volume or the API may be rate-limited.")
                    st.stop()

                df = df[df['isPartial'] == False].reset_index()
                data = df.rename(columns={'date': 'ds', keyword: 'y'})[['ds', 'y']]

                st.info("Training NeuralProphet model...")
                model = NeuralProphet(daily_seasonality=True)
                metrics = model.fit(data, freq="W")

                future = model.make_future_dataframe(data, periods=FORECAST_WEEKS, n_historic_predictions=HISTORIC)
                forecast = model.predict(future)

                result_data = forecast[['ds', 'y', 'yhat1']].rename(
                    columns={'ds': 'date', 'y': 'actual', 'yhat1': 'predicted'}
                )

                st.success(f"Forecast complete for '{keyword}'!")

                # Display chart
                st.subheader(f"Trend Forecast: {keyword}")

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(result_data['date'], result_data['actual'], label='Actual', color='#2A9D8F', linewidth=2)
                ax.plot(result_data['date'], result_data['predicted'], label='Predicted', color='#E76F51',
                        linewidth=2, linestyle='--')
                ax.set_xlabel('Date')
                ax.set_ylabel('Interest Over Time')
                ax.set_title(f'Google Trends Forecast: {keyword}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                # Show data table
                with st.expander("View forecast data"):
                    st.dataframe(result_data, use_container_width=True)

                # Download button
                csv = result_data.to_csv(index=False)
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name=f"gtrends_forecast_{keyword.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

            except KeyError as e:
                st.error("No data received from Google Trends. Please try again or use a different keyword.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif submitted and not keyword:
        st.warning("Please enter a keyword to forecast.")

else:
    # ==================== BATCH UPLOAD MODE ====================
    st.subheader("Batch Keyword Forecast")

    SLEEP_TIMER = st.sidebar.number_input('Delay between requests (seconds)', min_value=1, max_value=30, value=5)

    uploaded_file = st.file_uploader("Upload your CSV with keywords", type=['csv', 'txt'])

    if uploaded_file is not None:
        try:
            result = chardet.detect(uploaded_file.getvalue())
            encoding_value = result["encoding"]
            white_space = encoding_value == "UTF-16"
            df = pd.read_csv(uploaded_file, encoding=encoding_value, delim_whitespace=white_space, on_bad_lines='skip')

            if len(df) == 0:
                st.warning("Your file seems empty!")
                st.stop()

            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)

            with st.form(key='batch_form'):
                st.write("**Select the keyword column:**")
                kw_col = st.selectbox('Column containing keywords:', df.columns)
                submitted = st.form_submit_button('Start Batch Forecast', type='primary')

            if submitted:
                df = df[df[kw_col].notna()]
                df.drop_duplicates(subset=kw_col, inplace=True)
                ALL_KWS = df[kw_col].astype(str).to_list()

                st.info(f"Processing {len(ALL_KWS)} keywords...")

                pt = TrendReq(hl=LANGUAGE, timeout=(10, 25), retries=RETRIES, backoff_factor=0.5)

                # Create Excel workbook
                workbook = xlsxwriter.Workbook('gtrends_forecasts.xlsx')
                bold = workbook.add_format({'bold': 1})
                cell_format1 = workbook.add_format()
                cell_format1.set_num_format('d-m-yyyy')

                counter = 1
                start = 0
                errors = []

                with stqdm(total=len(ALL_KWS)) as pbar:
                    while counter <= len(ALL_KWS):
                        KW = ALL_KWS[start:counter]
                        worksheet_name = str(ALL_KWS[start]).replace(" ", "_")

                        # Strip special characters
                        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                                      "=", ">", "?", "@", "[", "\\", "]", "^",
                                      "`", "{", "|", "}", "~", "–"]

                        for char in spec_chars:
                            worksheet_name = worksheet_name.replace(char, '')
                        worksheet_name = worksheet_name[0:31]

                        if worksheet_name == "nan":
                            worksheet_name = f"nan{counter}"

                        pbar.set_description(f"Processing: {KW[0][:30]}...")
                        pbar.update(1)

                        try:
                            worksheet = workbook.add_worksheet(worksheet_name)
                            headings = ['Date', 'Actual', 'Predicted']
                            worksheet.write_row('A1', headings, bold)

                            pt.build_payload(KW)
                            trends_df = pt.interest_over_time()

                            if not trends_df.empty:
                                trends_df = trends_df[trends_df['isPartial'] == False].reset_index()
                                data = trends_df.rename(columns={'date': 'ds', KW[0]: 'y'})[['ds', 'y']]

                                model = NeuralProphet(daily_seasonality=True)
                                model.fit(data, freq="W")

                                future = model.make_future_dataframe(data, periods=FORECAST_WEEKS, n_historic_predictions=HISTORIC)
                                forecast = model.predict(future)
                                result_data = forecast.rename(columns={'ds': 'date', 'y': 'actual', 'yhat1': 'predicted'})

                                worksheet.set_column('A:A', 16, cell_format1)

                                for i, date in enumerate(result_data['date']):
                                    worksheet.write(i + 1, 0, str(date)[:10])

                                if 'actual' in result_data.columns:
                                    worksheet.write_column('B2', result_data['actual'].fillna(''))
                                if 'predicted' in result_data.columns:
                                    worksheet.write_column('C2', result_data['predicted'].fillna(''))

                                # Create chart
                                max_rows = len(result_data)
                                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})

                                chart.add_series({
                                    'name': f'={worksheet_name}!$B$1',
                                    'categories': f'={worksheet_name}!$A$2:$A${max_rows + 1}',
                                    'values': f'={worksheet_name}!$B$2:$B${max_rows + 1}',
                                    'line': {'color': 'gray'},
                                })

                                chart.add_series({
                                    'name': f'={worksheet_name}!$C$1',
                                    'categories': f'={worksheet_name}!$A$2:$A${max_rows + 1}',
                                    'values': f'={worksheet_name}!$C$2:$C${max_rows + 1}',
                                    'line': {'dash_type': 'round_dot', 'color': 'black'},
                                })

                                chart.set_title({'name': worksheet_name})
                                chart.set_x_axis({'name': 'Date', 'date_axis': True})
                                chart.set_y_axis({'name': 'Search Interest'})
                                chart.set_style(7)

                                worksheet.insert_chart('D2', chart, {'x_offset': 25, 'y_offset': 10, 'x_scale': 2.5, 'y_scale': 1.5})
                            else:
                                worksheet.write('A2', 'No data available')
                                errors.append(KW[0])

                        except Exception as e:
                            errors.append(f"{KW[0]}: {str(e)[:50]}")

                        start += 1
                        counter += 1

                        if counter <= len(ALL_KWS):
                            time.sleep(SLEEP_TIMER)

                workbook.close()

                st.success(f"Finished processing {len(ALL_KWS)} keywords!")

                if errors:
                    with st.expander(f"⚠️ {len(errors)} keywords had issues"):
                        for err in errors:
                            st.write(f"- {err}")

                # Download button
                with open('gtrends_forecasts.xlsx', 'rb') as f:
                    st.download_button(
                        label="Download Excel Report",
                        data=f,
                        file_name="gtrends_forecasts.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Cleanup
                if os.path.exists('gtrends_forecasts.xlsx'):
                    os.remove('gtrends_forecasts.xlsx')

        except UnicodeDecodeError:
            st.error("Could not read file. Please check the file format and encoding.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Upload a CSV file containing your keywords to get started.")
