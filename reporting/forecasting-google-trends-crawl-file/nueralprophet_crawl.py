# Author: Lee Foot
# Website: https://leefoot.com

# Author   : Lee Foot
# Website  : https://leefoot.com
####################################################################################
#                                                                                  #
#  Google Trends Forecasting (DataForSEO)                                          #
#                                                                                  #
#  Fetches Google Trends interest-over-time via the DataForSEO Trends API          #
#  and forecasts future search interest with NeuralProphet.                        #
#                                                                                  #
####################################################################################
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

import streamlit as st

st.set_page_config(page_title="Google Trends Forecasting", page_icon="📈",
                   layout="wide")

import chardet
from stqdm import stqdm
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import os
import time
import requests
from requests.auth import HTTPBasicAuth
import math

set_random_seed(0)

st.title("Google Trends & NeuralProphet Forecasting")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Forecasts Google Trends data using NeuralProphet ML
    - Analyses seasonal patterns and projects future performance
    - Works with single keywords or batch processing

    **Two modes available:**
    - **Single Keyword:** Enter one keyword, get an instant forecast with visualisation
    - **Batch Upload:** Upload a CSV of keywords, get an Excel file with forecasts for each

    **How to use:**
    1. Enter your DataForSEO API credentials in the sidebar
    2. Choose your mode (Single Keyword or Batch Upload)
    3. Configure forecast settings in the sidebar
    4. Enter your keyword or upload your file
    5. Click Submit and wait for results
    6. Download your predictions

    **Data source:**
    - Uses the [DataForSEO Trends API](https://dataforseo.com) (official, reliable)
    - Cost: approximately $0.0012 per keyword request
    - Get API credentials at [dataforseo.com](https://app.dataforseo.com/api-access)

    **Best for:**
    - Keyword trend forecasting
    - Content calendar planning
    - Seasonal keyword analysis
    - Identifying rising/declining topics
    """)

# ---------------------------------------------------------------------------
# Location mapping: display name -> (location_code, location_name)
# Uses standard DataForSEO / Google Ads geo target codes.
# ---------------------------------------------------------------------------
LOCATIONS = {
    "United Kingdom": (2826, "United Kingdom"),
    "United States": (2840, "United States"),
    "Spain": (2724, "Spain"),
    "Brazil": (2076, "Brazil"),
    "France": (2250, "France"),
    "Germany": (2276, "Germany"),
    "Italy": (2380, "Italy"),
    "India": (2356, "India"),
    "Poland": (2616, "Poland"),
    "Romania": (2642, "Romania"),
    "China": (2156, "China"),
    "Sweden": (2752, "Sweden"),
    "Turkey": (2792, "Turkey"),
    "Wales (United Kingdom)": (2826, "United Kingdom"),
    "Norway": (2578, "Norway"),
    "Japan": (2392, "Japan"),
    "Ukraine": (2804, "Ukraine"),
}

DATAFORSEO_TRENDS_URL = "https://api.dataforseo.com/v3/keywords_data/dataforseo_trends/explore/live"
COST_PER_REQUEST = 0.0012  # USD per request (up to 5 keywords)


def fetch_trends_data(keywords, api_login, api_password, location_code, time_range="past_5_years"):
    """Fetch Google Trends interest-over-time data from DataForSEO.

    Parameters
    ----------
    keywords : list[str]
        One to five keywords.
    api_login : str
        DataForSEO login (email).
    api_password : str
        DataForSEO API password.
    location_code : int
        DataForSEO location code.
    time_range : str
        One of the preset time ranges.

    Returns
    -------
    dict
        Mapping of keyword -> pandas DataFrame with columns ``ds`` (datetime)
        and ``y`` (interest 0-100). Empty dict on failure.
    """
    if not keywords:
        return {}

    payload = [{
        "keywords": keywords[:5],
        "location_code": location_code,
        "type": "web",
        "time_range": time_range,
    }]

    try:
        resp = requests.post(
            DATAFORSEO_TRENDS_URL,
            json=payload,
            auth=HTTPBasicAuth(api_login, api_password),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        st.error(f"DataForSEO API request failed: {exc}")
        return {}

    # Check top-level status
    if data.get("status_code") != 20000:
        st.error(f"DataForSEO error: {data.get('status_message', 'Unknown error')}")
        return {}

    tasks = data.get("tasks", [])
    if not tasks:
        return {}

    task = tasks[0]
    if task.get("status_code") != 20000:
        st.error(f"DataForSEO task error: {task.get('status_message', 'Unknown error')}")
        return {}

    results = task.get("result", [])
    if not results:
        return {}

    result = results[0]
    graph_items = result.get("items", [])

    # Find the dataforseo_trends_graph item
    graph_data = None
    for item in graph_items:
        if item.get("type") == "dataforseo_trends_graph":
            graph_data = item
            break

    if not graph_data or not graph_data.get("data"):
        return {}

    returned_keywords = graph_data.get("keywords", keywords[:5])
    data_points = graph_data["data"]

    # Build a DataFrame per keyword
    result_dfs = {}
    for idx, kw in enumerate(returned_keywords):
        rows = []
        for point in data_points:
            date_str = point.get("date_from")
            values = point.get("values", [])
            if idx < len(values) and date_str:
                rows.append({"ds": pd.to_datetime(date_str), "y": values[idx]})
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values("ds").reset_index(drop=True)
            result_dfs[kw] = df

    return result_dfs


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
st.sidebar.header("DataForSEO Credentials")

api_login = os.environ.get("DATAFORSEO_LOGIN", "")
api_password = os.environ.get("DATAFORSEO_PASSWORD", "")

api_login = st.sidebar.text_input(
    "DataForSEO Login (Email)",
    value=api_login,
    help="Your DataForSEO account email. Set DATAFORSEO_LOGIN env var for CLI use.",
)
api_password = st.sidebar.text_input(
    "DataForSEO Password",
    value=api_password,
    type="password",
    help="Your DataForSEO API password. Set DATAFORSEO_PASSWORD env var for CLI use.",
)

st.sidebar.markdown("---")
st.sidebar.header("Forecast Settings")

FORECAST_WEEKS = st.sidebar.number_input('Weeks to forecast', min_value=1, max_value=104, value=52)

selected_location = st.sidebar.selectbox(
    "Location",
    list(LOCATIONS.keys()),
    index=0,
)
location_code, _location_name = LOCATIONS[selected_location]

TIME_RANGE = st.sidebar.selectbox(
    "Historical time range",
    [
        "past_5_years",
        "past_12_months",
        "past_90_days",
        "past_30_days",
    ],
    index=0,
    help="Longer ranges give the model more data to learn seasonal patterns.",
)

HISTORIC = st.sidebar.checkbox('Include historic predictions?', value=True)

# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
mode = st.radio("Choose mode:", ["Single Keyword", "Batch Upload"], horizontal=True)

if mode == "Single Keyword":
    # ==================== SINGLE KEYWORD MODE ====================
    st.subheader("Single Keyword Forecast")

    keyword = st.text_input('Enter your search keyword')

    # Cost estimate
    if keyword:
        st.caption(f"Estimated cost: ${COST_PER_REQUEST:.4f} (1 API request)")

    with st.form(key='single_keyword_form'):
        submitted = st.form_submit_button('Generate Forecast', type='primary')

    if submitted and keyword:
        if not api_login or not api_password:
            st.warning("Please enter your DataForSEO credentials in the sidebar.")
            st.stop()

        with st.spinner(f"Fetching trends data for '{keyword}'..."):
            try:
                kw_data = fetch_trends_data(
                    [keyword], api_login, api_password, location_code, TIME_RANGE
                )

                if not kw_data:
                    st.warning(
                        "No data received from DataForSEO. The keyword may have "
                        "insufficient search volume or your credentials may be incorrect."
                    )
                    st.stop()

                # Use the first (and only) keyword's data
                data = list(kw_data.values())[0]

                if data.empty or len(data) < 4:
                    st.warning("Not enough data points to build a forecast. Try a longer time range.")
                    st.stop()

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
                st.error("No data received. Please try again or use a different keyword.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif submitted and not keyword:
        st.warning("Please enter a keyword to forecast.")

else:
    # ==================== BATCH UPLOAD MODE ====================
    st.subheader("Batch Keyword Forecast")

    SLEEP_TIMER = st.sidebar.number_input('Delay between requests (seconds)', min_value=1, max_value=30, value=2)

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
                if not api_login or not api_password:
                    st.warning("Please enter your DataForSEO credentials in the sidebar.")
                    st.stop()

                df = df[df[kw_col].notna()]
                df.drop_duplicates(subset=kw_col, inplace=True)
                ALL_KWS = df[kw_col].astype(str).to_list()

                # Cost estimate: each request handles up to 5 keywords
                num_requests = math.ceil(len(ALL_KWS) / 5)
                estimated_cost = num_requests * COST_PER_REQUEST
                st.info(
                    f"Processing {len(ALL_KWS)} keywords in {num_requests} API "
                    f"request(s). Estimated cost: **${estimated_cost:.4f}**"
                )

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
                        kw_str = KW[0]
                        worksheet_name = str(kw_str).replace(" ", "_")

                        # Strip special characters
                        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                                      "=", ">", "?", "@", "[", "\\", "]", "^",
                                      "`", "{", "|", "}", "~"]

                        for char in spec_chars:
                            worksheet_name = worksheet_name.replace(char, '')
                        worksheet_name = worksheet_name[0:31]

                        if worksheet_name == "nan":
                            worksheet_name = f"nan{counter}"

                        pbar.set_description(f"Processing: {kw_str[:30]}...")
                        pbar.update(1)

                        try:
                            worksheet = workbook.add_worksheet(worksheet_name)
                            headings = ['Date', 'Actual', 'Predicted']
                            worksheet.write_row('A1', headings, bold)

                            kw_data = fetch_trends_data(
                                [kw_str], api_login, api_password, location_code, TIME_RANGE
                            )

                            # Exact match first, then case-insensitive fallback
                            matched_kw = None
                            if kw_data:
                                if kw_str in kw_data:
                                    matched_kw = kw_str
                                else:
                                    kw_lower = kw_str.lower()
                                    for k in kw_data:
                                        if k.lower() == kw_lower:
                                            matched_kw = k
                                            break
                                    if matched_kw is None and len(kw_data) == 1:
                                        matched_kw = next(iter(kw_data))

                            if matched_kw is not None:
                                data = kw_data[matched_kw]

                                if not data.empty and len(data) >= 4:
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
                                    worksheet.write('A2', 'Insufficient data points')
                                    errors.append(kw_str)
                            else:
                                worksheet.write('A2', 'No data available')
                                errors.append(kw_str)

                        except Exception as e:
                            errors.append(f"{kw_str}: {str(e)[:50]}")

                        start += 1
                        counter += 1

                        if counter <= len(ALL_KWS):
                            time.sleep(SLEEP_TIMER)

                workbook.close()

                st.success(f"Finished processing {len(ALL_KWS)} keywords!")

                if errors:
                    with st.expander(f"{len(errors)} keywords had issues"):
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
