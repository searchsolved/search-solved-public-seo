import streamlit as st
import base64
st.set_page_config(page_title="Google Trends & NeuralProphet - Explainable Trends at Scale", page_icon="📈",
                   layout="wide")  # needs to be the first thing after the streamlit import

st.set_option('deprecation.showPyplotGlobalUse', False)

from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from pytrends.request import TrendReq
import requests
import matplotlib.pyplot as plt

session = requests.Session()
session.get('https://trends.google.com')
cookies_map = session.cookies.get_dict()
nid_cookie = cookies_map['NID']

set_random_seed(0)

st.write(
    "[![this is an image link](https://i.imgur.com/Ex8eeC2.png)](https://www.patreon.com/leefootseo) [Become a Patreon for Early Access, Support & More!](https://www.patreon.com/leefootseo)  |  Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://twitter.com/LeeFootSEO)")
st.title("Google Trends & NeuralProphet - Explainable Trends at Scale")

# streamlit variables
KW = st.text_input('Input your search keyword')
KW = [KW]
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
RETRIES = st.sidebar.text_input('Select the number of retries when scraping', value=3)
HISTORIC = st.sidebar.checkbox('Make historic predictions?', value=True)
RETRIES = int(RETRIES)
FORECAST_WEEKS = int(FORECAST_WEEKS)

with st.form(key='columns_in_form_2'):
    submitted = st.form_submit_button('Submit')

if submitted:
    st.write("Searching & Predicting: %s" % KW[0])
    pt = TrendReq(hl=LANGUAGE, timeout=(10, 25), retries=RETRIES, backoff_factor=0.5,
                  requests_args={'headers': {'Cookie': f'NID={nid_cookie}'}})

    pt.build_payload(KW)
    df = pt.interest_over_time()
    try:
        df = df[df['isPartial'] == False].reset_index()
    except KeyError:
        st.warning("No Data Received from Google Trends, Please Search Again!")
        st.stop()
    data = df.rename(columns={'date': 'ds', KW[0]: 'y'})[['ds', 'y']]
    model = NeuralProphet(daily_seasonality=True)
    metrics = model.fit(data, freq="W")

    future = model.make_future_dataframe(data, periods=FORECAST_WEEKS, n_historic_predictions=HISTORIC)

    forecast = model.predict(future)
    data = forecast[['ds', 'y', 'yhat1']].rename(columns={'ds': 'date', 'y': 'actual', 'yhat1': 'predicted'})
    ax = model.plot(forecast, ylabel='Google Searches', xlabel='Year', figsize=(14, 9))

    st.subheader(KW[0])
    
    @st.cache_data
    def get_csv_link(data):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="your_gtrends_predictions.csv">📥 Download your predictions!</a>'
        return href


    st.markdown(get_csv_link(data), unsafe_allow_html=True)

    # create the plot
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['actual'], label='Actual')
    ax.plot(data['date'], data['predicted'], label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Interest Over Time')
    ax.set_title('Google Trends Predictions')
    ax.legend()

    st.pyplot()
