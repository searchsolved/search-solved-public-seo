import streamlit as st

st.set_page_config(page_title="Google Trends & NeuralProphet - Explainable Trends at Scale", page_icon="📈",
                   layout="wide")  # needs to be the first thing after the streamlit import
st.set_option('deprecation.showPyplotGlobalUse', False)
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from pytrends.request import TrendReq

set_random_seed(0)

st.write(
    "Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://twitter.com/LeeFootSEO) / [![this is an image link](https://i.imgur.com/bjNRJra.png)](https://www.buymeacoffee.com/leefootseo) [Support My Work! Buy me a coffee!](https://www.buymeacoffee.com/leefootseo)")
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
    pt = TrendReq(hl=LANGUAGE, timeout=(10, 25), retries=RETRIES, backoff_factor=0.5)

    pt.build_payload(KW)
    df = pt.interest_over_time()

    df = df[df['isPartial'] == False].reset_index()
    data = df.rename(columns={'date': 'ds', KW[0]: 'y'})[['ds', 'y']]
    model = NeuralProphet(daily_seasonality=True)
    metrics = model.fit(data, freq="W")

    future = model.make_future_dataframe(data, periods=FORECAST_WEEKS, n_historic_predictions=HISTORIC)

    data = model.predict(future)
    data = data.rename(columns={'ds': 'date', 'y': 'actual', 'yhat1': 'predicted'})[['date', 'actual', 'predicted']]

    forecast = model.predict(future)
    ax = model.plot(forecast, ylabel='Google Searches', xlabel='Year', figsize=(14, 9))
    st.subheader(KW[0])
    st.pyplot()