import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import searchconsole

import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="Google Search Console Data", layout="wide")


# ---------------------
# Function Definitions
# ---------------------

# Configuration and Authentication Functions
def load_client_config():
    return {
        "installed": {
            "client_id": str(st.secrets["installed"]["client_id"]),
            "client_secret": str(st.secrets["installed"]["client_secret"]),
            "redirect_uris": ["http://localhost:8501"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
        }
    }


def initialize_oauth_flow(client_config):
    scopes = ['https://www.googleapis.com/auth/webmasters']
    flow = Flow.from_client_config(client_config, scopes=scopes,
                                   redirect_uri=client_config["installed"]["redirect_uris"][0])
    return flow


def authenticate_google(client_config):
    flow = initialize_oauth_flow(client_config)
    auth_url, _ = flow.authorization_url(prompt='consent')
    return flow, auth_url


def authenticate_searchconsole(client_config, credentials):
    token = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes,
        'id_token': getattr(credentials, 'id_token', None)
    }
    return searchconsole.authenticate(client_config=client_config, credentials=token)


# Data Fetching and Processing Functions
def list_search_console_properties(credentials):
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()
    return [site['siteUrl'] for site in site_list.get('siteEntry', [])] or ["No properties found"]


def get_device_options():
    """ Return device options for selection including 'All Devices' """
    return ['All Devices', 'desktop', 'mobile', 'tablet']


def update_dimensions_options(selected_search_type):
    """ Update available dimensions based on selected search type """
    base_dimensions = ['page', 'query', 'country', 'date']
    if selected_search_type in ['web', 'image', 'video', 'news', 'discover', 'googleNews']:
        return base_dimensions + ['device', 'searchAppearance']
    else:
        return base_dimensions


def fetch_search_console_data(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    query = webproperty.query.range(start_date, end_date).search_type(search_type)
    for dimension in dimensions:
        query = query.dimension(dimension)

    # Apply device type filter if 'device' is in dimensions and device_type is specified and not 'All Devices'
    if 'device' in dimensions and device_type and device_type != 'All Devices':
        device_filter = {
            'dimension': 'device',
            'operator': 'equals',
            'expression': device_type
        }
        query = query.filter(**device_filter)

    return query.get().to_dataframe()


def fetch_data_with_loading(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    with st.spinner('Fetching data...'):
        try:
            return fetch_search_console_data(webproperty, search_type, start_date, end_date, dimensions, device_type)
        except Exception as e:
            handle_error(e)
            return None


# Error Handling and Loading Functions
def handle_error(e):
    """ Log and display errors """
    st.error(f"An error occurred: {e}")


# ---------------------
# Streamlit App Layout
# ---------------------

st.title('Google Search Console Data App')

client_config = load_client_config()

if 'auth_flow' not in st.session_state or 'auth_url' not in st.session_state:
    st.session_state.auth_flow, st.session_state.auth_url = authenticate_google(client_config)

query_params = st.experimental_get_query_params()
auth_code = query_params.get("code", [None])[0]

if auth_code and not st.session_state.get('credentials'):
    st.session_state.auth_flow.fetch_token(code=auth_code)
    st.session_state.credentials = st.session_state.auth_flow.credentials

if not st.session_state.get('credentials'):
    st.markdown(f'<a href="{st.session_state.auth_url}" target="_self" class="btn btn-primary">Sign in with Google</a>',
                unsafe_allow_html=True)

if st.session_state.get('credentials'):
    account = authenticate_searchconsole(client_config, st.session_state.credentials)
    properties = list_search_console_properties(st.session_state.credentials)

    if properties:
        selected_property = st.selectbox("Select a Search Console Property:", properties)
        webproperty = account[selected_property]

        search_type = st.selectbox("Select Search Type:", ['web', 'image', 'video', 'news', 'discover', 'googleNews'],
                                   index=0)
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        # Dynamic dimension selection
        available_dimensions = update_dimensions_options(search_type)
        selected_dimensions = st.multiselect("Select Dimensions:", available_dimensions, default=['page', 'country'])

        # Sub-options for device selection
        if 'device' in selected_dimensions:
            device_options = get_device_options()
            selected_device = st.selectbox("Select Device Type:", device_options, index=0)  # 'All Devices' as default
        else:
            selected_device = None

        if st.button("Fetch Data"):
            report = fetch_data_with_loading(webproperty, search_type, start_date, end_date, selected_dimensions,
                                             selected_device)
            if report is not None:
                st.dataframe(report)
