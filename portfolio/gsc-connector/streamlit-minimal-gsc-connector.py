import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import searchconsole
import datetime

import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="Google Search Console Data", layout="wide")
max_rows = 10_000

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


def fetch_search_console_data(webproperty, search_type, start_date, end_date, dimensions, device_type=None, max_rows=1000):
    query = webproperty.query.range(start_date, end_date).search_type(search_type)

    # Add all dimensions in a single call
    if 'page' in dimensions and 'country' in dimensions:
        query = query.dimension(*dimensions)  # Unpack the dimensions list
    else:
        for dimension in dimensions:
            query = query.dimension(dimension)

    if 'device' in dimensions and device_type and device_type != 'All Devices':
        query = query.filter('device', 'equals', device_type)

    try:
        # Include a row limit as per your example, if needed
        return query.limit(max_rows).get().to_dataframe()
    except Exception as e:
        handle_error(e)
        return pd.DataFrame()


def fetch_data_with_loading(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    with st.spinner('Fetching data...'):
        try:
            return fetch_search_console_data(webproperty, search_type, start_date, end_date, dimensions, device_type)
        except Exception as e:
            handle_error(e)
            return None


def calculate_date_range(selection):
    """ Calculate start and end dates based on selection """
    today = datetime.date.today()
    if selection == 'Last 7 Days':
        return today - datetime.timedelta(days=7), today
    elif selection == 'Last 30 Days':
        return today - datetime.timedelta(days=30), today
    elif selection == 'Last 3 Months':
        return today - datetime.timedelta(days=90), today
    elif selection == 'Last 6 Months':
        return today - datetime.timedelta(days=180), today
    elif selection == 'Last 12 Months':
        return today - datetime.timedelta(days=365), today
    elif selection == 'Last 16 Months':
        return today - datetime.timedelta(days=480), today
    else:
        return None, None  # Default case, can be adjusted as needed


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


def on_property_change():
    """ Callback function to handle property change """
    st.session_state.selected_property = st.session_state['selected_property_selector']


if st.session_state.get('credentials'):
    account = authenticate_searchconsole(client_config, st.session_state.credentials)
    properties = list_search_console_properties(st.session_state.credentials)

    if properties:
        if 'selected_property' not in st.session_state:
            st.session_state.selected_property = properties[0]

        selected_property = st.selectbox(
            "Select a Search Console Property:",
            properties,
            index=properties.index(st.session_state.selected_property),
            key='selected_property_selector',
            on_change=on_property_change
        )

        webproperty = account[selected_property]

        # Manage search type state
        if 'selected_search_type' not in st.session_state:
            st.session_state.selected_search_type = 'web'

        search_type = st.selectbox("Select Search Type:",
                                   ['web', 'image', 'video', 'news', 'discover', 'googleNews'],
                                   index=['web', 'image', 'video', 'news', 'discover', 'googleNews'].index(
                                       st.session_state.selected_search_type),
                                   key='search_type_selector')

        if search_type != st.session_state.selected_search_type:
            st.session_state.selected_search_type = search_type

        # Date range selection with state management
        if 'selected_date_range' not in st.session_state:
            st.session_state.selected_date_range = 'Last 7 Days'

        date_range_selection = st.selectbox(
            "Select Date Range:",
            ['Last 7 Days', 'Last 30 Days', 'Last 3 Months', 'Last 6 Months', 'Last 12 Months', 'Last 16 Months'],
            index=['Last 7 Days', 'Last 30 Days', 'Last 3 Months', 'Last 6 Months', 'Last 12 Months',
                   'Last 16 Months'].index(st.session_state.selected_date_range),
            key='date_range_selector'
        )

        if date_range_selection != st.session_state.selected_date_range:
            st.session_state.selected_date_range = date_range_selection

        start_date, end_date = calculate_date_range(date_range_selection)

        # Dynamic dimension selection with state management
        available_dimensions = update_dimensions_options(search_type)
        if 'selected_dimensions' not in st.session_state:
            st.session_state.selected_dimensions = ['page', 'country']

        selected_dimensions = st.multiselect("Select Dimensions:", available_dimensions,
                                             default=st.session_state.selected_dimensions,
                                             key='dimensions_selector')

        st.session_state.selected_dimensions = selected_dimensions

        # Sub-options for device selection
        if 'device' in selected_dimensions:
            if 'selected_device' not in st.session_state:
                st.session_state.selected_device = 'All Devices'

            device_options = get_device_options()
            selected_device = st.selectbox("Select Device Type:", device_options,
                                           index=device_options.index(st.session_state.selected_device),
                                           key='device_selector')

            st.session_state.selected_device = selected_device
        else:
            selected_device = None

        if st.button("Fetch Data"):
            report = fetch_data_with_loading(webproperty, search_type, start_date, end_date, selected_dimensions,
                                             selected_device)
            if report is not None:
                st.dataframe(report)
