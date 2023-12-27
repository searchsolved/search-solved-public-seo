import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import searchconsole
import datetime
import pandas as pd

# Constants
SEARCH_TYPES = ['web', 'image', 'video', 'news', 'discover', 'googleNews']
DATE_RANGE_OPTIONS = ['Last 7 Days', 'Last 30 Days', 'Last 3 Months', 'Last 6 Months', 'Last 12 Months',
                      'Last 16 Months']
DEVICE_OPTIONS = ['All Devices', 'desktop', 'mobile', 'tablet']
BASE_DIMENSIONS = ['page', 'query', 'country', 'date']
MAX_ROWS = 10_000


# -------------
# Streamlit App Configuration
# -------------

def configure_streamlit():
    st.set_page_config(page_title="Google Search Console Data", layout="wide")
    st.title('Google Search Console Data App')


def initialize_session_state():
    # Initialize or set default values for necessary session state variables
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None

    if 'selected_search_type' not in st.session_state:
        st.session_state.selected_search_type = 'web'

    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = 'Last 7 Days'

    if 'selected_dimensions' not in st.session_state:
        st.session_state.selected_dimensions = ['page', 'country']

# -------------
# Google Authentication Functions
# -------------

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


def initialise_oauth_flow(client_config):
    scopes = ['https://www.googleapis.com/auth/webmasters']
    return Flow.from_client_config(client_config, scopes=scopes,
                                   redirect_uri=client_config["installed"]["redirect_uris"][0])


def authenticate_google(client_config):
    flow = initialise_oauth_flow(client_config)
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


# -------------
# Data Fetching Functions
# -------------

def list_search_console_properties(credentials):
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()
    return [site['siteUrl'] for site in site_list.get('siteEntry', [])] or ["No properties found"]


def fetch_search_console_data(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    query = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(*dimensions)

    # Correctly apply device filter
    if 'device' in dimensions and device_type and device_type != 'All Devices':
        query = query.filter('device', 'equals', device_type.lower())

    try:
        return query.limit(MAX_ROWS).get().to_dataframe()
    except Exception as e:
        handle_error(e)
        return pd.DataFrame()


def fetch_data_with_loading(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    with st.spinner('Fetching data...'):
        return fetch_search_console_data(webproperty, search_type, start_date, end_date, dimensions, device_type)


# -------------
# Utility Functions
# -------------

def update_dimensions_options(selected_search_type):
    return BASE_DIMENSIONS + ['device', 'searchAppearance'] if selected_search_type in SEARCH_TYPES else BASE_DIMENSIONS


def calculate_date_range(selection):
    range_map = {
        'Last 7 Days': 7,
        'Last 30 Days': 30,
        'Last 3 Months': 90,
        'Last 6 Months': 180,
        'Last 12 Months': 365,
        'Last 16 Months': 480
    }
    today = datetime.date.today()
    return today - datetime.timedelta(days=range_map.get(selection, 0)), today


def handle_error(e):
    st.error(f"An error occurred: {e}")


def initialise_session_state():
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'selected_search_type' not in st.session_state:
        st.session_state.selected_search_type = 'web'
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = 'Last 7 Days'
    if 'selected_dimensions' not in st.session_state:
        st.session_state.selected_dimensions = ['page', 'country']
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = 'All Devices'


def on_property_change():
    st.session_state.selected_property = st.session_state['selected_property_selector']


# -------------
# Streamlit UI Components
# -------------

def display_google_sign_in(auth_url):
    st.markdown(f'<a href="{auth_url}" target="_self" class="btn btn-primary">Sign in with Google</a>',
                unsafe_allow_html=True)


def display_property_selector(properties, account):
    selected_property = st.selectbox(
        "Select a Search Console Property:",
        properties,
        index=properties.index(
            st.session_state.selected_property) if st.session_state.selected_property in properties else 0,
        key='selected_property_selector',
        on_change=on_property_change
    )
    return account[selected_property]


def display_search_type_selector():
    return st.selectbox(
        "Select Search Type:",
        SEARCH_TYPES,
        index=SEARCH_TYPES.index(st.session_state.selected_search_type),
        key='search_type_selector'
    )


def display_date_range_selector():
    return st.selectbox(
        "Select Date Range:",
        DATE_RANGE_OPTIONS,
        index=DATE_RANGE_OPTIONS.index(st.session_state.selected_date_range),
        key='date_range_selector'
    )


def display_dimensions_selector(search_type):
    available_dimensions = update_dimensions_options(search_type)
    return st.multiselect(
        "Select Dimensions:",
        available_dimensions,
        default=st.session_state.selected_dimensions,
        key='dimensions_selector'
    )


def display_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions):
    if st.button("Fetch Data"):
        report = fetch_data_with_loading(webproperty, search_type, start_date, end_date, selected_dimensions)
        if report is not None:
            st.dataframe(report)


# -------------
# Main Streamlit App Function
# -------------

# Main Streamlit App Function
def main():
    configure_streamlit()
    client_config = load_client_config()
    st.session_state.auth_flow, st.session_state.auth_url = authenticate_google(client_config)

    query_params = st.experimental_get_query_params()
    auth_code = query_params.get("code", [None])[0]

    if auth_code and not st.session_state.get('credentials'):
        st.session_state.auth_flow.fetch_token(code=auth_code)
        st.session_state.credentials = st.session_state.auth_flow.credentials

    if not st.session_state.get('credentials'):
        display_google_sign_in(st.session_state.auth_url)
    else:
        initialize_session_state()
        account = authenticate_searchconsole(client_config, st.session_state.credentials)
        properties = list_search_console_properties(st.session_state.credentials)

        if properties:
            webproperty = display_property_selector(properties, account)
            search_type = display_search_type_selector()
            date_range_selection = display_date_range_selector()
            start_date, end_date = calculate_date_range(date_range_selection)
            selected_dimensions = display_dimensions_selector(search_type)
            display_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions)

if __name__ == "__main__":
    main()
