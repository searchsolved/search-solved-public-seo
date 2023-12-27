import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import searchconsole
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="Google Search Console Data", layout="wide")

# Initialize OAuth flow
def initialize_oauth_flow():
    client_secret = str(st.secrets["installed"]["client_secret"])
    client_id = str(st.secrets["installed"]["client_id"])
    redirect_uri = "http://localhost:8501"  # Ensure this matches the one set in Google Cloud Console

    credentials = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": [redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
        }
    }

    scopes = ['https://www.googleapis.com/auth/webmasters']
    flow = Flow.from_client_config(credentials, scopes=scopes, redirect_uri=redirect_uri)
    return flow

# Authentication function
def authenticate_google():
    flow = initialize_oauth_flow()
    auth_url, _ = flow.authorization_url(prompt='consent')
    return flow, auth_url

# List Search Console properties
def list_search_console_properties(credentials):
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()

    if 'siteEntry' in site_list:
        properties = [site['siteUrl'] for site in site_list['siteEntry']]
        return properties
    else:
        return ["No properties found"]

def authenticate_searchconsole(client_config, credentials):
    # Extract necessary items from credentials object
    token = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes,
        'id_token': getattr(credentials, 'id_token', None)  # Set to None if id_token is not present
    }

    account = searchconsole.authenticate(client_config=client_config, credentials=token)
    return account

# Streamlit app layout
st.title('Google Search Console Data App')

# Initialize session state variables
if 'auth_flow' not in st.session_state:
    st.session_state.auth_flow, _ = authenticate_google()

# Check for authorization code in URL on each run
query_params = st.experimental_get_query_params()
auth_code = query_params.get("code", [None])[0]

if auth_code and not st.session_state.get('credentials'):
    st.session_state.auth_flow.fetch_token(code=auth_code)
    st.session_state.credentials = st.session_state.auth_flow.credentials

if st.button('Sign in with Google'):
    st.session_state.auth_flow, auth_url = authenticate_google()
    st.markdown(f"[Authenticate with Google]({auth_url})", unsafe_allow_html=True)

if st.session_state.get('credentials'):
    # Your client_config setup as before
    client_config = {
        "installed": {
            "client_id": str(st.secrets["installed"]["client_id"]),
            "client_secret": str(st.secrets["installed"]["client_secret"]),
            "redirect_uris": ["http://localhost:8501"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
        }
    }

    account = authenticate_searchconsole(client_config, st.session_state.credentials)
    properties = list_search_console_properties(st.session_state.credentials)

    if properties:
        selected_property = st.selectbox("Select a Search Console Property:", properties)
        webproperty = account[selected_property]

        # Select Search Type
        search_type_options = ['web', 'image', 'video', 'news', 'discover', 'googleNews']
        search_type = st.selectbox("Select Search Type:", search_type_options, index=0)

        # Select Date Range
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        # Fetch data and display in DataFrame
        if st.button("Fetch Data"):
            report = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(
                'page').get().to_dataframe()
            st.dataframe(report)
