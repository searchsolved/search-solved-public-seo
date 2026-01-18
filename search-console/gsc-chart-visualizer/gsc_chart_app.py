# Author   : Lee Foot
# Website  : https://leefoot.com
# Simple GSC Connector for Streamlit
# More scripts and apps like this at https://www.leefoot.com

# Standard library imports
import datetime
import base64

# Related third-party imports
import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import pandas as pd
import searchconsole
import altair as alt

# Configuration: Set to True if running locally, False if running on Streamlit Cloud
IS_LOCAL = False

# Constants
SEARCH_TYPES = ["web", "image", "video", "news", "discover", "googleNews"]
DATE_RANGE_OPTIONS = [
    "Last 7 Days",
    "Last 30 Days",
    "Last 3 Months",
    "Last 6 Months",
    "Last 12 Months",
    "Last 16 Months",
]
DEVICE_OPTIONS = ["All Devices", "desktop", "mobile", "tablet"]
BASE_DIMENSIONS = ["page", "query", "country"]
MAX_ROWS = 250_000
DF_PREVIEW_ROWS = 100


# -------------
# Streamlit App Configuration
# -------------

def setup_streamlit():
    st.set_page_config(page_title="GSC Chart Visualizer | Lee Foot", layout="wide")
    st.title("GSC Chart Visualizer")
    st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Visualizes GSC data with interactive charts
    - Creates custom performance dashboards
    - Exports charts for reporting

    **How to use:**
    1. Upload GSC export data
    2. Select metrics to visualize
    3. Configure chart options
    4. Export visualizations

    **Best for:**
    - GSC data visualization
    - Client reporting
    - Performance trend analysis
    """)
    st.markdown(f"### Lightweight GSC Data Extractor with Charts (Max {MAX_ROWS:,} Rows)")

    st.markdown(
        """
        <p>
            Created by <a href="https://www.leefoot.com" target="_blank">Lee Foot</a> |
            <a href="https://github.com/searchsolved/search-solved-public-seo" target="_blank">More Tools on GitHub</a>
        """,
        unsafe_allow_html=True
    )
    st.divider()


def init_session_state():
    defaults = {'selected_property': None, 'selected_search_type': 'web',
                'selected_date_range': 'Last 7 Days', 'selected_dimensions': ['page'],
                'selected_device': 'All Devices'}
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# -------------
# Google Authentication Functions
# -------------

def load_config():
    # Check if secrets are configured
    try:
        client_id = st.secrets["installed"]["client_id"]
        client_secret = st.secrets["installed"]["client_secret"]
        redirect_uris = st.secrets["installed"]["redirect_uris"] if not IS_LOCAL else ["http://localhost:8501"]
    except (KeyError, FileNotFoundError):
        st.error("""
        ⚠️ **OAuth credentials not configured**

        This app requires Google OAuth credentials to connect to Search Console.

        To set up:
        1. Create OAuth credentials in [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
        2. Add secrets in Streamlit Cloud dashboard under Settings → Secrets:

        ```toml
        [installed]
        client_id = "your-client-id.apps.googleusercontent.com"
        client_secret = "your-client-secret"
        redirect_uris = ["https://your-app.streamlit.app"]
        ```

        For local development, set `IS_LOCAL = True` at the top of this file.
        """)
        st.stop()

    client_config = {
        "installed": {
            "client_id": str(client_id),
            "client_secret": str(client_secret),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
            "redirect_uris": (
                ["http://localhost:8501"]
                if IS_LOCAL
                else [str(redirect_uris[0])]
            ),
        }
    }
    return client_config


def init_oauth_flow(client_config):
    scopes = ["https://www.googleapis.com/auth/webmasters"]
    return Flow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=client_config["installed"]["redirect_uris"][0],
    )


def google_auth(client_config):
    flow = init_oauth_flow(client_config)
    auth_url, _ = flow.authorization_url(prompt="consent")
    return flow, auth_url


def auth_search_console(client_config, credentials):
    token = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "id_token": getattr(credentials, "id_token", None),
    }
    return searchconsole.authenticate(client_config=client_config, credentials=token)


# -------------
# Data Fetching Functions
# -------------

def list_gsc_properties(credentials):
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()
    return [site['siteUrl'] for site in site_list.get('siteEntry', [])] or ["No properties found"]


def fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    dimensions_with_date = dimensions + ['date']
    query = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(*dimensions_with_date)

    if 'device' in dimensions and device_type and device_type != 'All Devices':
        query = query.filter('device', 'equals', device_type.lower())

    try:
        df = query.limit(MAX_ROWS).get().to_dataframe()
        if 'date' in dimensions_with_date:
            df['date'] = pd.to_datetime(df['date'])

        # Calculate query count if 'query' is in dimensions
        if 'query' in dimensions:
            df['query_count'] = df.groupby('date')['query'].transform('nunique')

        return df
    except Exception as e:
        show_error(e)
        return pd.DataFrame()


def show_query_count_option():
    return st.sidebar.checkbox("Show Query Count", value=True)


@st.cache_data()
def fetch_and_load_data(_webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    with st.spinner('Fetching data...'):
        return fetch_gsc_data(_webproperty, search_type, start_date, end_date, dimensions, device_type)


# -------------
# Utility Functions
# -------------

def update_dimensions(selected_search_type):
    return BASE_DIMENSIONS + ['device'] if selected_search_type in SEARCH_TYPES else BASE_DIMENSIONS


def calc_date_range(selection):
    range_map = {
        'Last 7 Days': 7, 'Last 30 Days': 30, 'Last 3 Months': 90,
        'Last 6 Months': 180, 'Last 12 Months': 365, 'Last 16 Months': 480
    }
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=range_map.get(selection, 0))
    return start_date, end_date


def show_error(e):
    st.error(f"An error occurred: {e}")


def handle_property_change():
    st.session_state.selected_property = st.session_state['selected_property_selector']


# -------------
# File & Download Operations
# -------------

def show_dataframe(report):
    with st.expander("Preview the First 100 Rows"):
        st.dataframe(report.head(DF_PREVIEW_ROWS))


def download_csv_link(report):
    def to_csv(df):
        return df.to_csv(index=False, encoding='utf-8-sig')

    csv = to_csv(report)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="search_console_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)


# -------------
# Streamlit UI Components
# -------------

def show_google_sign_in(auth_url):
    with st.sidebar:
        if st.button("Sign in with Google"):
            # Open the authentication URL
            st.write('Please click the link below to sign in:')
            st.markdown(f'[Google Sign-In]({auth_url})', unsafe_allow_html=True)


def update_selected_property():
    # Update the session_state with the selected property
    st.session_state.selected_property = st.session_state.selected_property_selector


def show_property_selector(properties, account):
    # If the selected property is in the list of properties, use its index; otherwise, default to 0
    default_index = properties.index(
        st.session_state.selected_property) if st.session_state.selected_property in properties else 0

    selected_property = st.selectbox(
        "Select a Search Console Property:",
        properties,
        index=default_index,
        key='selected_property_selector',
        on_change=update_selected_property
    )

    return account[selected_property]


def show_search_type_selector():
    return st.selectbox(
        "Select Search Type:",
        SEARCH_TYPES,
        index=SEARCH_TYPES.index(st.session_state.selected_search_type),
        key='search_type_selector'
    )


def show_date_range_selector():
    return st.selectbox(
        "Select Date Range:",
        DATE_RANGE_OPTIONS,
        index=DATE_RANGE_OPTIONS.index(st.session_state.selected_date_range),
        key='date_range_selector'
    )


# Use this function to display device options
def show_device_selector():
    return st.selectbox(
        "Select Device Type:",
        DEVICE_OPTIONS,
        index=DEVICE_OPTIONS.index(st.session_state.selected_device),
        key='device_selector'
    )


def show_dimensions_selector(search_type):
    available_dimensions = update_dimensions(search_type)
    return st.multiselect(
        "Select Dimensions:",
        available_dimensions,
        default=st.session_state.selected_dimensions,
        key='dimensions_selector'
    )


def show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions):
    if st.button("Fetch Data"):
        report = fetch_and_load_data(webproperty, search_type, start_date, end_date, selected_dimensions)

        if report is not None:
            show_dataframe(report)
            download_csv_link(report)
            show_metrics_chart(report)


# -------------
# Visualisation
# -------------

def prepare_data_for_chart(report, metric):
    report['date'] = pd.to_datetime(report['date']).dt.date
    grouped_report = report.groupby('date')[metric].sum().reset_index()
    grouped_report.sort_values('date', inplace=True)
    return grouped_report


# Function to format numbers in K or M format
def format_number(num):
    if num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    else:
        return str(num)


def calculate_metrics(report):
    total_clicks = report['clicks'].sum()
    total_impressions = report['impressions'].sum()
    average_ctr = round(report['ctr'].mean() * 100, 2)
    average_position = report['position'].mean()
    return total_clicks, total_impressions, average_ctr, average_position


def apply_position_filter(report, position_filter):
    if position_filter == "1 - 3":
        return report[(report['position'] >= 1) & (report['position'] <= 3)]
    elif position_filter == "4 - 10":
        return report[(report['position'] >= 4) & (report['position'] <= 10)]
    elif position_filter == "11 - 20":
        return report[(report['position'] >= 11) & (report['position'] <= 20)]
    elif position_filter == "21 - 100":
        return report[(report['position'] >= 21) & (report['position'] <= 100)]
    elif position_filter == "First Page Keywords":
        return report[(report['position'] >= 1) & (report['position'] <= 10)]
    else:  # "All Results" or any other case
        return report


def create_chart(data, metric, color, title, metrics_selected):
    chart_title = title if metrics_selected <= 2 else None
    chart = alt.Chart(data).mark_line().encode(
        x='date:T',
        y=alt.Y(f'{metric}:Q', axis=alt.Axis(title=chart_title)),
        color=alt.Color('legend:N', legend=alt.Legend(title="Metrics")),
        tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip(f'{metric}:Q', title=metric.capitalize())]
    )
    return chart


def create_clicks_chart(report, metrics_selected):
    clicks_data = report.groupby('date')['clicks'].sum().reset_index()
    clicks_data['legend'] = 'Clicks'  # Add a legend field
    return create_chart(clicks_data, 'clicks', 'blue', "Clicks", metrics_selected)


def create_impressions_chart(report, metrics_selected):
    impressions_data = report.groupby('date')['impressions'].sum().reset_index()
    impressions_data['legend'] = 'Impressions'  # Add a legend field
    return create_chart(impressions_data, 'impressions', 'green', "Impressions", metrics_selected)


def create_ctr_chart(report, metrics_selected):
    ctr_data = report.groupby('date')['ctr'].mean().reset_index()
    ctr_data['legend'] = 'CTR'  # Add a legend field
    return create_chart(ctr_data, 'ctr', 'orange', "CTR", metrics_selected)


def create_position_chart(report, metrics_selected):
    position_data = report.groupby('date')['position'].mean().reset_index()
    position_data['legend'] = 'Position'  # Add a legend field
    return create_chart(position_data, 'position', 'red', "Position", metrics_selected)


def create_query_count_chart(report, metrics_selected):
    query_count_data = report.groupby('date')['query_count'].first().reset_index()
    query_count_data['legend'] = 'Query Count'  # Add a legend field
    return create_chart(query_count_data, 'query_count', 'purple', "Query Count", metrics_selected)


def calculate_metric_totals(report):
    total_clicks = int(report['clicks'].sum())
    total_impressions = int(report['impressions'].sum())
    average_ctr = round(report['ctr'].mean() * 100, 2)
    average_position = round(report['position'].mean(), 2)

    # Check if 'query_count' is in the DataFrame
    total_query_count = int(report['query_count'].sum()) if 'query_count' in report.columns else None

    return total_clicks, total_impressions, average_ctr, average_position, total_query_count


def show_position_filter():
    position_filter = st.selectbox(
        "Filter by Position:",
        ["All Results", "First Page Keywords", "1 - 3", "4 - 10", "11 - 20", "21 - 100"],
        index=0  # Default to 'All Results'
    )
    return position_filter


def show_metrics_chart(report):
    position_filter = show_position_filter()  # Get the position filter selection
    filtered_report = apply_position_filter(report, position_filter)  # Apply the filter
    charts = []

    METRIC_OPTIONS = ["Clicks", "Impressions", "CTR", "Position", "Query Count"]
    selected_metrics = st.multiselect("Select Metrics:", METRIC_OPTIONS, default=METRIC_OPTIONS)

    if "Clicks" in selected_metrics:
        clicks_chart = create_clicks_chart(filtered_report, len(charts) + 1)
        charts.append(clicks_chart)

    if "Impressions" in selected_metrics:
        impressions_chart = create_impressions_chart(filtered_report, len(charts) + 1)
        charts.append(impressions_chart)

    if "CTR" in selected_metrics:
        ctr_chart = create_ctr_chart(filtered_report, len(charts) + 1)
        charts.append(ctr_chart)

    if "Position" in selected_metrics:
        position_chart = create_position_chart(filtered_report, len(charts) + 1)
        charts.append(position_chart)

    if "Query Count" in selected_metrics and 'query_count' in report.columns:
        query_count_chart = create_query_count_chart(filtered_report, len(charts) + 1)
        charts.append(query_count_chart)

    if charts:
        combined_chart = alt.layer(*charts).resolve_scale(
            y='independent' if len(charts) <= 2 else 'shared'
        ).properties(width=700, height=400)
        st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.write("Please select at least one metric to display the chart.")


# -------------
# Main Streamlit App Function
# -------------

def handle_authentication(client_config):
    # Use st.query_params instead of deprecated st.experimental_get_query_params
    query_params = st.query_params
    auth_code = query_params.get("code", None)

    if auth_code and not st.session_state.get('credentials'):
        st.session_state.auth_flow.fetch_token(code=auth_code)
        st.session_state.credentials = st.session_state.auth_flow.credentials

    if not st.session_state.get('credentials'):
        show_google_sign_in(st.session_state.auth_url)
    else:
        return auth_search_console(client_config, st.session_state.credentials)
    return None


def handle_data_fetching(account):
    properties = list_gsc_properties(st.session_state.credentials)
    if properties:
        # Creating a layout with columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            webproperty = show_property_selector(properties, account)

        with col2:
            search_type = show_search_type_selector()

        with col3:
            date_range_selection = show_date_range_selector()

        with col4:
            device_type = show_device_selector()

        selected_dimensions = show_dimensions_selector(search_type)

        if st.button("Fetch Data"):
            st.cache_data.clear()
            start_date, end_date = calc_date_range(date_range_selection)
            st.session_state.report = fetch_and_load_data(webproperty, search_type, start_date, end_date,
                                                          selected_dimensions, device_type)
        return st.session_state.get('report')
    return None


def main():
    setup_streamlit()
    client_config = load_config()
    st.session_state.auth_flow, st.session_state.auth_url = google_auth(client_config)

    account = handle_authentication(client_config)
    if account:
        init_session_state()
        report = handle_data_fetching(account)

        if report is not None and not report.empty:
            show_dataframe(report)
            download_csv_link(report)
            show_metrics_chart(report)


if __name__ == "__main__":
    main()
