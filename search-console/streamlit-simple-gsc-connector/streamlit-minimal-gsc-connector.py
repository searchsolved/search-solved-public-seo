####################################################################################
# Simple GSC Connector for Streamlit by Lee Foot 3rd Oct 2025                     #
# Website  : https://leefoot.com/                                                  #
# Contact  : https://leefoot.com/hire-me/                                          #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://twitter.com/LeeFootSEO                                        #
####################################################################################

# Standard library imports
import datetime

# Related third-party imports
import streamlit as st
from streamlit_elements import Elements
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import pandas as pd
import searchconsole

# Configuration: Set to True if running locally, False if running on Streamlit Cloud
# IS_LOCAL = True
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
    "Custom Range"
]
DEVICE_OPTIONS = ["All Devices", "desktop", "mobile", "tablet"]
BASE_DIMENSIONS = ["page", "query", "country", "date"]


# -------------
# Streamlit App Configuration
# -------------

def setup_streamlit():
    """
    Configures Streamlit's page settings and displays the app title and markdown information.
    Sets the page layout, title, and markdown content with links and app description.
    """
    st.set_page_config(page_title="✨ GSC Data Exporter | LeeFoot.com", layout="wide")
    
    # Main title
    st.title("✨ GSC Data Exporter")
    
    # Subtitle and social links in a cleaner format
    st.markdown(
        """
        <style>
        .social-links {
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
            color: #666;
        }
        .social-links a {
            color: #1f77b4;
            text-decoration: none;
            margin-right: 1.5rem;
            font-weight: 500;
        }
        .social-links a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="social-links">
            Created by <strong>Lee Foot</strong> • 
            <a href="https://twitter.com/LeeFootSEO" target="_blank">Follow me on 𝕏</a> • 
            <a href="https://www.linkedin.com/in/lee-foot/" target="_blank">Connect on LinkedIn</a> • 
            <a href="https://leefoot.com" target="_blank">🌐 More Tools on my Website</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add help/instructions expander
    with st.expander("ℹ️ About This Tool & How to Use It", expanded=False):
        st.markdown("""
        ### What Does This Tool Do?
        
        The **GSC Data Exporter** allows you to extract and analyse your Google Search Console data quickly and efficiently. 
        Unlike the standard GSC interface which limits exports to 1,000 rows, this tool fetches complete datasets with 
        automatic batch processing for larger date ranges.
        
        ### Key Features
        
        - **Unlimited data extraction** – No 1,000 row limit
        - **Flexible date ranges** – From 7 days to 16 months of historical data
        - **Multiple dimensions** – Analyse by page, query, country, date, and device
        - **Batch processing** – Automatically splits large requests to prevent timeouts
        - **Query position analysis** – See how many queries rank in different position ranges (1-3, 4-10, 11-20, 20+)
        - **Multiple search types** – Web, image, video, news, Discover, and Google News
        - **CSV export** – Download your data for further analysis in Excel or other tools
        
        ### How to Use
        
        1. **Sign in** – Click the "Sign in with Google" button in the sidebar to authenticate
        2. **Select property** – Choose which website you want to analyse
        3. **Configure settings** – Select your search type, date range, and dimensions
        4. **Fetch data** – Click "Fetch Data" to retrieve your GSC information
        5. **Download** – Export your results as CSV files
        
        ### Tips
        
        - For **query position analysis**, make sure to include both 'query' and 'date' dimensions
        - **Large date ranges** (>30 days) will automatically use batch processing – this is normal and ensures reliable data retrieval
        - The tool remembers your last selections, making repeat exports faster
        - You can select multiple dimensions to get more granular insights
        - **For very large sites** (expecting >500k rows): Consider splitting your export into smaller date ranges (e.g., quarterly or monthly) to avoid browser memory limits
        
        ### Privacy & Security
        
        This tool uses Google's official OAuth authentication and only requests read-only access to your Search Console data. 
        Your credentials are never stored, and all data processing happens in your browser session.
        """)
    
    st.divider()


def init_session_state():
    """
    Initialises or updates the Streamlit session state variables for property selection,
    search type, date range, dimensions, and device type.
    """
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'selected_search_type' not in st.session_state:
        st.session_state.selected_search_type = 'web'
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = 'Last 7 Days'
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=7)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.date.today()
    if 'selected_dimensions' not in st.session_state:
        st.session_state.selected_dimensions = ['page', 'query']
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = 'All Devices'
    if 'custom_start_date' not in st.session_state:
        st.session_state.custom_start_date = datetime.date.today() - datetime.timedelta(days=7)
    if 'custom_end_date' not in st.session_state:
        st.session_state.custom_end_date = datetime.date.today()


# -------------
# Google Authentication Functions
# -------------

def load_config():
    """
    Loads the Google API client configuration from Streamlit secrets.
    Returns a dictionary with the client configuration for OAuth.
    """
    client_config = {
        "installed": {
            "client_id": str(st.secrets["installed"]["client_id"]),
            "client_secret": str(st.secrets["installed"]["client_secret"]),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
            "redirect_uris": (
                ["http://localhost:8501"]
                if IS_LOCAL
                else [str(st.secrets["installed"]["redirect_uris"][0])]
            ),
        }
    }
    return client_config


def init_oauth_flow(client_config):
    """
    Initialises the OAuth flow for Google API authentication using the client configuration.
    Sets the necessary scopes and returns the configured Flow object.
    """
    scopes = ["https://www.googleapis.com/auth/webmasters"]
    return Flow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=client_config["installed"]["redirect_uris"][0],
    )


def google_auth(client_config):
    """
    Starts the Google authentication process using OAuth.
    Generates and returns the OAuth flow and the authentication URL.
    """
    flow = init_oauth_flow(client_config)
    auth_url, _ = flow.authorization_url(prompt="consent")
    return flow, auth_url


def auth_search_console(client_config, credentials):
    """
    Authenticates the user with the Google Search Console API using provided credentials.
    Returns an authenticated searchconsole client.
    """
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
    """
    Lists all Google Search Console properties accessible with the given credentials.
    Returns a list of property URLs or a message if no properties are found.
    """
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()
    return [site['siteUrl'] for site in site_list.get('siteEntry', [])] or ["No properties found"]


def fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    """
    Fetches Google Search Console data for a specified property, date range, dimensions, and device type.
    Handles errors and returns the data as a DataFrame.
    """
    query = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(*dimensions)

    if 'device' in dimensions and device_type and device_type != 'All Devices':
        query = query.filter('device', 'equals', device_type.lower())

    try:
        df = query.get().to_dataframe()
        
        # Ensure proper data types for all columns
        if 'ctr' in df.columns:
            df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').round(2).astype(float)
        
        if 'position' in df.columns:
            df['position'] = pd.to_numeric(df['position'], errors='coerce').round(2).astype(float)
        
        if 'clicks' in df.columns:
            df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
        
        if 'impressions' in df.columns:
            df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        show_error(e)
        return pd.DataFrame()


def fetch_data_in_batches(webproperty, search_type, start_date, end_date, dimensions, device_type=None, batch_days=7):
    """
    Fetches Google Search Console data in batches to prevent timeouts.
    Splits the date range into smaller chunks and combines the results.
    """
    all_data = []
    current_date = start_date
    total_days = (end_date - start_date).days
    estimated_batches = (total_days // batch_days) + 1
    
    # Create progress indicators
    batch_info = st.info(f"📊 Fetching {total_days + 1} days of data in {estimated_batches} batches...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    rows_text = st.empty()
    
    batch_count = 0
    total_rows = 0
    
    while current_date <= end_date:
        batch_end = min(current_date + datetime.timedelta(days=batch_days - 1), end_date)
        batch_count += 1
        
        # Update progress
        progress = min((current_date - start_date).days / total_days, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"🔄 Batch {batch_count}/{estimated_batches}: {current_date.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}")
        
        # Fetch data for this batch
        batch_data = fetch_gsc_data(webproperty, search_type, current_date, batch_end, dimensions, device_type)
        
        if not batch_data.empty:
            all_data.append(batch_data)
            total_rows += len(batch_data)
            rows_text.text(f"📈 Total rows collected: {total_rows:,}")
            
            # Warn if dataset is getting very large
            if total_rows > 1000000 and batch_count == int(estimated_batches * 0.5):
                st.warning(f"⚠️ Large dataset alert: {total_rows:,} rows collected at 50% completion. Final dataset may exceed 2M rows, which can cause browser memory issues.\n\n**Recommendation:** After this completes, consider splitting your next export into smaller date ranges (e.g., 3-month periods instead of 12 months) to ensure reliable downloads.")
        
        current_date = batch_end + datetime.timedelta(days=1)
    
    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed! Fetched {batch_count} batches successfully.")
    rows_text.text(f"📊 Total rows collected: {total_rows:,}")
    
    # Combine all batches
    if all_data:
        with st.spinner('Combining batches...'):
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Optimize memory usage for large datasets
            if len(combined_df) > 100000:
                # Convert object columns to category type where appropriate
                for col in combined_df.select_dtypes(include=['object']).columns:
                    if combined_df[col].nunique() / len(combined_df) < 0.5:  # If less than 50% unique values
                        combined_df[col] = combined_df[col].astype('category')
        
        # Clear all the progress messages now that we're done
        batch_info.empty()
        progress_bar.empty()
        status_text.empty()
        rows_text.empty()
        
        return combined_df
    else:
        st.warning("No data found for the selected date range.")
        return pd.DataFrame()


def fetch_data_loading(webproperty, search_type, start_date, end_date, dimensions, device_type=None):
    """
    Fetches Google Search Console data with a loading indicator. Utilises 'fetch_gsc_data' for data retrieval.
    Returns the fetched data as a DataFrame.
    """
    # Calculate date range in days
    days_diff = (end_date - start_date).days
    
    # Use batch processing for date ranges longer than 30 days
    if days_diff > 30:
        info_msg = st.info(f"Large date range detected ({days_diff} days). Using batch processing to prevent timeouts...")
        result = fetch_data_in_batches(webproperty, search_type, start_date, end_date, dimensions, device_type, batch_days=7)
        info_msg.empty()  # Clear the info message after completion
        return result
    else:
        with st.spinner('Fetching data...'):
            return fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, device_type)


# -------------
# Utility Functions
# -------------

def update_dimensions(selected_search_type):
    """
    Updates and returns the list of dimensions based on the selected search type.
    Adds 'device' to dimensions if the search type requires it.
    """
    return BASE_DIMENSIONS + ['device'] if selected_search_type in SEARCH_TYPES else BASE_DIMENSIONS


def calc_date_range(selection, custom_start=None, custom_end=None):
    """
    Calculates the date range based on the selected range option.
    Returns the start and end dates for the specified range.
    """
    range_map = {
        'Last 7 Days': 7,
        'Last 30 Days': 30,
        'Last 3 Months': 90,
        'Last 6 Months': 180,
        'Last 12 Months': 365,
        'Last 16 Months': 480
    }
    today = datetime.date.today()
    if selection == 'Custom Range':
        if custom_start and custom_end:
            return custom_start, custom_end
        else:
            return today - datetime.timedelta(days=7), today
    return today - datetime.timedelta(days=range_map.get(selection, 0)), today


def show_error(e):
    """
    Displays an error message in the Streamlit app.
    Formats and shows the provided error 'e'.
    """
    st.error(f"An error occurred: {e}")


def property_change():
    """
    Updates the 'selected_property' in the Streamlit session state.
    Triggered on change of the property selection.
    """
    st.session_state.selected_property = st.session_state['selected_property_selector']


# -------------
# File & Download Operations
# -------------

def create_query_count_chart(query_analysis):
    """
    Creates a stacked bar chart showing query count distribution by position range over time.
    Only shows if query analysis data is available.
    """
    if query_analysis is None or query_analysis.empty:
        return None
    
    import plotly.graph_objects as go
    
    # Get position range columns (exclude 'Month' and 'Total Queries')
    position_columns = [col for col in query_analysis.columns if col not in ['Month', 'Total Queries']]
    
    fig = go.Figure()
    
    # Define colors for each position range
    colors = {
        'Positions 1-3': '#2ecc71',
        'Positions 4-10': '#3498db',
        'Positions 11-20': '#f39c12',
        'Positions 20+': '#e74c3c'
    }
    
    # Reverse the order so highest positions (1-3) appear on the left in legend
    position_columns_reversed = list(reversed(position_columns))
    
    # Add a bar for each position range
    for col in position_columns_reversed:
        fig.add_trace(go.Bar(
            name=col,
            x=query_analysis['Month'],
            y=query_analysis[col],
            marker_color=colors.get(col, '#95a5a6')
        ))
    
    fig.update_layout(
        title='Query Count Distribution by Position Range (Monthly)',
        xaxis_title='Month',
        yaxis_title='Number of Unique Queries',
        barmode='stack',
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            traceorder='normal'
        )
    )
    
    return fig


def analyze_query_counts(report):
    """
    Analyzes query counts by position ranges and month.
    Returns a DataFrame with monthly breakdown of queries in different position ranges.
    """
    if report.empty or 'query' not in report.columns or 'position' not in report.columns or 'date' not in report.columns:
        return pd.DataFrame()
    
    # Convert date to datetime if it's not already
    report['date'] = pd.to_datetime(report['date'])
    report['year_month'] = report['date'].dt.to_period('M').astype(str)
    
    # Define position ranges
    def categorize_position(pos):
        if pos <= 3:
            return 'Positions 1-3'
        elif pos <= 10:
            return 'Positions 4-10'
        elif pos <= 20:
            return 'Positions 11-20'
        else:
            return 'Positions 20+'
    
    # Apply position categorization
    report['position_range'] = report['position'].apply(categorize_position)
    
    # Count unique queries per month per position range
    query_counts = report.groupby(['year_month', 'position_range'])['query'].nunique().reset_index()
    query_counts.columns = ['Month', 'Position Range', 'Unique Queries']
    
    # Pivot to create a more readable format
    pivot_table = query_counts.pivot(index='Month', columns='Position Range', values='Unique Queries').fillna(0)
    
    # Ensure all position ranges exist in the correct order
    position_order = ['Positions 1-3', 'Positions 4-10', 'Positions 11-20', 'Positions 20+']
    for pos_range in position_order:
        if pos_range not in pivot_table.columns:
            pivot_table[pos_range] = 0
    
    # Reorder columns
    pivot_table = pivot_table[position_order]
    
    # Add total column
    pivot_table['Total Queries'] = pivot_table.sum(axis=1)
    
    # Reset index to make Month a column
    pivot_table = pivot_table.reset_index()
    
    # Convert counts to integers
    for col in pivot_table.columns:
        if col != 'Month':
            pivot_table[col] = pivot_table[col].astype(int)
    
    return pivot_table


def download_csv_link(report):
    """
    Generates and displays a download button for the report DataFrame in CSV format.
    Optimized for large datasets to prevent memory issues.
    """
    # Warn users about very large datasets
    row_count = len(report)
    if row_count > 500000:
        st.warning(f"⚠️ Large dataset detected ({row_count:,} rows). CSV generation may take 30-60 seconds and could cause browser slowdown. Consider filtering dimensions or splitting the date range for better performance.")
    
    # Convert to CSV in memory-efficient way with caching
    @st.cache_data(show_spinner=False)
    def convert_df(df):
        return df.to_csv(index=False, encoding='utf-8-sig')
    
    # Show spinner during CSV generation for large datasets
    if row_count > 100000:
        with st.spinner(f'Preparing CSV file ({row_count:,} rows)... This may take a moment.'):
            csv = convert_df(report)
    else:
        csv = convert_df(report)
    
    st.download_button(
        label=f"📥 Download CSV File ({row_count:,} rows)",
        data=csv,
        file_name=f"search_console_data_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )


def download_query_analysis_csv(query_analysis):
    """
    Generates and displays a download button for the query analysis DataFrame in CSV format.
    """
    csv = query_analysis.to_csv(index=False, encoding='utf-8-sig')
    
    st.download_button(
        label="📊 Download Query Analysis CSV",
        data=csv,
        file_name=f"query_analysis_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )


# -------------
# Streamlit UI Components
# -------------

def show_google_sign_in(auth_url):
    """
    Displays the Google sign-in button with an animated arrow and authentication URL in the Streamlit sidebar.
    """
    with st.sidebar:
        # Add animated arrow pointing to the button with ocean blue styling
        st.markdown(
            """
            <style>
            @keyframes bounce {
                0%, 20%, 50%, 80%, 100% {
                    transform: translateY(0);
                }
                40% {
                    transform: translateY(-8px);
                }
                60% {
                    transform: translateY(-4px);
                }
            }
            @keyframes pulse {
                0% {
                    transform: scale(1);
                }
                50% {
                    transform: scale(1.03);
                }
                100% {
                    transform: scale(1);
                }
            }
            .arrow-container {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 15px;
                padding: 20px;
                background: #1eb8d9;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(30, 184, 217, 0.4);
                animation: pulse 2s infinite;
            }
            .arrow-text {
                font-size: 22px;
                color: #ffffff;
                font-weight: bold;
                margin-right: 15px;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            }
            .arrow-icon-box {
                background: rgba(255, 255, 255, 0.25);
                padding: 8px 12px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .arrow-icon {
                font-size: 28px;
                animation: bounce 2s infinite;
                filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.2));
            }
            .instructions-text {
                text-align: center;
                color: #666;
                font-size: 14px;
                margin-top: 10px;
                font-style: italic;
            }
            /* Style the Streamlit link button to match ocean blue theme */
            .stButton > button {
                background: #1eb8d9 !important;
                color: white !important;
                border: none !important;
                padding: 12px 24px !important;
                font-size: 16px !important;
                font-weight: 600 !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 8px rgba(30, 184, 217, 0.3) !important;
            }
            .stButton > button:hover {
                background: #1aa5c4 !important;
                box-shadow: 0 4px 12px rgba(30, 184, 217, 0.5) !important;
            }
            </style>
            <div class="arrow-container">
                <span class="arrow-text">👇 Click here to get started!</span>
                <div class="arrow-icon-box">
                    <span class="arrow-icon">⬇</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Use link_button for direct single-click authentication
        st.link_button(
            "🔐 Sign in with Google",
            auth_url,
            type="primary",
            use_container_width=True
        )
        
        # Add helpful instruction text
        st.markdown(
            '<p class="instructions-text">Click the button above to authenticate with Google</p>',
            unsafe_allow_html=True
        )


def show_property_selector(properties, account):
    """
    Displays a dropdown selector for Google Search Console properties.
    Returns the selected property's webproperty object.
    """
    selected_property = st.selectbox(
        "Select a Search Console Property:",
        properties,
        index=properties.index(
            st.session_state.selected_property) if st.session_state.selected_property in properties else 0,
        key='selected_property_selector',
        on_change=property_change
    )
    return account[selected_property]


def show_search_type_selector():
    """
    Displays a dropdown selector for choosing the search type.
    Returns the selected search type.
    """
    return st.selectbox(
        "Select Search Type:",
        SEARCH_TYPES,
        index=SEARCH_TYPES.index(st.session_state.selected_search_type),
        key='search_type_selector'
    )


def show_date_range_selector():
    """
    Displays a dropdown selector for choosing the date range.
    Returns the selected date range option.
    """
    return st.selectbox(
        "Select Date Range:",
        DATE_RANGE_OPTIONS,
        index=DATE_RANGE_OPTIONS.index(st.session_state.selected_date_range),
        key='date_range_selector'
    )


def show_custom_date_inputs():
    """
    Displays date input fields for custom date range selection.
    Updates session state with the selected dates.
    """
    st.session_state.custom_start_date = st.date_input("Start Date", st.session_state.custom_start_date)
    st.session_state.custom_end_date = st.date_input("End Date", st.session_state.custom_end_date)


def show_dimensions_selector(search_type):
    """
    Displays a multi-select box for choosing dimensions based on the selected search type.
    Returns the selected dimensions.
    """
    available_dimensions = update_dimensions(search_type)
    selected = st.multiselect(
        "Select Dimensions:",
        available_dimensions,
        default=st.session_state.selected_dimensions,
        key='dimensions_selector'
    )
    
    # Show helpful tip about query analysis feature
    if selected and not (('query' in selected) and ('date' in selected)):
        st.info("💡 Tip: Include both 'query' and 'date' dimensions to see query count analysis by position ranges.")
    
    return selected


def show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions):
    """
    Displays a button to fetch data based on selected parameters.
    Shows the report DataFrame, query analysis chart, and download links upon successful data fetching.
    """
    if st.button("Fetch Data", key="fetch_button"):
        report = fetch_data_loading(webproperty, search_type, start_date, end_date, selected_dimensions)

        if report is not None and not report.empty:
            # Store report and metadata in session state
            st.session_state.report = report
            st.session_state.query_analysis = None
            st.session_state.show_data_preview = True
            st.session_state.show_query_analysis = False
            st.session_state.fetch_summary = f"✅ Successfully fetched {len(report):,} rows"
            
            # Check if we should show query analysis
            if all(dim in selected_dimensions for dim in ['query', 'date']):
                query_analysis = analyze_query_counts(report)
                
                if not query_analysis.empty:
                    st.session_state.query_analysis = query_analysis
                    st.session_state.show_query_analysis = True
    
    # Display fetch summary if available (persists across downloads)
    if st.session_state.get('fetch_summary'):
        st.success(st.session_state.fetch_summary)
    
    # Display info and download options if available (persists across downloads)
    if st.session_state.get('show_data_preview', False) and 'report' in st.session_state:
        # Download options
        st.subheader("📥 Download Options")
            
    # Show download buttons if data exists in session state
    if 'report' in st.session_state and st.session_state.report is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            download_csv_link(st.session_state.report)
        
        with col2:
            if 'query_analysis' in st.session_state and st.session_state.query_analysis is not None:
                download_query_analysis_csv(st.session_state.query_analysis)
            else:
                st.info("Query analysis not available - Re-run with 'query' and 'date' dimensions selected to see results")
        
        # Show stacked bar chart underneath download buttons if query analysis is available
        if 'query_analysis' in st.session_state and st.session_state.query_analysis is not None:
            chart = create_query_count_chart(st.session_state.query_analysis)
            if chart:
                st.plotly_chart(chart, use_container_width=True)


# -------------
# Main Streamlit App Function
# -------------

# Main Streamlit App Function
def main():
    """
    The main function for the Streamlit application.
    Handles the app setup, authentication, UI components, and data fetching logic.
    """
    setup_streamlit()
    client_config = load_config()
    st.session_state.auth_flow, st.session_state.auth_url = google_auth(client_config)

    query_params = st.query_params
    auth_code = query_params.get("code", None)

    if auth_code and not st.session_state.get('credentials'):
        st.session_state.auth_flow.fetch_token(code=auth_code)
        st.session_state.credentials = st.session_state.auth_flow.credentials

    if not st.session_state.get('credentials'):
        show_google_sign_in(st.session_state.auth_url)
    else:
        init_session_state()
        account = auth_search_console(client_config, st.session_state.credentials)
        properties = list_gsc_properties(st.session_state.credentials)

        if properties:
            webproperty = show_property_selector(properties, account)
            search_type = show_search_type_selector()
            date_range_selection = show_date_range_selector()

            if date_range_selection == 'Custom Range':
                show_custom_date_inputs()
                start_date, end_date = st.session_state.custom_start_date, st.session_state.custom_end_date
            else:
                start_date, end_date = calc_date_range(date_range_selection)

            selected_dimensions = show_dimensions_selector(search_type)
            show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions)


if __name__ == "__main__":
    main()
