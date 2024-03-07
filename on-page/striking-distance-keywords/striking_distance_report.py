import pandas as pd
import searchconsole
from tqdm import tqdm

# Paths to your client_secrets.json and credentials (you will have to provide these)
client_secrets_path = '/python_scripts/client_secrets.json'
credentials_path = '/python_scripts/credentials.json'

# URL of your site (you will have to provide this)
site_url = 'https://www.example.com/'

print("Reading crawl data...")
# Read your crawl data
crawl_df = pd.read_csv('/python_scripts/internal_html.csv', dtype="str")

# Columns to check for the presence of keywords
columns_to_check = ['Address', 'Title 1', 'H1-1', 'product_desc 1']

sort_metric = 'clicks'  # This can be 'clicks' or 'impressions'

COUNTRY_FILTER = 'gbr'
POSITION_MIN = 3
POSITION_MAX = 20
IMPRESSIONS_MIN = 0

# Function to authenticate and get data from Google Search Console
def get_search_console_data(client_secrets_path, credentials_path, site_url, days=-7, search_type='web'):
    print("Authenticating with Google Search Console...")
    # Authenticate and get the account
    account = searchconsole.authenticate(client_config=client_secrets_path, credentials=credentials_path)

    # Select the property
    webproperty = account[site_url]

    print(f"Preparing query for {site_url}...")
    # Create a query with 'query', 'page', and 'country' dimensions for the last X days
    query = webproperty.query.range('today', days=days).search_type(search_type).dimension('query', 'page', 'country')

    # Get the report
    print("Pulling data from Search Console, Hold Tight!")
    report = query.get().to_dataframe()

    return report

print("Getting data from Search Console...")
# Get data from Search Console
search_console_data = get_search_console_data(client_secrets_path, credentials_path, site_url)
search_console_data.to_csv("/python_scripts/test.csv")

# Apply the filters to the search_console_data DataFrame
print("Filtering the Search Console data...")
search_console_data = search_console_data[
    (search_console_data['country'].str.lower() == COUNTRY_FILTER) &
    (search_console_data['position'] >= POSITION_MIN) &
    (search_console_data['position'] <= POSITION_MAX) &
    (search_console_data['impressions'] >= IMPRESSIONS_MIN)
]

def get_top_keywords_by_page(df, sort_metric):
    print("Determining top keywords by page...")
    # Group by the 'page' column and get the top keywords based on sort_metric
    top_keywords_by_page = (
        df.groupby('page')
        .apply(lambda x: x.nlargest(5, sort_metric)[['query', sort_metric]])
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    return top_keywords_by_page

# Get the top keywords per page
print("Sorting and filtering top keywords...")
top_keywords = get_top_keywords_by_page(search_console_data, sort_metric)

# Function to check if the keywords are in the specified columns and to include total clicks or impressions
# Function to check if the keywords are in the specified columns and to include total clicks or impressions
def check_keywords_in_columns(crawl_df, search_console_data, top_keywords, columns_to_check):
    print("Checking for keyword presence and aggregating data...")
    results = pd.DataFrame()

    # Prepare a dictionary with pages as keys and their respective dataframes as values
    page_groups = dict(tuple(crawl_df.groupby('Address')))

    # Iterate over the top keywords DataFrame with tqdm for a progress bar
    for index, row in tqdm(top_keywords.iterrows(), total=top_keywords.shape[0], desc="Processing"):
        keyword = row['query']
        page = row['page']
        total_metric = row[sort_metric]
        num_keywords = search_console_data[search_console_data['page'] == page]['query'].nunique()

        # Check if the current page exists in the crawl data
        if page in page_groups:
            page_data = page_groups[page]
            # Create a dictionary to hold the results for this keyword
            keyword_result = {
                'Page': page,
                'Keyword': keyword,
                f'Total {sort_metric.capitalize()}': total_metric,
                'Total Keywords': num_keywords
            }

            # Check each column for the keyword presence
            for column in columns_to_check:
                keyword_result[column] = page_data[column].str.contains(keyword, case=False, na=False).any()

            # Append the results to the DataFrame
            results = pd.concat([results, pd.DataFrame([keyword_result])], ignore_index=True)
        else:
            print(f"Page not found in crawl data: {page}")

    return results

print("Analyzing keywords in content...")
# Check if the keywords are present in the columns
keyword_presence = check_keywords_in_columns(crawl_df, search_console_data, top_keywords, columns_to_check)

# new code

# Create a new DataFrame to hold wide format data
wide_format_data = pd.DataFrame()

# Extract unique pages
unique_pages = keyword_presence['Page'].unique()

# Create an empty list to store dictionaries
page_dicts = []

# Iterate through each unique page
for page in unique_pages:
    page_data = keyword_presence[keyword_presence['Page'] == page]
    page_dict = {
        'Page': page,
        'Total Clicks': page_data['Total Clicks'].sum(),  # Assuming you want the sum of clicks
        'Total Keywords': page_data['Total Keywords'].iloc[0]  # Assuming this is the same for all rows of the same page
    }

    # For each of the top 5 keywords
    for i in range(1, 6):
        if i <= len(page_data):
            keyword_data = page_data.iloc[i - 1]  # Get data for the ith keyword
            page_dict[f'KW{i} Clicks'] = keyword_data['Total Clicks']
            page_dict[f'KW{i} in Title'] = keyword_data['Title 1']
            page_dict[f'KW{i} in H1'] = keyword_data['H1-1']
            page_dict[f'KW{i} in Description'] = keyword_data['product_desc 1']
        else:
            # If there are less than 5 keywords, fill in with zeros or appropriate values
            page_dict[f'KW{i} Clicks'] = 0
            page_dict[f'KW{i} in Title'] = False
            page_dict[f'KW{i} in H1'] = False
            page_dict[f'KW{i} in Description'] = False

    # Add the page's data to the list
    page_dicts.append(page_dict)

# Convert the list of dictionaries to a DataFrame
wide_format_data = pd.DataFrame(page_dicts)

# Reordering the columns to match the desired output (you may need to adjust column names)
column_order = ['Page', 'Total Clicks', 'Total Keywords'] + \
               [f'KW{i} {info}' for i in range(1, 6) for info in ['Clicks', 'in Title', 'in H1', 'in Description']]
wide_format_data = wide_format_data[column_order]

# Save the wide format data to a CSV file
output_path_wide = '/python_scripts/striking_distance_report.csv'
wide_format_data.to_csv(output_path_wide, index=False)

print('Wide format data saved.')

