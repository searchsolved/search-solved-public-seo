import pandas as pd
import searchconsole
from tqdm import tqdm

# Paths to your client_secrets.json and credentials (you will have to provide these)
client_secrets_path = '/python_scripts/client_secrets.json'
credentials_path = '/python_scripts/credentials.json'

# URL of your site (you will have to provide this)
site_url = 'https://www.example/'

print("Reading crawl data...")
# Read your crawl data
crawl_df = pd.read_csv('/python_scripts/internal_html.csv', dtype="str")

# Columns to check for the presence of keywords
columns_to_check = ['Address', 'Title 1', 'H1-1', 'product_desc 1']

sort_metric = 'clicks'  # This can be 'clicks' or 'impressions'
MAX_KEYWORDS_PER_PAGE = 6  # Set this to the number of keywords you want to include per page
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
        .apply(lambda x: x.nlargest(MAX_KEYWORDS_PER_PAGE, sort_metric)[['query', sort_metric]])
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    return top_keywords_by_page

# Get the top keywords per page
print("Sorting and filtering top keywords...")
top_keywords = get_top_keywords_by_page(search_console_data, sort_metric)

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

# Create the wide format DataFrame
def create_wide_format_data(df, max_keywords):
    # Find the maximum number of keywords per page in the DataFrame
    max_num_keywords = min(df.groupby('Page').size().max(), max_keywords)

    # Create a new DataFrame with the correct number of columns for keywords
    columns = ['Page', 'Total Clicks', 'Total Keywords'] + \
              [f'KW{i} Clicks' for i in range(1, max_num_keywords + 1)] + \
              [f'KW{i} in Title' for i in range(1, max_num_keywords + 1)] + \
              [f'KW{i} in H1' for i in range(1, max_num_keywords + 1)] + \
              [f'KW{i} in Description' for i in range(1, max_num_keywords + 1)]
    wide_df = pd.DataFrame(columns=columns)

    # Populate the DataFrame
    for page, group in df.groupby('Page'):
        page_data = {
            'Page': page,
            'Total Clicks': group['Total Clicks'].sum(),
            'Total Keywords': group.shape[0]
        }
        # For each keyword
        for i, (idx, row) in enumerate(group.iterrows(), start=1):
            if i > max_num_keywords:
                break
            page_data[f'KW{i} Clicks'] = row['Total Clicks'] if pd.notnull(row['Total Clicks']) else 0
            page_data[f'KW{i} in Title'] = row['Title 1'] if pd.notnull(row['Title 1']) else False
            page_data[f'KW{i} in H1'] = row['H1-1'] if pd.notnull(row['H1-1']) else False
            page_data[f'KW{i} in Description'] = row['product_desc 1'] if pd.notnull(row['product_desc 1']) else False

        wide_df = wide_df.append(page_data, ignore_index=True)

    # Fill in missing values for pages with less than the maximum number of keywords
    fill_values = {f'KW{i} Clicks': 0 for i in range(1, max_num_keywords + 1)}
    fill_values.update({f'KW{i} in Title': False for i in range(1, max_num_keywords + 1)})
    fill_values.update({f'KW{i} in H1': False for i in range(1, max_num_keywords + 1)})
    fill_values.update({f'KW{i} in Description': False for i in range(1, max_num_keywords + 1)})
    wide_df.fillna(fill_values, inplace=True)

    return wide_df


# Check if the keywords are present in the columns and prepare the final wide format data
keyword_presence = check_keywords_in_columns(crawl_df, search_console_data, top_keywords,
                                             ['Address', 'Title 1', 'H1-1', 'product_desc 1'])

print("Creating wide format data...")
wide_format_data = create_wide_format_data(keyword_presence, MAX_KEYWORDS_PER_PAGE)

# Dynamically reorder the columns based on MAX_KEYWORDS_PER_PAGE
def reorder_columns(wide_format_df, max_keywords):
    column_order = ['Page', 'Total Clicks', 'Total Keywords'] + \
                   [f'KW{i} {info}' for i in range(1, max_keywords + 1) for info in ['Clicks', 'in Title', 'in H1', 'in Description']]
    return wide_format_df[column_order]

# Save the wide format data to a CSV file
output_path_wide = '/python_scripts/wide_format_data.csv'
wide_format_data.to_csv(output_path_wide, index=False)

print('Wide format data saved.')
