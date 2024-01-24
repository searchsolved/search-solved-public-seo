# Timing a script
import time
startTime = time.time()
print('The script took {0} seconds!'.format(time.time() - startTime))
print(f'Completed in {time.time() - startTime:.2f} Seconds')  # Rounded to 2 decimal places

# URL extractor, extract URLs in a pandas dataframe
from urlextract import URLExtract
extractor = URLExtract()
df['extracted_data'] = df['Breadcrumb 1'].apply(lambda x: extractor.find_urls(x))

# Fixes text
import ftfy
df['col1'] = df['col1'].apply(ftfy.fix_text)

# Drop column on condition (less than, greater than, equal to, etc.)
df = df[df.colname != 0]

# Get the length of all words in a column
df['Length'] = df['Keyword'].astype(str).map(len)

# Check if string values in one column are found in another in the same or different dataframe
df_final['KW Found in Ads?'] = df_final["Search term"].isin(training_df["Search term"])

# Convert pandas column names to snake case
df.columns = [x.lower() for x in df.columns]  # Make lowercase
df.columns = df.columns.str.replace(' ', '_')  # Replace space with underscore

# Count words in a column by counting the spaces
df['totalwords'] = df['col'].str.count(' ') + 1

# List comprehension to only keep URLs that end with ".html"
urls = [val for val in urls if val.endswith(".html")]

# String split with expand and naming prefix for columns
df = df.join(df['Title 1'].str.split('|', expand=True).add_prefix('title_'))

# Apply a function to a dataframe
df['question_sentence_count'] = df['questions'].apply(lambda x: textstat.sentence_count(x))

# Sort words in a column in alphabetical order
df['New'] = [' '.join(sorted(x)) for x in df['Keyword'].str.split()]

# Replace words in one column with words in another
df['New'] = [' '.join(c for c in a.split() if c != b) for a, b in zip(df['New'], df['Cluster Name'])]

# Check if value/string is found in a column and return a value in a new column if found
df.loc[df["col_to_check"] == "value_to_check", "col_to_return_result_in"] = "result to return"  # Example 1
df.loc[df["col_to_check"] == "value_to_check", "col_to_return_result_in"] = df['existing_column_as_value']  # Example 2: Return another column as the value

# Get three words before a string by checking one column with another column
s = df['keyword']
df = df.assign(WordsBefore=df['description'].str.extract("(^\D+(?=f'{s}'))"),
               WordsAfter=df['description'].str.extract("((?<=f'{s}')\D+)"))
print(df)

# Check all rows for a specific string and get the count (used to check how many products are out of stock in a category)
df['count'] = (df[:].values == 'Out of stock').sum(axis=1)

# Check if a string is found in a column and return a specific string in a different column
mask = df['col_to_check'].str.contains('partial_string_to_match_on', na=True)
df.loc[mask, 'Col_to_return_value_in'] = "String to Populate Column With"

# Shift left on empty cells
v = df.values
a = [[n] * v.shape[1] for n in range(v.shape[0])]
b = pd.isnull(v).argsort(axis=1, kind="mergesort")
new_array = v[a, b]

# Count folder depth
df_gsc_data["Depth"] = df_gsc_data["URL"].str.count("\/") - 3

# Drop rows if partial keywords found in list
to_drop = ['amazon', 'b & q', 'b and q', 'b&q']
df = df[~df['query'].str.contains('|'.join(to_drop))]

# Sorting data by multiple columns
df.sort_values(["Depth", "URL"], ascending=[True, True], inplace=True)
df.drop_duplicates(subset=['Keyword'], keep="first", inplace=True)

# Group and keep only the top 5 unique values
df = df.groupby(['URL']).head(5)

# Import using GA data using a wildcard match
for f in glob("/python_scripts/Analytics*.xlsx"):
    df = pd.read_excel(f, sheet_name="Dataset1")

# Limit max words in a column (e.g., max_kw_length = 3)
df["col_name"] = df["col_name"].apply(lambda x: ' '.join(x.split(maxsplit=max_kw_length)[:max_kw_length]))

# Split on the nth occurrence of a special character
str = ".".join(str.split("/")[:3])

# Retrieving Series/DataFrame Information: Basic Information
count_row = df.shape[0]  # Gives the number of rows count
count_col = df.shape[1]  # Gives the number of column count
df["col_name"].nunique()  # Number of distinct values in a column
df.describe()  # Basic descriptive statistics for each column
df.columns  # Describe DataFrame columns
df.info()  # Info on DataFrame
df.count()  # Number of non-NA values
len(df)  # Count the number of rows in the dataframe
df["col_name"].value_counts()  # Count the number of rows with each unique value of the variable
df["Folder Depth"] = df["URL"].str.count("\/") - 3  # Count folder depth

# Summarizing Data
df.sum()  # Sum of values
df.cumsum()  # Cumulative sum of values
df['col_name'].max()  # Get the maximum value of a column
df['col_name'].min()  # Get the minimum value of a column
df.idxmin()  # Minimum index value
df.idxmax()  # Maximum index value
df.mean()  # Mean of values
df.median()  # Median of values
df = df.groupby(['URL']).head(5)  # Group and keep only the top 5 unique values

# Applying Functions
f = lambda x: x * 2
df.apply(f)  # Apply function
df.applymap(f)  # Apply function element-wise

# Handling Missing Data
df_copy["KW1"] = df_copy["KW1"].replace(r'^\s*$', np.nan, regex=True)  # Replace empty strings with NaN values
df.dropna()  # Drop rows (all columns) that have NA/null data
df = df[df["Col"].notna()]  # Only keep rows that are not NaN
df.fillna(value)  # Replace NA/null data with a specific value
df.fillna({"Meta Description": 0, "No. of Sub Categories": 0
