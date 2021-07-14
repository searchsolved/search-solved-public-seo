# conda env list
# from glob import glob  # Used to parse wildcard for csv import
# time the code execution
startTime = time.time()
# Your code here !
print ('The script took {0} seconds!'.format(time.time() - startTime))

# Shift left on empty cells
v = df.values
a = [[n] * v.shape[1] for n in range(v.shape[0])]
b = pd.isnull(v).argsort(axis=1, kind="mergesort")
new_array = v[a, b]

# count folder depth
df_gsc_data["Depth"] = (df_gsc_data["URL"].str.count("\/") - 3) 

# sorting data by a multiple columns
df_gsc_data.sort_values(["Depth", "URL"], ascending=[True, True], inplace=True)  
df_gsc_data.drop_duplicates(subset=['Keyword'], keep="first", inplace=True)

# this groups and keeps only the top 5 unique values
df_gsc_data = df_gsc_data.groupby(['URL']).head(5)

# import using GA data using a wildcard match
for f in glob("/python_scripts/Analytics*.xlsx"):
    df_ga = pd.read_excel((f), sheet_name="Dataset1")

#                           ==Retrieving Series/DataFrame Information: Basic Information==
count_row = df.shape[0]  # gives number of rows count
count_col = df.shape[1]  # gives number of col count
df["col_name"].nunique()  # Number of distinct values in a column
df.describe  # Basic descriptive statistics for each column
df.columns  # Describe DataFrame columns
df.info()  # Info on DataFrame
df.count()  # Number of non-NA values
df.count()  # Number of non-NA values
len(df)  # count the number of rows in dataframe
df["col_name"].value_counts()  # count number of rows with each unique value of variable
df["Folder Depth"] = (df["URL"].str.count("\/") - 3)  # # count folder depth

#                           ==Summarising Data==
df.sum()  # Sum of values
df.cumsum()  # Cumulative sum of values
df['col_name'].max()  # get the max value of a column
df['col_name'].min()  # get the min value of a column
df.idxmin()  # Minimum index value
df.idxmax()  # Maximum index value
df.mean()  # Mean of values
df.median()  # df.median()
df = df.groupby(['URL']).head(5)  # this groups and keeps only the top 5 unique values (so volume is calculated by the five opportunities shown) (Doesn't actually group, just keeps top x unique values)

#                           ==Applying Functions==
f = lambda x: x*2
df.apply(f)  # Apply function
df.applymap(f)  # Apply function element-wise

#                           ==Handling Missing Data==
df_copy["KW1"] = df_copy["KW1"].replace(r'^\s*$', np.nan, regex=True)  # replace empty strings with NaN values
df.dropna()  # Drop rows (all columns) that have NA/null data
df = df[df["Col"].notna()]  # Only Keep Rows which are not NaN
df.fillna(value)  # Replace NA/null data with specific value.
df.fillna({"Meta Description": 0, "No. of Sub Categories": 0}, inplace=True)  # Fill NA multiple columns

#                           ==Cleaning Data==
df["new_col_name"] = (df["col_name"].str.replace(r"[^a-zA-Z ]+", " ").str.strip())  # strip out special characters
df["col_name"].str.replace({"": "0"}, inplace=True)  # replace empty string with another string
df['employee_id'] = df['employee_id'].str.strip()  # remove whitespace
df["Col_Name"] = df["Col_Name"].str.replace(r"\r\n.*", "")  # remove whitespace etc

df["col"] = df["col"].apply(lambda x: x.replace("chars_to_replace", "new_chars"))  # this works if all other replace methods do not

# start strip out all special characters from a column
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
for char in spec_chars:
    df_sf['Title 1'] = df_sf['Title 1'].str.replace(char, ' ')

#                           ==Merging Data==
df = pd.merge(df,df2[['Key_Col','Target_Col']],on='Key_Column', how='left')  # merge only certain columns vlookup style
df_new = pd.merge(df_1, df_2, on="col_name", how="left")  # merging dataframes when column names are the same

#                           ==Converting Data==
df = pd.Dataframe(mylist)  # Make dataframe from a list
df["col_name"].tolist()  # Convert dataframe to list object
df["col_name"] = df["col_name"].round(decimals=2)  # Round Float to two decimal places
df = df.round({'a': 2, 'c': 2})

#                           ==String Manipulation==
df['col'].str.lower()  # Converts all characters to lowercase.
df['col'].str.upper()  # Converts all characters to uppercase.
df['col'].str.title()  # Converts first character of each word to uppercase and remaining to lowercase.
df['col'].str.capitalize()  # Converts first character to uppercase and remaining to lowercase.
df['col'].str.swapcase()  # Converts uppercase to lowercase and lowercase to uppercase.
df['col'].str.casefold()  # Removes all case distinctions in the string.
df.replace({"mystring": ""}, inplace=True)  # Replaces all string values across df. (Useful for 'nan' as a string)
df["Col"] = np.where(df["Col"].str[-1] == "s", df["Col"].str[:-1], df["Col"])  # test if the last character is an 's'

#                           ==Changing Data Types==
df = df.astype({"column_1": int, "column_2": str})  # change multiple data types at once.
df = df.astype(str)  # cast entire dataframe to a different datatype

#                           ==Column Manipulation==
del df["col_name"]  # delete a column
df["new_col"] = None  # add a new column with no data
df.rename(columns={"oldName1": "newName1", "oldName2": "newName2"}, inplace=True)  # renaming multiple columns
df["New_Col"] = df["Col_to_Lookup"].str.contains('|'.join(df_2['Col_to_Check']),case=False)  # check whether a column value is found in another column in a different dataframe
df["New_Col"] = df.apply(lambda row: row["Col_to_Lookup"] in row["Col_to_Check"], axis=1)  # check whether a column value is found in another column in the same dataframe
df = df.reindex(columns=cols)  # re-indexes columns in a new order. new columns can also be inserted

#                           ==Row Manipulation==
df = df[~df["col"].isin(["string"])]  # drop rows on exact match [used to drop non-indexable from a csv crawl file]
df = df[~df["col"].str.contains("string", na=False)]  # drop rows on partial string match [used to drop pagination etc]
df.drop_duplicates(subset="col_name", inplace=True)  # Drop Duplicate Rows
df = df[df["col"] != 0]  # drop rows not equal to int [used to drop rows with 0 transactions]
df.drop_duplicates(subset=["URL", "Keyword"], keep="first", inplace=True) # drop dupes if both cols are duped
df = df[df["Source Page"] != df["Keyword"]] # drop dupes if both cols are duped (alternative way)
df_s_dist['Combined KWs Deduped'] = (df_s_dist['Combined KWs'].str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))  # removes non-consecutive duplicates from a single cell

#                           ==Sorting Data==
df = df.sort_values(by="col_name", ascending=True)  # sorting data by a single column
df.sort_values(["Page", "H1"], ascending=[True, False], inplace=True,)  # sorting data by a multiple columns

#                           ==Grouping Data==
df = (df.groupby("query").agg({"clicks": "sum", "impressions": "mean", "ctr": "mean",}).reset_index())
df = (df.groupby("query").agg({"clicks": "sum", "impressions": "mean", "ctr": "mean",}).reset_index())
df['new_col'] = df['col_to_count_ifs_on'].map(df.groupby('col_to_count_ifs_on')['col_to_count_ifs_on'].count())  # countifs

# make new dtaframes based on str.contains
df_new = df_sf[df_sf['Address'].str.contains("/product/")].copy()

#                           ==Splitting / Joining Data==
df["New_Col"] = (df["URL_Col_to_Split"].str.split("/").str[-1])  # extract the last folder from a URL
df["Clean Categories"] = df["Category"].str.split(".").str[0]  # split on a specific character
my_string = my_string.split()  # default split on space
my_string = my_string.split(',')  # split on any character e.g. ,
df['SOURCE_NAME'] = df['SOURCE_NAME'].str.rsplit('_', n=1).str.get(0)  # remove everything after special character

# note the difference between str.split() and split() is that str.split() will *treat each cell as a series*
#                           ==Selection: Getting==
df['col_name']  # get one element
df[:3]  # get a subset of a dataframe
df.iloc([0],[0])  # select a single value by row and column position
df.loc([0],  ['col-name'])  # select a single value by row and column label

#                           ==Selection: Boolean Indexing==
s[~(s > 1)]  # Series s where value is not >1
s[(s < -1) | (s > 2)]  # s where value is <-1 or >2
df[df['Population']>1200000000]  # Use filter to adjust DataFrame
s['a'] = 6  # Set index a of Series s to 6

#                           ==Dropping==
s.drop(['a',  'c'])  # Drop values from rows (axis=0)
df.drop('Country', axis=1)  # Drop values from columns(axis=1)

#                           ==Sort and Rank==
df.sort_index()  # Sort by labels along an axis
df.sort_values(by='Country')  # Sort by the values along an axis
df.rank()  # Assign ranks to entries










