import pandas as pd
from pandas_profiling import ProfileReport

from pathlib import Path
import sys
import os

data_path = Path(__file__).parent

df = pd.read_csv(os.path.join(data_path, 'raw/census.csv'))

profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

profile.to_file(os.path.join(data_path,'rawdata_profile.html'))

# drop duplicates
df.drop_duplicates(inplace=True)

# strip white space from column names
df.columns = [col.strip() for col in df.columns]

# strip white space in the column value
for col in df.columns:
    if df[col].dtype == object:
        df[col] = [x.strip() for x in df[col]]
        
# saved the cleaned data
cleaned_file = os.path.join(data_path, 'cleaned/census.csv')   
df.to_csv(cleaned_file, index=False)