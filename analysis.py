import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns # type: ignore

df = pd.read_csv("./Cleaned_dataset.csv")


print(df.info())
print(df.head())
print(df.duplicated().sum())
df.drop_duplicates(keep='last',ignore_index=True ,inplace=True)
df.reset_index(drop=True, inplace=True)

from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore

ohencode = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
encode = LabelEncoder()

# One Hot encoding the rest of the categorical data
for col in ["Class", "Journey_day", "Airline", "Source", "Departure", "Total_stops", "Arrival", "Destination"]:
    ohetransform = ohencode.fit_transform(df[[col]])
    df = pd.concat([df, ohetransform], axis=1).drop(columns=[col])

# Label Encoding Flight_code
df.Flight_code = encode.fit_transform(df.Flight_code)

print(df.head(10))

print(df.shape, '\n')
print("Total seat occupancy in each class", '\n')
#print(df.groupby('Class').size().sort_values(by=''))

df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'])

print(df.head())
print(df.info())