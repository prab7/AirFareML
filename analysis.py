import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

df = pd.read_csv("./Cleaned_dataset.csv")


print(df.info())
print(df.head())
print(df.duplicated().sum())
df.drop_duplicates(keep='last',ignore_index=True ,inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape, '\n')
print("Total seat occupancy in each class", '\n')
#print(df.groupby('Class').size().sort_values(by=''))

df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'])

print(df.head())
print(df.info())