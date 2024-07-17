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
print(df.groupby('Class').size().sort_values(by=''))

fig,axs = plt.subplots(1,2, figsize=(20,10),sharey=True)
sns.countplot(data=df,y="Source",ax=axs[0])
sns.countplot(data=df,y="Destination",ax=axs[1])
for ax in axs:
    ax.set_xlabel('Number of Flight Tickets')
axs[0].set_title("Number of Flights flying from",size=15)
axs[1].set_title("Number of Flights flying to",size=15)
plt.show()


