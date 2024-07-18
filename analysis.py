import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns # type: ignore

df = pd.read_csv("./Cleaned_dataset.csv")


print(df.info())
print(df.head())
print(df.duplicated().sum())
df = df.dropna()
df.drop_duplicates(keep='last',ignore_index=True ,inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.shape)

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

models_dict = {}

df.sort_values(by='Date_of_journey')

X=df.drop(['Fare'],axis=1)
y=df['Fare']

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X['Duration_in_hours'] = scaler.fit_transform(X[['Duration_in_hours']])
X['Days_left'] = scaler.fit_transform(X[['Days_left']])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

X_train

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm # type: ignore

dict = {}

destinations = ['Destination_Ahmedabad','Destination_Bangalore','Destination_Chennai','Destination_Delhi','Destination_Hyderabad','Destination_Kolkata','Destination_Mumbai']

        
subset = df[df["Airline_Indigo"] == 1]
# subset = subset[subset["Class_Economy"] == 1]
# subset = subset[subset["Source_Delhi"] == 1]
# subset = subset[subset["Destination_Mumbai"] == 1]

result = adfuller(subset["Fare"])
dict["Destination_Mumbai"] = (result[0], result[1])

'''except ValueError as e:
        print(f"ValueError for {destination}: {e}")
        dict[destination] = (None, None)'''

results_df = pd.DataFrame(dict).T
results_df.columns = ["ADF Statistic", "p-value"]

print('\n',results_df)

model = pm.auto_arima(subset["Fare"], seasonal=True, trace=True)
print(model.summary())