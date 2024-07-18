
# This file is just for me to try out stuff before putting in the notebook.

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

aggregated_df = df.groupby(['Class_Business', 'Class_Economy', 'Class_First',
       'Class_Premium Economy','Date_of_journey', 'Flight_code', 'Airline_Air India',
       'Airline_AirAsia', 'Airline_AkasaAir', 'Airline_AllianceAir',
       'Airline_GO FIRST', 'Airline_Indigo', 'Airline_SpiceJet',
       'Airline_StarAir', 'Airline_Vistara',
       ]).agg(
    total_passengers=('Fare', 'size'),
    average_fare=('Fare', 'mean'),

    Source_Ahmedabad=('Source_Ahmedabad', 'first'),
    Source_Bangalore=('Source_Bangalore', 'first'),
    Source_Chennai=('Source_Chennai', 'first'),
    Source_Delhi=('Source_Delhi', 'first'),
    Source_Hyderabad=('Source_Hyderabad', 'first'),
    Source_Kolkata=('Source_Kolkata', 'first'),
    Source_Mumbai=('Source_Mumbai', 'first'),
    Destination_Ahmedabad=('Destination_Ahmedabad', 'first'),
    Destination_Bangalore=('Destination_Bangalore', 'first'),
    Destination_Chennai=('Destination_Chennai', 'first'),
    Destination_Delhi=('Destination_Delhi', 'first'),
    Destination_Hyderabad=('Destination_Hyderabad', 'first'),
    Destination_Kolkata=('Destination_Kolkata', 'first'),
    Destination_Mumbai=('Destination_Mumbai', 'first'),
    Journey_day_Friday=('Journey_day_Friday', 'first'),
    Journey_day_Monday=('Journey_day_Monday', 'first'),
    Journey_day_Saturday=('Journey_day_Saturday', 'first'),
    Journey_day_Sunday=('Journey_day_Sunday', 'first'),
    Journey_day_Thursday=('Journey_day_Thursday', 'first'),
    Journey_day_Tuesday=('Journey_day_Tuesday', 'first'),
    Journey_day_Wednesday=('Journey_day_Wednesday', 'first'),
    Days_left=('Days_left', 'mean'),
    Arrival_12PM_6PM=('Arrival_12 PM - 6 PM', 'first'),
    Arrival_6AM_12PM=('Arrival_6 AM - 12 PM', 'first'),
    Arrival_After6PM=('Arrival_After 6 PM', 'first'),
    Arrival_Before6AM=('Arrival_Before 6 AM', 'first'),
    Departure_12PM_6PM=('Departure_12 PM - 6 PM', 'first'),
    Departure_6AM_12PM=('Departure_6 AM - 12 PM', 'first'),
    Departure_After6PM=('Departure_After 6 PM', 'first'),
    Departure_Before6AM=('Departure_Before 6 AM', 'first'),
    Duration_in_hours=('Duration_in_hours', 'first'),
    Total_stops_1_stop=('Total_stops_1-stop','first'),
    Total_stops_2plus_stop=('Total_stops_2+-stop','first'),
    Total_stops_non_stop=('Total_stops_non-stop', 'first')
).reset_index()


print(df.head(10))

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import r2_score # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # type: ignore

models_dict = {}

aggregated_df.sort_values(by='Date_of_journey')

X=aggregated_df.drop(['total_passengers'],axis=1)
y=aggregated_df['total_passengers']

from sklearn.preprocessing import StandardScaler # type: ignore
scaler=StandardScaler()

X['Duration_in_hours'] = scaler.fit_transform(X[['Duration_in_hours']])
X['Days_left'] = scaler.fit_transform(X[['Days_left']])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

X_train

# subset = passengers_df
subset = aggregated_df[aggregated_df["Class_Economy"] == 1]
# subset = subset[subset["Airline"] == 0]
# plot_acf(subset['total_passengers'], lags=100)

from statsmodels.tsa.stattools import adfuller # type: ignore

dict = {}

result = adfuller(subset['total_passengers'])
dict["Results"] = (result[0], result[1])

'''except ValueError as e:
        print(f"ValueError for {destination}: {e}")
        dict[destination] = (None, None)'''

results_df = pd.DataFrame(dict).T
results_df.columns = ["ADF Statistic", "p-value"]

print('\n',results_df)

from statsmodels.tsa.arima.model import ARIMA# type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore
import pmdarima as pm # type: ignore

# this will crash my jupyter server, so i will do it locally
model = pm.auto_arima(subset['total_passengers'], seasonal=True, trace=True)
print(model.summary())