# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:33:25 2022

@author: norrs
"""

# ST2195 Coursework project

# 1: When is the best time of day, day of the week, and time of year to fly to minimise delays?

# using years 2001-3 from the Harvard Dataverse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as stats
from tabulate import tabulate
import dateutil.parser as dparser
import datetime
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer  # transform different types
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict


# import csv files for 2001-2003 as Pandas Dataframes
df2005 = pd.read_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/2005.csv", sep=',', encoding="iso-8859-1", low_memory=False)
df2003 = pd.read_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/2003.csv", sep=',', encoding="iso-8859-1", low_memory=False)
df2004 = pd.read_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/2004.csv", sep=',', encoding="iso-8859-1", low_memory=False)

# write the dataset to SQLite for easy browsing with DB Browser
 try:
    os.remove('flights.db')
 except OSError:
    pass
# browse datasets using sqlite3 & DB Browser
import sqlite3
conn = sqlite3.connect('flights.db')
df2005.to_sql('2005', con = conn, index = False)
df2003.to_sql('2003', con = conn, index = False)
df2004.to_sql('2004', con = conn, index = False)
sample.to_sql('sample', con = conn, index = False)
c = conn.cursor()

# check column structures are the same across all three dataframes
set(df2005.columns) == set(df2003.columns) == set(df2004.columns)

# True so concatenate the dataframes
flights = pd.concat([df2005, df2003, df2004], ignore_index=True)
pd.options.display.float_format = '{:.0f}'.format

# some summary data for the flights dataset
type(flights)
flights.info()
flights.info(memory_usage='deep')
flights.tail()
flights.describe()
# this is a very large data set that uses a lot of processing power and RAM, we should consider taking a sample of appropriate size to run analyses on more easily

# Status represents whether the flight was on time (0), slightly delayed (1), highly delayed (2), diverted (3), or cancelled (4)
for dataset in flights:
    flights.loc[flights['ArrDelay'] <= 15, 'Status'] = 0
    flights.loc[flights['ArrDelay'] >= 15, 'Status'] = 1
    flights.loc[flights['ArrDelay'] >= 60, 'Status'] = 2
    flights.loc[flights['Diverted'] == 1, 'Status'] = 3
    flights.loc[flights['Cancelled'] == 1, 'Status'] = 4

# show number of delays of each type
Status_counts_flights = flights.value_counts('Status')
# show as a percentage
percent_status_flights = Status_counts_flights*100/len(flights)
print(percent_status_flights)

# create a pie chart of delay status
f, ax = plt.subplots(1, 2, figsize=(20, 8))
flights['Status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Status')
ax[0].set_ylabel('')
sns.countplot('Status', order=flights['Status'].value_counts().index, data=flights, ax=ax[1])
ax[1].set_title('Status')
plt.show()
print('Status represents whether the flight was on time (0), slightly delayed (1), highly delayed (2), diverted (3), or cancelled (4)')
# 79.1% of flights arrived on time or with a delay of fewer than 15 minutes

# correlation matrix
corrmat1 = flights.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat1, vmax=.8, square=True)
plt.show()

# save to a single csv file for R analysis later
flights.to_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/flights.coursework.csv",index=None, na_rep='NULL')

# take a random sample of 10,000 values to simplify analysis
sample = flights.sample(n=10000, replace=True, weights=None, random_state=None, axis=None, ignore_index=True)
# sample summary data
sample.info()
sample.describe()
list(sample.columns)

# consider the sample number of flights that are delayed or cancelled
for dataset in sample:
    sample.loc[sample['ArrDelay'] <= 15, 'Status'] = 0
    sample.loc[sample['ArrDelay'] >= 15, 'Status'] = 1
    sample.loc[sample['ArrDelay'] >= 60, 'Status'] = 2
    sample.loc[sample['Diverted'] == 1, 'Status'] = 3
    sample.loc[sample['Cancelled'] == 1, 'Status'] = 4

# show number of delays of each type
Status_counts = sample.value_counts('Status')
# show as a percentage
percent_status = Status_counts*100/len(sample)
print(percent_status)

# create a pie chart of delay statuses
f, ax = plt.subplots(1, 2, figsize=(20, 8))
sample['Status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Status')
ax[0].set_ylabel('')
sns.countplot('Status', order=sample['Status'].value_counts().index, data=sample, ax=ax[1])
ax[1].set_title('Status')
plt.show()
print('Status represents whether the flight was on time (0), slightly delayed (1), highly delayed (2), diverted (3), or cancelled (4)')

# does the choice of sample seem a reasonable fit?
# The proportion of on time flights differs by only 0.01%. We assume that the sample is representative of the population proportions and so proceed with our analysis with n=10,0000
# RUN ANALYSIS ON PROPORTION MSE HERE

# CHI SQUARED TEST FOR GOODNESS OF FIT
# H0 : random sample, n=10,000 is a suitable estimator of the flights population
# H1 : random sample, n=10,000 is not a suitable estimator of the flights population
# set expected values as numeric variable means in the flight dataset
# set observed values as numeric variable means in the sample dataset
expected = flights[["CRSDepTime", "CRSArrTime", "DepTime", "ArrTime", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Distance", "Cancelled", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]].mean()
sum(list(expected))
observed = sample[["CRSDepTime", "CRSArrTime", "DepTime", "ArrTime", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Distance", "Cancelled", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]].mean()
sum(list(observed))
fexp = expected * (np.sum(observed)/np.sum(expected)) #scale the sums to match
stats.chisquare(f_obs=list(observed), f_exp=fexp) # = 0.0714
#critical value= 5.142 . Therefore accept the null, proceed with the sample.

# sample correlation matrix
corrmat2 = sample.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat2, vmax=.8, square=True)
plt.show()

# write to csv 
sample.to_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/sample.10000.flights.coursework.csv", index=None, na_rep='NULL')

# MISSING VALUES
missing = sample.isna().sum()
# find NAs as a % of each column
percent_missing = sample.isnull().sum() * 100 / len(sample)
missing_names = ['Number of Missing Values per Column', 'Percentage of Missing Values per Column']
print(tabulate([[missing, percent_missing]], headers=missing_names))


# there is a low proprtion of mission values in most columns, 
# but a very high proportion in Cancellation Code : drop this column 
sample = sample.drop("CancellationCode", 1)

# now impute missing values in ArrDelay, DepDelay, ElapsedTime, AirTIme etc??? WILL THEY ADD USEFUL INFO OR NOT
sample[["DepTime", "ArrTime", "ActualElapsedTime", "AirTime", "ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay","SecurityDelay", "LateAircraftDelay"]].fillna(sample[["DepTime", "ArrTime", "ActualElapsedTime", "AirTime", "ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay","SecurityDelay", "LateAircraftDelay"]].mean(), inplace=True)
#impute TailNum with mode
sample['TailNum'].fillna(sample['TailNum'].mode(), inplace=True)
sample.describe()

#### 
meanstats = sample[["DepTime", "ArrTime", "ActualElapsedTime", "AirTime", "ArrDelay", "DepDelay"]].mean()
meanstats
medianstats = sample[["DepTime", "ArrTime", "ActualElapsedTime", "AirTime", "ArrDelay", "DepDelay"]].median()
medianstats
np.subtract(meanstats, medianstats)
####

# considering the average difference between median and mean ArrDelay values is 6 minutes, assume we can proceed with mean as a good imputation value

# mean similar to median; assume that we can fill NAs with the mean expected value without adding too much noise

# longwinded way of executing line 204
sample['ActualElapsedTime'].fillna(sample['ActualElapsedTime'].mean(), inplace=True)
sample['DepTime'].fillna(sample['DepTime'].mean(), inplace=True)
sample['ArrTime'].fillna(sample['ArrTime'].mean(), inplace=True)
sample['ActualElapsedTime'].fillna(sample['ActualElapsedTime'].mean(), inplace=True)
sample['AirTime'].fillna(sample['AirTime'].mean(), inplace=True)
sample['ArrDelay'].fillna(sample['ArrDelay'].mean(), inplace=True)
sample['DepDelay'].fillna(sample['DepDelay'].mean(), inplace=True)
sample['CarrierDelay'].fillna(sample['CarrierDelay'].mean(), inplace=True)
sample['WeatherDelay'].fillna(sample['WeatherDelay'].mean(), inplace=True)
sample['NASDelay'].fillna(sample['NASDelay'].mean(), inplace=True)
sample['SecurityDelay'].fillna(sample['SecurityDelay'].mean(), inplace=True)
sample['LateAircraftDelay'].fillna(sample['LateAircraftDelay'].mean(), inplace=True)
sample['TailNum'].fillna(sample['TailNum'].mode(), inplace=True)

#____________

#1.  When is the best time of day, day of the week, and time of year to fly to minimise delays? 
# we will begin with / focus on arrival delays as this will be the factor most likely to affect passengers (delayed departures may not = delayed arrivals)
#barplot of mean delay at departure and arrival
fig = plt.figure(1, figsize=(11,6))
ax = sns.barplot(x="DepDelay", y="UniqueCarrier", data=sample, color="lightskyblue", ci=None)
ax = sns.barplot(x="ArrDelay", y="UniqueCarrier", data=sample, color="r", hatch = '///', alpha = 0.0, ci=None)
plt.xlabel('Mean delay [min] (@departure: blue, @arrival: hatch lines)', fontsize=14)


#plot Arrival Delays as timeseries data 
sample['datetime'] = pd.to_datetime(sample.Year*10000+sample.Month*100+sample.DayofMonth,format='%Y%m%d')
date_delay = sample[['datetime', 'ArrDelay']]
date_delay = date_delay.groupby(by='datetime').sum()
date_delay.head()
plt.figure(figsize=(25,8))
sns.lineplot(data=date_delay, palette="Set1", linewidth=0.5)


# plot to show the distribution of Arrival Delays in minutes
sns.displot(sample, x="ArrDelay")
sample['ArrDelay'].describe()
print("Skewness: %f" % sample['ArrDelay'].skew())
print("Kurtosis: %f" % sample['ArrDelay'].kurt())


# remove values outside the IQR to reduce the scale of the distribution temporarily
Q1 = sample['ArrDelay'].quantile(0.25)
Q3 = sample['ArrDelay'].quantile(0.75)
IQR = Q3 - Q1  # IQR is interquartile range.
filter = (sample['ArrDelay'] >= Q1 - 1.5 * IQR) & (sample['ArrDelay'] <= Q3 + 1.5 * IQR)
sample.loc[filter]

sns.displot(sample.loc[filter], x="ArrDelay") # Distribution plot with values outside IQR removed
#confirms that arrival delay counts are high around zero. 

# Visualising correlation between Month and ArrDelay:

# graph Delayed Flights by Arrival delays on minutes over months
DelayedFlightsMonths = sample[(sample.Status >= 1) & (sample.Status < 3)]
f, ax = plt.subplots(1, 2, figsize=(20, 8))
DelayedFlightsMonths[['Month', 'ArrDelay']].groupby(['Month']).mean().plot(ax=ax[0])
ax[0].set_title('Average delay time by month')
DelayedFlightsMonths[['Month', 'ArrDelay']].groupby(['Month']).sum().plot(ax=ax[1])
ax[1].set_title('Total Number of minutes delayed by month')
plt.show()

#bar chart of delays by Month
month_delay = sample[['Month', 'ArrDelay']]
month_delay = month_delay.groupby(by='Month').sum().reset_index(drop=False)
month_delay.head(12)
sns.barplot(x="Month", y="ArrDelay", data=month_delay, palette="Set1").set_title('Total Number of minutes delayed by month')
plt.show()

#represent this data in a table
AverageMonthly = DelayedFlightsMonths[['Month', 'ArrDelay']].groupby(['Month']).mean().sort_values(['ArrDelay'])
TotalMonthly = DelayedFlightsMonths[['Month', 'ArrDelay']].groupby(['Month']).sum().sort_values(['ArrDelay'])
Variance =  DelayedFlightsMonths[['Month', 'ArrDelay']].groupby(['Month']).var().sort_values(['ArrDelay'])
col_names = ['Average Monthly Delay', 'Total Minutes Delayed Per Month', 'Monthly Variance in Delay Times']
delay_monthly = print(tabulate([[AverageMonthly, TotalMonthly, Variance]], headers=col_names))

# now consider days of the week
# graph Delayed Flights by Arrival delays over days
DelayedFlightsDays = sample[(sample.Status >= 1) & (sample.Status < 3)]
f, ax = plt.subplots(1, 2, figsize=(15, 8))
DelayedFlightsDays[['DayOfWeek', 'ArrDelay']].groupby(['DayOfWeek']).mean().plot(ax=ax[0])
ax[0].set_title('Average delay by Day of Week')
DelayedFlightsDays[['DayOfWeek', 'ArrDelay']].groupby('DayOfWeek').sum().plot(ax=ax[1])
ax[1].set_title('Number of minutes delayed by Day')
plt.show()

#bar chart of delays by day 
day_delay = sample[['DayOfWeek', 'ArrDelay']]
day_delay = day_delay.groupby(by='DayOfWeek').sum().reset_index(drop=False)
sns.barplot(x="DayOfWeek", y="ArrDelay", data=day_delay, palette="Set1").set_title('Total Number of minutes delayed by day')
plt.show()

#represent this data in a table
AverageWeekly = DelayedFlightsDays[['DayOfWeek', 'ArrDelay']].groupby(['DayOfWeek']).mean().sort_values(['ArrDelay'])
TotalWeekly = DelayedFlightsDays[['DayOfWeek', 'ArrDelay']].groupby(['DayOfWeek']).sum().sort_values(['ArrDelay'])
VarianceWeekly =  DelayedFlightsDays[['DayOfWeek', 'ArrDelay']].groupby(['DayOfWeek']).var().sort_values(['ArrDelay'])
col_names_weekly = ['Average Delay per Day', 'Total Minutes Delayed Per Day', 'Variance in Daily Delays']
delay_weekly = print(tabulate([[AverageWeekly, TotalWeekly, VarianceWeekly]], headers=col_names_weekly))

#now consider time of day 
# graph Delayed Flights by Arrival delays over CRSDepartureTime
DelayedFlightsHours = sample[(sample.Status >= 1) & (sample.Status < 3)]
f, ax = plt.subplots(1, 2, figsize=(15, 8))
DelayedFlightsHours[['CRSDepTime', 'ArrDelay']].groupby(['CRSDepTime']).mean().plot(ax=ax[0])
ax[0].set_title('Average arrival delay by Scheduled Departure Time')
DelayedFlightsHours[['CRSDepTime', 'ArrDelay']].groupby('CRSDepTime').sum().plot(ax=ax[1])
ax[1].set_title('Number of minutes delayed by Scheduled Departure Time')
plt.show()

#lots of data. Let's plot a jointplot with regression line instead 
sns.jointplot(x='CRSDepTime',y='ArrDelay',data=DelayedFlightsHours)
sns.jointplot(x='CRSDepTime',y='ArrDelay',data=DelayedFlightsHours, kind='reg',fit_reg = True)
sns.regplot(x=sample["CRSDepTime"], y=sample["ArrDelay"], scatter_kws={'s': 0.5}, fit_reg=True)
type('ArrDelay')

# represent this data in a table
AverageHourly = DelayedFlightsHours[['CRSDepTime', 'ArrDelay']].groupby(['CRSDepTime']).mean().sort_values(['ArrDelay'])
TotalHourly = DelayedFlightsHours[['CRSDepTime', 'ArrDelay']].groupby(['CRSDepTime']).sum().sort_values(['ArrDelay'])
VarianceHourly =DelayedFlightsHours[['CRSDepTime', 'ArrDelay']].groupby(['CRSDepTime']).var().sort_values(['ArrDelay'])
col_names_hourly = ['Average Delay per Hour', 'Total Minutes Delayed Per Hour', 'Variance in Hourly Delays']
delay_hourly = print(tabulate([[AverageHourly, TotalHourly, VarianceHourly]], headers=col_names_hourly))
#not very useful: too many distinct Departure Times. =


#_____________________________________________

#2. Do older planes suffer more delays? 

#import plane data
plane_data = pd.read_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/plane-data.csv", sep=',', encoding="iso-8859-1")

plane_data.describe() # additional data on tailnums and associated issue date
plane_data.head()
plane_data.sort_values(['issue_date']) # there are a number of missing values 
plane_data.rename(columns={'tailnum':'TailNum'}, inplace=True) #rename ytailnum column to match sample column name

#review NAs
plane_missing = plane_data.isna().sum()
percent_missing_planes = plane_data.isnull().sum() * 100 / len(sample)
missing_planenames = ['Number of Missing Values per Column', 'Percentage of Missing Values per Column']
print(tabulate([[plane_missing, percent_missing_planes]], headers=missing_planenames))

#missing values in the dataframe tell us nothing about the planes associated with the tailnums. it makes no sense to impute them, so drop these
plane_data.dropna(inplace = True)

#remove row(s) with None in issue_date
plane_data.loc[plane_data['issue_date']=='None'] #1 row with None issue_date
plane_data.drop(1529, inplace=True)

#transform issue date to datetime format
plane_data['issuedate'] = pd.to_datetime(plane_data['issue_date'], format='%m/%d/%Y')

#merge TailNum data, carriers and Issue date into one df
left = sample[["UniqueCarrier", "datetime", "TailNum", "ArrDelay", "Status", "LateAircraftDelay", "CarrierDelay"]]
right = plane_data[["TailNum", "issuedate", "year"]]
tailnums = pd.merge(left, right, how='left', on='TailNum')
tailnums.head()
tailnums.describe()

#count unique items
tailnums.value_counts()
tailnums.sort_values(by='issuedate')
tailnums.isna().sum() #there are a considerable number of missing inputs for datetime and year. Are they unique?
tailnums.isna().head()
#index by date 
tailnums = tailnums.set_index('issuedate')
#group by decades
tailnums_decades = tailnums.resample('10AS').sum()

#average delays by decade for general view(although this is clunky and loses a lot of nuance / information)
tailnums_dec_mean = tailnums.resample('10AS').mean()
#reset indexes
tailnums.reset_index(inplace=True)
tailnums_decades.reset_index(inplace=True)
tailnums_dec_mean.reset_index(inplace=True)
#check indexes are reverted
tailnums_decades
tailnums_dec_mean
tailnums.head()
#drop nas
#tailnums.dropna(inplace = True)

#scatter plot of manufacture year vs Arrival Delays

#plot Arrival Delay times by years
years = tailnums.sort_values(by='year')
years
f, ax = plt.subplots(1, 2, figsize=(30, 8))
scatter_ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(data=years, x='year', y='ArrDelay').plot(ax=ax[0])
ax[0].set_title('Arrival delays by Year of Tail Number issue')


#check that year can be used in place of issue_date for simpler graphs 
sns.jointplot(data=years, x='year', y='issuedate', kind='reg').plot(ax=ax[1])
#strong correlation therefore assume okay 

#lineplot of delay status by manufacture year
sns.lineplot(data=years, x='year', y='ArrDelay', ci=50, palette="Set1", linewidth=1.5, hue='Status').plot(ax=ax[1])
ax[1].set_title('Line plot of Arrival delays by Tail Num Year')
plt.show()

DelayedFlights = tailnums[(tailnums.Status >= 1) & (tailnums.Status < 3)]
f, ax = plt.subplots(1, 2, figsize=(25, 8))
DelayedFlights[['issuedate', 'ArrDelay']].groupby(['issuedate']).plot(ax=ax[0])
ax[0].set_title('Arrival delays by Issue Date')
DelayedFlights[['year', 'ArrDelay']].groupby('year').plot(ax=ax[1])
ax[1].set_title('Arrival Delays by Year')
plt.show()


#________________________
#3. How does the number of people flying between different locations change over time?

import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
import datetime, warnings, scipy
warnings.filterwarnings("ignore")
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools


airports = pd.read_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/airports.csv", sep=',', encoding="iso-8859-1")
sample = pd.read_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/sample.10000.flights.coursework.csv", sep=',', encoding="iso-8859-1")
carriers = pd.read_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/carriers.csv", sep=',', encoding="iso-8859-1")

#plot all flights over time 
totalflights = sample['datetime'].value_counts()
totalflights   
plt.figure(figsize=(25,6))
sns.lineplot(data=totalflights, x=totalflights.index, y=totalflights.values).plot(ax=ax[1])
plt.title('Number of flights over time', fontsize=16)
plt.xlabel('date', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.show()

#find the 15 most common Destinations
Destination_top = sample['destiata'].value_counts().sort_values(ascending=False).head(15)
Destination_top #ATL top destination

#display in barplot
sns.barplot(x=Destination_top.index, y=Destination_top.values)
plt.grid(True, color='g', linewidth=1)
plt.show()

Destination_top.to_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/top_destinations.csv")

#display as grid
Dest_Count = sample.groupby(['destiata'],as_index=False).agg({'Month':'count'})
max_dest_count = Dest_Count.sort_values(['Month'], ascending=False)
print("The top_10 destination flights are")
top_dest_flight = max_dest_count.head(10)
plt.scatter(top_dest_flight.destiata, top_dest_flight.Month, color='red')
plt.legend
plt.grid(True, color='g', linewidth=1)
plt.show()
top_dest_flight

#15 most common Origin Airports
Origin_top15 = sample['originiata'].value_counts().sort_values(ascending=False).head(15)
originairports['originiata'].value_counts().sort_values(ascending=False).head(15)
print(Origin_top15) #ATL comes out top again

#barplot of top origins 
sns.barplot(x=Origin_top15.index, y=Origin_top15.values)
plt.grid(True, color='g', linewidth=1)
plt.show()

Origin_top15.to_csv("C:/Users/norrs/OneDrive/Documents/UNIVERSITY/COURSEWORK/top_origins.csv")

#display in grid
Origin_Count = sample.groupby(['originiata'],as_index=False).agg({'Month':'count'})
max_orig_count = Origin_Count.sort_values(['Month'], ascending=False)
print("The top_10 departure airports are")
top_orig_flight = max_orig_count.head(10)
plt.scatter(top_orig_flight.originiata, top_orig_flight.Month, color='red')
plt.legend
plt.grid(True, color='g', linewidth=1)
plt.show()
top_orig_flight

#number of unique origins / destinations
sample['originiata'].describe() #249 unique origins
sample['destiata'].describe() #250 unique destinations

#take top 5 departure airports : ATL, ORD, DFW, LAX, IAH

#find 602 rows where ATL is origin 
ATL = sample[sample['originiata'].eq('ATL')]
intoATL = sample[sample['destiata'].eq('ATL')]
type(ATL)
destATL = ATL['destiata'].value_counts().sort_values(ascending=False).head(10)
#number of flights out of ATL per month
monthATL = ATL['Month'].value_counts().sort_values(ascending=False).head(12) # it is a series
monthATL
#number of flights into ATL each month
monthintoATL = intoATL['Month'].value_counts().sort_values(ascending=False).head(12)
monthintoATL

#plot flights into and out of ATL
f, ax = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(data=monthATL, x=monthATL.index, y=monthATL.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[0])
ax[0].set_title('Number of flights out of ATL by month')
sns.lineplot(data=monthintoATL, x=monthintoATL.index, y=monthintoATL.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into / out of ATL by month')
plt.show()

#now let's look at ORD
ORD = sample[sample['originiata'].eq('ORD')]
intoORD = sample[sample['destiata'].eq('ORD')]
type(ORD)
#most popular destinations from ORD
destORD = ORD['destiata'].value_counts().sort_values(ascending=False).head(10)
#number of flights out of ORD per month
monthORD = ORD['Month'].value_counts().sort_values(ascending=False).head(12) # it is a series
monthORD
#number of flights into ATL each month
monthintoORD = intoORD['Month'].value_counts().sort_values(ascending=False).head(12)
monthintoORD

#plot flights into and out of ORD
f, ax = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(data=monthORD, x=monthORD.index, y=monthATL.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[0])
ax[0].set_title('Number of flights out of Chicago by month')
sns.lineplot(data=monthintoORD, x=monthintoORD.index, y=monthintoORD.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into / out of Chicago by month')
plt.show()

#compare flights into ATL with ORD
f, ax = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(data=monthintoATL, x=monthintoATL.index, y=monthintoATL.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into / out of ATL by month')
sns.lineplot(data=monthintoORD, x=monthintoORD.index, y=monthintoORD.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into Atlanta and Chicago by month')
plt.show()

#compare flights out of ATL with ORD
f, ax = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(data=monthATL, x=monthATL.index, y=monthATL.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into / out of ATL by month')
sns.lineplot(data=monthORD, x=monthORD.index, y=monthORD.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into Atlanta and Chicago by month')
plt.show()

#let's look at one more airport : DFW

DFW = sample[sample['originiata'].eq('DFW')]
intoDFW = sample[sample['destiata'].eq('DFW')]
type(DFW)
#most popular destinations from ORD
destDFW = DFW['destiata'].value_counts().sort_values(ascending=False).head(10)
#number of flights out of ORD per month
monthDFW = DFW['Month'].value_counts().sort_values(ascending=False).head(12) # it is a series
monthDFW
#number of flights into ATL each month
monthintoDFW = intoDFW['Month'].value_counts().sort_values(ascending=False).head(12)
monthintoDFW

#plot flights into and out of DFW
f, ax = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(data=monthDFW, x=monthDFW.index, y=monthDFW.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[0])
ax[0].set_title('Number of flights out of Dallas by month')
sns.lineplot(data=monthintoDFW, x=monthintoDFW.index, y=monthintoDFW.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into / out of Dallas by month')
plt.show()

#compare flights out of ATL with ORD and DFW
f, ax = plt.subplots(1,2, figsize=(30, 8))
sns.lineplot(data=monthATL, x=monthATL.index, y=monthATL.values, ci=50, palette="Set3", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into / out of ATL by month')
sns.lineplot(data=monthDFW, x=monthDFW.index, y=monthDFW.values, ci=50, palette="Set3", linewidth=1.5).plot(ax=ax[0])
ax[0].set_title('Number of flights out of Dallas by month')
sns.lineplot(data=monthORD, x=monthORD.index, y=monthORD.values, ci=50, palette="Set3", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights out of Atlanta, Dallas and Chicago by month')
plt.show()

#compare flights out of ATL with ORD and DFW
f, ax = plt.subplots(1,2, figsize=(30, 8))
sns.lineplot(data=monthintoATL, x=monthintoATL.index, y=monthintoATL.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into / out of ATL by month')
sns.lineplot(data=monthintoDFW, x=monthintoDFW.index, y=monthintoDFW.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[0])
ax[0].set_title('Number of flights out of Dallas by month')
sns.lineplot(data=monthintoORD, x=monthintoORD.index, y=monthintoORD.values, ci=50, palette="Set1", linewidth=1.5).plot(ax=ax[1])
ax[1].set_title('Number of flights into Atlanta, Dallas and Chicago by month')
plt.show()

#create new df of origin airports: merge sample with airports on origin airport
originairports = airports.rename(columns={'iata':'originiata'}, inplace=True) #rename airports column to match sample column name
leftairports = sample[["UniqueCarrier", "originiata", "datetime"]]
rightairports = airports[["originiata", "airport", "city", "state", "lat","long"]]
originairports = pd.merge(leftairports, rightairports, how='left', on='originiata')
originairports.describe()

#create new df of destination airport codes
destairports = airports.rename(columns={'originiata':'destiata'}, inplace=True) 
sample.rename(columns={'Dest':'destiata'}, inplace=True) 
leftairportdest = sample[["UniqueCarrier", "destiata", "datetime"]]
rightairportdest = airports[["destiata", "airport", "city", "state", "lat","long"]]
destairports = pd.merge(leftairportdest, rightairportdest, how='left', on='destiata')
destairports.head()





#__________________________________

## 4. Can you detect cascading failures as delays in one airport create delays in others? 
import collections
import datetime
from pyspark.context import SparkContext

#plot Arrival Delays as with other Variables as timeseries data
sample['datetime'] = pd.to_datetime(sample.Year*10000+sample.Month*100+sample.DayofMonth,format='%Y%m%d')
date_delay = sample[['datetime', 'LateAircraftDelay', 'ArrDelay']]
date_delay = date_delay.groupby(by='datetime').sum()
date_delay.head()
plt.figure(figsize=(25,8))
sns.lineplot(data=date_delay, palette="Set1", linewidth=0.5)

#joint plot to demonstrate correlation between ArrDelay and LateAircraftDelay
sns.jointplot(x='LateAircraftDelay',y='ArrDelay', data=date_delay,kind='reg', color='b',fit_reg = True)
sns.jointplot(x='LateAircraftDelay',y='ArrDelay', data=sample,kind='reg', color='b',fit_reg = True)

tailnums['LateAircraftDelay'].value_counts().sort_values(ascending=False).describe()
#there are 145 cases of LateAircraftDelay in the dataset 


#group tailnums by LateAircraftDelay times
late = tailnums.sort_values('LateAircraftDelay', ascending=False).head(145)
late.describe()
#select flights delayed by 107 minutes, the mean LateAircraftDelay time


#extract rows at random  to examine the tail Number
late.iloc[[67]]#TailNum = N614, ArrDelay = 250 minutes 
late[late['TailNum'].eq('N806FR')].value_counts()

#link tailnum to other flights on the same day 
sample_model['delaystatus'] = delaystatus
sample_model.value_counts('delaystatus') 


#lineplot of delay causes by datetime
df = sample.filter(['datetime','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay'], axis=1)
df = df.groupby('datetime')['LateAircraftDelay','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay'].sum().plot()
df.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
plt.show()

#lineplot of delay causes by Month
df2 = sample.filter(['Month','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay'], axis=1)
df2 = df2.groupby('Month')['LateAircraftDelay','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay'].sum().plot()
df2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
plt.show()

#scatterplot matrix
sns.set()
cols = ['ArrDelay', 'CarrierDelay', 'LateAircraftDelay', 'NASDelay', 'WeatherDelay']
sns.pairplot(sample[cols], size = 2.5)
plt.show()

#____________________________________


#5. Model to predict delays 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.optimize import curve_fit
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml  # using openml to import data
from sklearn.metrics import plot_roc_curve, accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn import utils
from sklearn import linear_model
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC

#create a new 'Status'-like variable: a binary measure of delay
delaystatus=[]
for row in sample_model['ArrDelay']:
  if row > 15:
    delaystatus.append(1)
  else:
    delaystatus.append(0)  
sample_model['delaystatus'] = delaystatus
sample_model.value_counts('delaystatus') #8236 are ontime, 1764 are delayed

#visualise delay distribution
counts = sample_model.value_counts('delaystatus') 
print(counts)
plt.figure(figsize=(10,6))
sns.barplot(x=counts.index, y=counts.values)
plt.title('Delay Distribution', fontsize=16)
plt.xlabel('Flight Status', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.xticks(range(len(counts.index)), ['ON TIME(0)', 'DELAYED(1)'])
plt.show()

#revert to original sample to properly split train / test datasets
sample = flights.sample(n=10000, replace=True, weights=None, random_state=None, axis=None, ignore_index=True)

# MODEL 1: DecissionTreeClassifier
#numerical features only. drop nonnumerical features
sample_model = sample.drop(['Year', 'Origin', 'Dest', 'FlightNum','TailNum', 'UniqueCarrier', 'TaxiOut', 'TaxiIn', 'Status', 'Cancelled', 'CancellationCode', 'ArrTime', 'ActualElapsedTime','WeatherDelay', 'CarrierDelay', 'NASDelay', 'LateAircraftDelay', 'SecurityDelay', 'Diverted'],axis=1)
sample_model.info()

#we want to predict Arrival Delays
sample_model = sample_model.drop(['ArrTime', 'ArrDelay'], axis=1)
sample_model.info()

#impute missing values 
sample_model.isna().sum()
sample_model['DepTime'].fillna(sample_model['DepTime'].mean(), inplace=True)
sample_model['AirTime'].fillna(sample_model['AirTime'].mean(), inplace=True)
sample_model['ArrDelay'].fillna(sample_model['ArrDelay'].mean(), inplace=True)
sample_model['DepDelay'].fillna(sample_model['DepDelay'].mean(), inplace=True)

#split for train & test 

data = sample_model.values
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)  # splitting in the ratio 70:30
scaled_features = StandardScaler().fit_transform(X_train, X_test)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_test = enc.fit_transform(y_test)

pred_prob = clf.predict_proba(X_test)
auc_score = roc_auc_score(y_test, pred_prob[:,1])
auc_score #0.953 , 95.3% accuracy

#run linear model types 
classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

for item in classifiers :
    print(item)
    clf = item
    clf = clf.fit(X_train,y_train)
    clf = clf.predict(X_test),'\n'
    print(clf)
    
#this is a nice loop but graphing the auc curve is difficult. 
#Let's try to separate out models more
 
#MODEL IDEA 2 
   
# logistic regression
clf1 = LogisticRegression()
clf1 = clf1.fit(X_train,y_train)

y_test = enc.fit_transform(y_test)
y_train = enc.fit_transform(y_train)

pred_prob = clf1.predict_proba(X_test)
auc_score = roc_auc_score(y_test, pred_prob[:,1])
auc_score #0.953 , 95.3% accuracy

#bayesian ridge reg
 model2 = BayesianRidge()
#fit
model1.fit(X_train, y_train)
model2.fit(X_train, y_train) 
#predict   
pred_prob1 = model1.predict_proba(X_test)
pred_prob2 = model2.predict(X_test)
    
# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
print(auc_score1, auc_score2)

# MODEL 3

sample_model3 = sample.drop(['Year','FlightNum','TailNum','TaxiOut', 'TaxiIn', 'Status', 'Cancelled', 'CancellationCode', 'ActualElapsedTime','WeatherDelay', 'Origin', 'Dest','CarrierDelay', 'NASDelay', 'LateAircraftDelay',  'SecurityDelay', 'Diverted'],axis=1)

sample_model3.features = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay']

# Building pipelines
#different pre-processing steps for continuous (numerical) and categorical features.

numerical_features = [ 'CRSArrTime','ArrDelay', 'DepDelay', 'delaystatus']

# For numerical features (age and fare) we apply imputation on the missing values with SimpleImputer() and scale the values
# Applying SimpleImputer and StandardScaler into a pipeline 

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

# For categorical features, use OneHotEcoder() to create dummy variables
categorical_features = ['Month', 'DayofMonth', 'UniqueCarrier']

# Applying SimpleImputer and then OneHotEncoder into another pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#pre-processed the numerical / categorical we can merge them with ColumnTransformer()
data_transformer = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)]) 
data_transformer 

#split into test & train
data = sample_model3.values
type(data) #numpy array
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=1) #start splitting in the ratio 50:50


#encode continuous variables to multiclass
enc = preprocessing.LabelEncoder()
y_train = enc.fit_transform(y_train)
y_test  = enc.fit_transform(y_test)


param_grid = {
    'data_transformer__numerical__imputer__strategy': ['mean', 'median'],
    'data_transformer__categorical__imputer__strategy': ['constant','most_frequent']
}

#Logistic Regression
pipe_lr = Pipeline(steps=[('data_transformer', data_transformer),
                      ('pipe_lr', LogisticRegression(max_iter=10000, penalty = 'none'))])
grid_lr = GridSearchCV(pipe_lr,
grid_lr.fit(X_train, y_train))

#DecisionTree Classifier
pipe_tree = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_tree', DecisionTreeClassifier(random_state=0))])
grid_tree = GridSearchCV(pipe_tree, param_grid=param_grid)
grid_tree.fit(X_train, y_train);

#Gradient Boosting
pipe_gdb = Pipeline(steps=[('data_transformer', data_transformer),
       ('pipe_gdb',GradientBoostingClassifier(random_state=2))])

grid_gdb = GridSearchCV(pipe_gdb, param_grid=param_grid)
grid_gdb.fit(X_train, y_train);

#Random Forests 
pipe_rf = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_rf', RandomForestClassifier(random_state=0))])
grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid)
grid_rf.fit(X_train, y_train);

#Support vector machines
pipe_svm = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_svm',  LinearSVC(random_state=0, max_iter=10000, tol=0.01))])
grid_svm = GridSearchCV(pipe_svm, param_grid=param_grid)
grid_svm.fit(X_train, y_train);

#compare the performance of models by ROC curve
ax = plt.gca()
plot_roc_curve(grid_lr, X_test, y_test, ax=ax, name='Logistic Regression')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.show()

plot_roc_curve(grid_gdb, X_test, y_test, ax=ax, name='Gradient Boosting')
#plot_roc_curve(grid_plr, X_test, y_test, ax=ax, name='Penalised logistic regression')
plot_roc_curve(grid_tree, X_test, y_test, ax=ax, name='Classification trees')
plot_roc_curve(grid_rf, X_test, y_test, ax=ax, name='Random forests')
#plot_roc_curve(grid_gp, X_test, y_test, ax=ax, name='Gaussian process classification')
plot_roc_curve(grid_svm, X_test, y_test, ax=ax, name='Support vector machines')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.show()
