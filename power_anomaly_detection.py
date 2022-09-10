from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os
import datetime
# from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.manifold import MDS
# from sklearn.preprocessing import Imputer
# from sklearn.cluster import KMeans
import math

DIR='/content/drive/MyDrive/Colab Notebooks/Data'
FILENAME='household_power_consumption.txt'
os.chdir(DIR)

df=pd.read_csv(FILENAME,sep=';',index_col=None,header=0)

df = pd.read_csv(FILENAME, sep=";", header=0)
df['Date'] = pd.to_datetime(df["Date"],format="%d/%m/%Y")
df['Time'] = pd.to_datetime(df["Time"],format="%H:%M:%S")

df.drop(df[df['Global_active_power'] == "?"].index, inplace = True)
df["Global_active_power"] = df[["Global_active_power"]].apply(pd.to_numeric)
df.head()

yr=2007
df_2007=df[(df['Date'].dt.year==yr)&(df['Time'].dt.minute%30==0)][['Date','Time','Global_active_power']] 
df_2007.head()

start=datetime.datetime(yr, 1, 1)
end = datetime.datetime(yr, 12, 31)
tempdf=pd.DataFrame(pd.date_range(start,end,freq='0.5H'),columns=['column'])
tempdf['column'].dt.date
tempdf = pd.DataFrame([tempdf['column'].dt.date,tempdf['column'].dt.time]).T
tempdf.columns=['Date','Time']
tempdf["Time"] = tempdf["Time"].astype(str)
tempdf["Date"] = tempdf["Date"].astype(str)

df_2007["Time"] = df_2007["Time"].astype(str)
df_2007["Time"]=df_2007["Time"].str.replace('1900-01-01 ','')
df_2007["Date"] = df_2007["Date"].astype(str)


df_2007_updated=pd.merge(tempdf,df_2007,how='left',left_on=['Date','Time'],right_on=['Date','Time'])
df_2007_updated['Global_active_power'].isnull().sum()

df_2007_updated['Global_active_power']=df_2007_updated['Global_active_power'].interpolate(method='linear')

df_2007_updated['Date']=pd.to_datetime(df_2007_updated["Date"],format="%Y-%m-%d")
df_2007_updated['Time']=pd.to_datetime(df_2007_updated["Time"],format="%H:%M:%S")
df_2007_updated.head()#changes the format of date and time in df_2007_updated

groupdates=df_2007_updated.groupby("Date")
print(groupdates)

#fourier transform

func = lambda x: fft(x['Global_active_power'].values)
fft_transformed_data=np.abs(groupdates.apply(func))
#print(fft_transformed_data.values)
distance_matrix=[]
for i in fft_transformed_data.values:
  tmp=[]
  for j in fft_transformed_data.values:
    tmp.append(np.linalg.norm(i-j))
  distance_matrix.append(tmp)
  
  print(distance_matrix[0])


#using fft the time domain series changes to frequency domain.using normalisation method the norm values is calculated and appended to another matrix called distance matrix having value 365*365 


  model=MDS(n_components=2,dissimilarity='precomputed',random_state=1)
mds_out=model.fit_transform(distance_matrix)
#dates=df_2007_updated['Date'].dt.date.values
x=mds_out[:, 0]
y=mds_out[:, 1]
plt.figure(figsize=(15,10))
plt.scatter(x,y)
for i in np.arange(len(x)):
  try:
    plt.annotate((i+1),(x[i]+0.3,y[i]+0.3))
  except:
    continue
plt.title(yr)
plt.show()


df_distance=pd.DataFrame(distance_matrix)
df_distance.head(5)

radius=[]
k=19
for i in np.arange(len(df_distance)):
  sorted_values=df_distance.sort_values(by=i,axis=1).iloc[i:i+1,1:k]
  radius.append(sorted_values.iloc[:, -1].values)
radius

#in 2 dimensional space ,the nearest neighbour technique is used in order to sorted the highest nearest neighbour with in 19 (sqrt(365)).when d=2,we have area and takes the area and find a single radius value taking mean of it.


df_probability=pd.DataFrame()

df_probability['Date Index']=np.arange(365)+1

df_probability['Date']=pd.date_range(start,end)
density=(k/((math.pi)*(np.array(radius)**2)))
df_probability['probability']=1-(density/max(density))

df_probability.head()

df_probability.loc[62,'probability']
df_probability.loc[76,'probability']

df_probability_anomalous=df_probability[df_probability['probability']>=0.85]

(len(df_probability_anomalous)/365)*100

df_probability_anomalous['Month']=df_probability_anomalous['Date'].dt.month.values
#month=df_probability_anomalous['Date'].dt.month.values

df_probability_anomalous.head()
#seprates month from the date and grouped the iterated probability values based on month ploted the number of days in a month which showing anomaly in power as histogram.
group=df_probability_anomalous.groupby('Month')

func= lambda x:len(x)
group.apply(func)


import matplotlib.pyplot as plt
x=df_probability_anomalous['Month']
y=len(x)
plt.hist(x,y)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel("Months")
plt.ylabel("Number of days")
plt.title("power anomalous detection for year 2007")
plt.show()