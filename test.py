#!/usr/bin/env python
# coding: utf-8

# # CHANGE PATH !!!!!!!!HERE!!!!!!!!!!!

# In[1]:


path = 'proj3_test.csv'


# In[2]:


import pandas as pd
import csv
import numpy as np
from collections import Counter 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import io
pd.set_option('display.max_columns', None)
from statistics import mean
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


# In[3]:


def file_read(filename):
    f = open(filename)
    csv_f = csv.reader(f)
    data = []
    for row in csv_f:
        data.append(row)
    return data


# In[4]:


def most_frequent(List): 
    counter = 0
    num = List[0] 
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 


# In[5]:


from scipy.fftpack import fft, ifft
def fft_and_peaks(df):
#     df = df[0]    
    fft_val=[]
    fft_val.append(abs(fft(df))) 
    peak_val=[]
    for z in range(len(fft_val)):
        a = list(set(fft_val[z]))
        a.sort()
        a = a[::-1][1:5]
        peak_val.append(a)
    return(fft_val,peak_val)


# In[6]:


def velocity(data):
#     data= data[0]
    data = data.reset_index().drop(columns = 'index').T
    mean = []
    std =[]
    median =[]
    
    interval = 10
    for k in range(len(data)):
        window_size = 4
        velocity = []
        row_data = data.iloc[k].values
        row_length = len(row_data)
        counter = 0
        cgmvel = []
        for i in range((len(row_data) - window_size)):
            cgmvel.append(counter)
            counter += 5
            p = (row_data[i] - row_data[i + window_size])
            vel = p / interval
            velocity.append(vel)
        mean.append(np.mean(velocity))
        std.append(np.std(velocity))
        median.append(np.median(velocity))
    df = list(zip(mean, std, median))[0]
    return df


# In[7]:


def min_max_glucose_level(df):
    df = df
    df = df[::-1]
    min_val = df.iloc[5]
    max_val = max(df.iloc[5:])
    diff = max_val - min_val
    return diff


# In[8]:


def full_width_at_half_maximum(df):
    y=df
    max_y = max(y)  # Find the maximum y value
    xs = [x for x in range(len(y)) if y[x] > ((max(y)/2)+55)]
    if(max(xs,default=1000)==1000):
        width = 0
    else:
        width = max(xs) - min(xs)
    return width


# In[9]:


def polyfit_coeffs(df):
    coeff_list = []
    y = df.to_numpy()
    time = np.linspace(1,df.shape[0],df.shape[0])
    coeff = np.polyfit(time,y,6)
    coeff_list.append(coeff)
    df1 = coeff_list[0]
    return df1


# In[10]:


def autocorrelation(df):   
    autocorr_lag_2=[]
    autocorr_lag_3=[]
    autocorr_lag_4=[]
    autocorr_lag_5=[]
    autocorr_lag_6=[]
    autocorr_lag_7=[]
    autocorr_lag_8=[]
    autocorr_lag_9=[]
    autocorr_lag_10=[]
    autocorr_lag_11=[]
    autocorr_lag_12=[]
    autocorr_lag_13=[]
    autocorr_lag_14=[]

    auto_corr_2= df.autocorr(lag=2)
    autocorr_lag_2.append(auto_corr_2)

    auto_corr_3= df.autocorr(lag=3)
    autocorr_lag_3.append(auto_corr_3)

    auto_corr_4= df.autocorr(lag=4)
    autocorr_lag_4.append(auto_corr_4)

    auto_corr_5= df.autocorr(lag=5)
    autocorr_lag_5.append(auto_corr_5)

    auto_corr_6= df.autocorr(lag=6)
    autocorr_lag_6.append(auto_corr_6)

    auto_corr_7= df.autocorr(lag=7)
    autocorr_lag_7.append(auto_corr_7)

    auto_corr_8= df.autocorr(lag=8)
    autocorr_lag_8.append(auto_corr_8)

    auto_corr_9= df.autocorr(lag=9)
    autocorr_lag_9.append(auto_corr_9)

    auto_corr_10=df.autocorr(lag=10)
    autocorr_lag_10.append(auto_corr_10)

    auto_corr_11= df.autocorr(lag=11)
    autocorr_lag_11.append(auto_corr_11)

    auto_corr_12= df.autocorr(lag=12)
    autocorr_lag_12.append(auto_corr_12)

    auto_corr_13= df.autocorr(lag=13)
    autocorr_lag_13.append(auto_corr_13)

    auto_corr_14= df.autocorr(lag=14)
    autocorr_lag_14.append(auto_corr_14)

    df1 = list(zip(autocorr_lag_2, autocorr_lag_3,autocorr_lag_4,autocorr_lag_5,autocorr_lag_6,autocorr_lag_7,autocorr_lag_8,autocorr_lag_9,autocorr_lag_10,autocorr_lag_11,autocorr_lag_12,autocorr_lag_13,autocorr_lag_14))[0]
    return (df1)


# In[11]:


def calculate_features(data):   
    fft_peak_val =[]
    cgm_vel = []
    index_list = []
    amplitude_list =[]
    width_list = []
    polyfit_coeff=[]
    auto_corr=[]

    for i in range(len(data)):
        intermediate = pd.DataFrame(data[i]).reset_index().drop(columns = 'index')
        each_row = pd.to_numeric(intermediate[0],errors='coerce').interpolate().dropna().reset_index().drop(columns='index')[0]
        
        if(len(each_row)>=15):
            ampl = min_max_glucose_level(each_row)
            amplitude_list.append(ampl)
            
            x,y = fft_and_peaks(each_row)
            fft_peak_val.append(y[0])
            
            v = velocity(each_row)
            cgm_vel.append(v)
            
            width = full_width_at_half_maximum(each_row)
            width_list.append(width)
            
            p = polyfit_coeffs(each_row)
            polyfit_coeff.append(p)
            
            acr = autocorrelation(each_row)
            auto_corr.append(acr)
            
        elif(len(each_row)<10):
            index_list.append(i)

    peak_val = pd.DataFrame(list(fft_peak_val),columns = ['Peak 2','Peak 3','Peak 4','Peak 5'])
    cgm_vel_val = pd.DataFrame(cgm_vel,columns=['vel_Mean','vel_STD','vel_Median'])
    ampl_val = pd.DataFrame(amplitude_list,columns=['Min_Max Diff'])
    width_val = pd.DataFrame(width_list,columns=['FWHM'])
    polyfit_val = pd.DataFrame(polyfit_coeff,columns=['coeff1','coeff2','coeff3','coeff4','coeff5','coeff6','coeff7'])
    autocorr_val = pd.DataFrame(auto_corr,columns=['autocorr_lag_2', 'autocorr_lag_3','autocorr_lag_4','autocorr_lag_5','autocorr_lag_6','autocorr_lag_7','autocorr_lag_8','autocorr_lag_9','autocorr_lag_10','autocorr_lag_11','autocorr_lag_12','autocorr_lag_13','autocorr_lag_14']) 
    
    dataset = pd.concat([cgm_vel_val,ampl_val,width_val,peak_val,polyfit_val],axis=1).fillna(0)
    return(dataset,index_list)


# # Reading Meal Data and Calculating the features

# In[12]:


meal_pat = file_read(path)
features_meal_pat,indices = calculate_features(meal_pat)


# In[13]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
feature_matrix = StandardScaler().fit_transform(features_meal_pat)
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(feature_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])


# In[14]:


output_matrix = pd.DataFrame()


# In[15]:


from joblib import dump, load
with open('KMeans_KNN.pickle', 'rb') as pre_trained_kmeans:
    pickle_file_kmeans = load(pre_trained_kmeans)
    predict_kmeans = pickle_file_kmeans.predict(principalDf)    
    pre_trained_kmeans.close()


# In[16]:


output_matrix['KMeans']=predict_kmeans


# In[17]:


from joblib import dump, load
with open('DBSCAN_KNN.pickle', 'rb') as pre_trained_dbscan:
    pickle_file_dbscan = load(pre_trained_dbscan)
    predict_dbscan = pickle_file_dbscan.predict(principalDf)    
    pre_trained_dbscan.close()


# In[18]:


output_matrix['DBSCAN']=predict_dbscan


# In[21]:


output_matrix[['DBSCAN','KMeans']].to_csv('output_matrix.csv',index=False,header=None)


# In[ ]:




