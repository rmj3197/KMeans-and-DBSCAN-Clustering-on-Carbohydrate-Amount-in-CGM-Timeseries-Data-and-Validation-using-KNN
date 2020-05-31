#!/usr/bin/env python
# coding: utf-8

# # Raktim Mukhopadhyay 1217167380

# In[1]:


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


# In[2]:


def most_frequent(List): 
    counter = 0
    num = List[0] 
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 


# In[3]:


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


# In[4]:


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


# In[5]:


def min_max_glucose_level(df):
    df = df
    df = df[::-1]
    min_val = df.iloc[5]
    max_val = max(df.iloc[5:])
    diff = max_val - min_val
    return diff


# In[6]:


def full_width_at_half_maximum(df):
    y=df
    max_y = max(y)  # Find the maximum y value
    xs = [x for x in range(len(y)) if y[x] > ((max(y)/2)+55)]
    if(max(xs,default=1000)==1000):
        width = 0
    else:
        width = max(xs) - min(xs)
    return width


# In[7]:


def polyfit_coeffs(df):
    coeff_list = []
    y = df.to_numpy()
    time = np.linspace(1,df.shape[0],df.shape[0])
    coeff = np.polyfit(time,y,6)
    coeff_list.append(coeff)
    df1 = coeff_list[0]
    return df1


# In[8]:


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


# In[9]:


def file_read(filename):
    f = open(filename)
    csv_f = csv.reader(f)
    data = []
    for row in csv_f:
        data.append(row)
    return data


# In[10]:


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

# In[11]:


meal_pat1 = file_read('mealData1.csv')
features_meal_pat1,indices_1 = calculate_features(meal_pat1)


# In[12]:


meal_pat2 = file_read('mealData2.csv')
features_meal_pat2,indices_2 = calculate_features(meal_pat2)


# In[13]:


meal_pat3 = file_read('mealData3.csv')
features_meal_pat3,indices_3 = calculate_features(meal_pat3)


# In[14]:


meal_pat4 = file_read('mealData4.csv')
features_meal_pat4,indices_4 = calculate_features(meal_pat4)


# In[15]:


meal_pat5 = file_read('mealData5.csv')
features_meal_pat5,indices_5 = calculate_features(meal_pat5)


# In[16]:


carb_levels_pat1 = pd.read_csv('mealAmountData1.csv',header=None)[0:51].rename(columns={0:'Carb Level'})
carb_levels_pat2 = pd.read_csv('mealAmountData2.csv',header=None)[0:51].rename(columns={0:'Carb Level'})
carb_levels_pat3 = pd.read_csv('mealAmountData3.csv',header=None)[0:51].rename(columns={0:'Carb Level'})
carb_levels_pat4 = pd.read_csv('mealAmountData4.csv',header=None)[0:51].rename(columns={0:'Carb Level'})
carb_levels_pat5 = pd.read_csv('mealAmountData5.csv',header=None)[0:51].rename(columns={0:'Carb Level'})


# In[17]:


carb_levels_pat1=carb_levels_pat1.drop(indices_1).reset_index().drop(columns=['index'])
carb_levels_pat2=carb_levels_pat2.drop(indices_2).reset_index().drop(columns=['index'])
carb_levels_pat3=carb_levels_pat3.drop(indices_3).reset_index().drop(columns=['index'])
carb_levels_pat4=carb_levels_pat4.drop(indices_4).reset_index().drop(columns=['index'])
carb_levels_pat5=carb_levels_pat5.drop(indices_5).reset_index().drop(columns=['index'])


# In[18]:


meal_data = pd.concat([features_meal_pat1,features_meal_pat2,features_meal_pat3,features_meal_pat4,features_meal_pat5],axis=0).reset_index().drop(columns=['index'])


# In[19]:


carb_levels = pd.concat([carb_levels_pat1,carb_levels_pat2,carb_levels_pat3,carb_levels_pat4,carb_levels_pat5],axis=0).reset_index().drop(columns=['index'])


# In[20]:


dataset = pd.concat([meal_data,carb_levels],axis=1)


# In[21]:


def create_bin(x):
    if (x==0):
        x=1
    elif ((x>0) and (x<=20)):
        x = 2
    elif((x>=21) and (x<=40)):
        x = 3
    elif((x>=41) and (x<=60)):
        x = 4
    elif((x>=61) and (x<=80)):
        x = 5
    elif((x>=81) and (x<=100)):
        x = 6
    elif((x>=101) and (x<=120)):
        x = 7
    elif((x>=121) and (x<=140)):
        x = 8
    else:
        pass
    return x


# In[22]:


dataset['Carb Level'] = dataset['Carb Level'].apply(create_bin)


# In[23]:


dataset_for_clustering = dataset.drop(columns=['Carb Level'])


# In[24]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
feature_matrix = StandardScaler().fit_transform(dataset_for_clustering)
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(feature_matrix)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])


# # K-Means Clustering

# In[25]:


original_bin_ = dataset['Carb Level'].reset_index().drop(columns='index').rename(columns={'Carb Level':'Original Label'})


# In[26]:


def label_reassignment(kmeans_labels,original_bin_labels):
    
    data_labels_after_kmeans = kmeans_labels
    data_labels_before_kmeans = original_bin_labels
    
    label_0 = []
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    label_5 = []

    for i in range(data_labels_after_kmeans.shape[0]):
        if (list(data_labels_after_kmeans.iloc[i])[0] == 0):
            label_0.append(i)
        elif (list(data_labels_after_kmeans.iloc[i])[0] == 1):
            label_1.append(i)
        elif (list(data_labels_after_kmeans.iloc[i])[0] == 2):
            label_2.append(i)
        elif (list(data_labels_after_kmeans.iloc[i])[0] == 3):
            label_3.append(i)
        elif (list(data_labels_after_kmeans.iloc[i])[0] == 4):
            label_4.append(i)
        elif (list(data_labels_after_kmeans.iloc[i])[0] == 5):
            label_5.append(i)
        else:
            pass
    
    orig_lable_cluster_0 = []
    orig_lable_cluster_1 = []
    orig_lable_cluster_2 = []
    orig_lable_cluster_3 = []
    orig_lable_cluster_4 = []
    orig_lable_cluster_5 = []
    for i in label_0:
        orig_lable_cluster_0.append(list(data_labels_before_kmeans.iloc[i])[0])
    for i in label_1:
        orig_lable_cluster_1.append(list(data_labels_before_kmeans.iloc[i])[0])
    for i in label_2:
        orig_lable_cluster_2.append(list(data_labels_before_kmeans.iloc[i])[0])
    for i in label_3:
        orig_lable_cluster_3.append(list(data_labels_before_kmeans.iloc[i])[0])
    for i in label_4:
        orig_lable_cluster_4.append(list(data_labels_before_kmeans.iloc[i])[0])
    for i in label_5:
        orig_lable_cluster_5.append(list(data_labels_before_kmeans.iloc[i])[0])
    
    frame_for_accuracy = data_labels_before_kmeans
    frame_for_accuracy['KMeans Labels Reassigned'] = 0
    
    for i in label_0:
        frame_for_accuracy['KMeans Labels Reassigned'].iloc[i] = most_frequent(orig_lable_cluster_0)
    for i in label_1:
        frame_for_accuracy['KMeans Labels Reassigned'].iloc[i] = most_frequent(orig_lable_cluster_1)
    for i in label_2:
        frame_for_accuracy['KMeans Labels Reassigned'].iloc[i] = most_frequent(orig_lable_cluster_2)
    for i in label_3:
        frame_for_accuracy['KMeans Labels Reassigned'].iloc[i] = most_frequent(orig_lable_cluster_3)
    for i in label_4:
        frame_for_accuracy['KMeans Labels Reassigned'].iloc[i] = most_frequent(orig_lable_cluster_4)
    for i in label_5:
        frame_for_accuracy['KMeans Labels Reassigned'].iloc[i] = most_frequent(orig_lable_cluster_5)
    
    return (frame_for_accuracy['KMeans Labels Reassigned'])


# # K-Fold Cross Validation and KNN

# In[27]:


def kfold_knn_accuracy(principalDf,original_bin_labels):
    accuracy_list = []
    principalDfData = principalDf
    kfold = KFold(n_splits=20,shuffle=False)
    for train_index, test_index in kfold.split(principalDfData):
        X_train,X_test= principalDfData.loc[train_index],principalDfData.loc[test_index]
        y_train,y_test = original_bin_labels.loc[train_index],original_bin_labels.loc[test_index]

        kmeans = KMeans(n_clusters= 6 , random_state=100).fit(X_train)
        kmeans_labels = pd.DataFrame(kmeans.labels_).rename(columns={0:'KMeans-Label'})
        reassigned_labels = label_reassignment(kmeans_labels,y_train)
        knn = KNeighborsClassifier(n_neighbors=35,metric='minkowski')
        knn.fit(X_train, reassigned_labels)
        y_predict = knn.predict(X_test)

        accuracy_list.append((accuracy_score(y_test, y_predict)))
    return (mean(accuracy_list))


# In[28]:


print('The average accuracy in (KMEANS) K-Fold KNN is',kfold_knn_accuracy(principalDf,original_bin_))


# In[29]:


X_train, y_train = principalDf,original_bin_
kmeans = KMeans(n_clusters= 6 , random_state=42).fit(X_train)
kmeans_labels = pd.DataFrame(kmeans.labels_).rename(columns={0:'KMeans-Label'})
reassigned_labels = label_reassignment(kmeans_labels,y_train)
knn = KNeighborsClassifier(n_neighbors=35,metric='minkowski')
knn.fit(X_train, reassigned_labels)


# # Pickling the Model

# In[30]:


from joblib import dump, load
dump(knn, 'KMeans_KNN.pickle')


# # DBSCAN

# In[ ]:





# In[31]:


def dbscan_kfold(principalDf,original_bin_):
    
    accuracy_list = []
    kfold = KFold(n_splits=10,shuffle=False)
    for train_index, test_index in kfold.split(principalDf):
        X_train,X_test= principalDf.loc[train_index],principalDf.loc[test_index]
        y_train,y_test = original_bin_.loc[train_index],original_bin_.loc[test_index]
        
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=1.45, min_samples=4)
        clusters = dbscan.fit_predict(X_train)
        dbscan_labels = dbscan.labels_
    
        noise_points = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == -1 ]
        cluster_0 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 0 ]
        cluster_1 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 1 ]
        cluster_2 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 2 ]
        cluster_3 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 3 ]
        
        X_train = X_train.reset_index().drop(columns=['index'])
        y_train = y_train.reset_index().drop(columns=['index'])
        
        merge_cluster = noise_points + cluster_0
        
        merged_cluster_data = X_train.iloc[merge_cluster].reset_index().drop(columns=['index'])
        merged_cluster_orig_bin = y_train.iloc[merge_cluster].reset_index().drop(columns=['index'])
        
        cluster1_data = X_train.iloc[cluster_1].reset_index().drop(columns=['index'])
        cluster1_dbscan_labels = pd.DataFrame(dbscan.labels_[cluster_1]).rename(columns={0:'DBSCAN-Label'})
        cluster1_orig_bin = y_train.iloc[cluster_1].reset_index().drop(columns=['index'])
        
        cluster2_data = X_train.iloc[cluster_2].reset_index().drop(columns=['index'])
        cluster2_dbscan_labels = pd.DataFrame(dbscan.labels_[cluster_2]).rename(columns={0:'DBSCAN-Label'})
        cluster2_orig_bin = y_train.iloc[cluster_2].reset_index().drop(columns=['index'])
        
        cluster3_data = X_train.iloc[cluster_3].reset_index().drop(columns=['index'])
        cluster3_dbscan_labels = pd.DataFrame(dbscan.labels_[cluster_3]).rename(columns={0:'DBSCAN-Label'})
        cluster3_orig_bin = y_train.iloc[cluster_3].reset_index().drop(columns=['index'])
        
        #KMeans on the merged noise points and cluster 0
        kmeans = KMeans(n_clusters= 4 , random_state=100).fit(merged_cluster_data)
        kmeans_labels_merged_cluster = pd.DataFrame(kmeans.labels_).rename(columns={0:'KMeans-Label'})
        
        reassigned_labels_merged_cluster = label_reassignment(kmeans_labels_merged_cluster,merged_cluster_orig_bin)
        reassigned_labels_cluster1 = label_reassignment(cluster1_dbscan_labels,cluster1_orig_bin)
        reassigned_labels_cluster2 = label_reassignment(cluster2_dbscan_labels,cluster2_orig_bin)
        reassigned_labels_cluster3 = label_reassignment(cluster3_dbscan_labels,cluster3_orig_bin)
        
        reassigned_labels_all = pd.concat([reassigned_labels_merged_cluster,reassigned_labels_cluster1,reassigned_labels_cluster2,reassigned_labels_cluster3]).reset_index().drop(columns=['index'])
        data_all = pd.concat([merged_cluster_data,cluster1_data,cluster2_data,cluster3_data]).reset_index().drop(columns=['index'])
        
        knn = KNeighborsClassifier(n_neighbors=35,metric='minkowski')
        knn.fit(data_all, reassigned_labels_all)
        y_predict = knn.predict(X_test)
        
    y_test = list(y_test['Original Label'])
    accuracy_list.append((accuracy_score(y_test, y_predict)))
    return(mean(accuracy_list))


# In[32]:


print('The average accuracy in (DBSCAN) K-Fold KNN is',dbscan_kfold(principalDf,original_bin_))


# In[33]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.26, min_samples=4)
clusters = dbscan.fit_predict(principalDf)
dbscan_labels = dbscan.labels_

noise_points = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == -1 ]
cluster_0 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 0 ]
cluster_1 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 1 ]
cluster_2 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 2 ]
cluster_3 = [ i for i in range(len(dbscan_labels)) if dbscan_labels[i] == 3 ]

merge_cluster = noise_points + cluster_0

merged_cluster_data = principalDf.iloc[merge_cluster].reset_index().drop(columns=['index'])
merged_cluster_orig_bin = original_bin_.iloc[merge_cluster].reset_index().drop(columns=['index'])

cluster1_data = principalDf.iloc[cluster_1].reset_index().drop(columns=['index'])
cluster1_dbscan_labels = pd.DataFrame(dbscan.labels_[cluster_1]).rename(columns={0:'DBSCAN-Label'})
cluster1_orig_bin = original_bin_.iloc[cluster_1].reset_index().drop(columns=['index'])

cluster2_data = principalDf.iloc[cluster_2].reset_index().drop(columns=['index'])
cluster2_dbscan_labels = pd.DataFrame(dbscan.labels_[cluster_2]).rename(columns={0:'DBSCAN-Label'})
cluster2_orig_bin = original_bin_.iloc[cluster_2].reset_index().drop(columns=['index'])

cluster3_data = X_train.iloc[cluster_3].reset_index().drop(columns=['index'])
cluster3_dbscan_labels = pd.DataFrame(dbscan.labels_[cluster_3]).rename(columns={0:'DBSCAN-Label'})
cluster3_orig_bin = y_train.iloc[cluster_3].reset_index().drop(columns=['index'])
        
#KMeans on the merged noise points and cluster 0
kmeans = KMeans(n_clusters= 4 , random_state=100).fit(merged_cluster_data)
kmeans_labels_merged_cluster = pd.DataFrame(kmeans.labels_).rename(columns={0:'KMeans-Label'})

reassigned_labels_merged_cluster = label_reassignment(kmeans_labels_merged_cluster,merged_cluster_orig_bin)
reassigned_labels_cluster1 = label_reassignment(cluster1_dbscan_labels,cluster1_orig_bin)
reassigned_labels_cluster2 = label_reassignment(cluster2_dbscan_labels,cluster2_orig_bin)
reassigned_labels_cluster3 = label_reassignment(cluster3_dbscan_labels,cluster3_orig_bin)

reassigned_labels_all = pd.concat([reassigned_labels_merged_cluster,reassigned_labels_cluster1,reassigned_labels_cluster2,reassigned_labels_cluster3]).reset_index().drop(columns=['index']).rename(columns={'KMeans Labels Reassigned':'DBSCAN Labels Reassigned'})
data_all = pd.concat([merged_cluster_data,cluster1_data,cluster2_data,cluster3_data]).reset_index().drop(columns=['index'])


# In[34]:


X_train,y_train = data_all,reassigned_labels_all
knn = KNeighborsClassifier(n_neighbors=35,metric='minkowski')
knn.fit(X_train, y_train)

