# KMeans-and-DBSCAN-Clustering-on-Carbohydrate-Amount-in-CGM-Timeseries-Data-and-Validation-using-KNN

#### Given: 	Meal Data of 5 subjects
#### Amount of carbohydrates in each meal
#### To do: 
##### a)	Extract features from Meal data
#### b)	Cluster Meal data based on the amount of carbohydrates in each meal
##### First consider the given Meal data. Take the first 50 rows of the meal data. Each row is the meal amount of the corresponding row in the mealDataX.csv of every subject. So mealAmountData1.csv corresponds to the first subject. The first 50 rows of the mealAmountData1.csv corresponds to the first 50 rows of mealDataX.csv.

##### Extracting Ground Truth: Consider meal amount to range from 0 to 100. Discretize the meal amount in bins of size 20. Consider each row in the mealDataX.csv and according to their meal amount label put them in the respective bins. There will be 6 bins starting from 0, >0 to 20, 21 to 40, 41 to 60, 61 to 80, 81 to 100. 

#### Now ignore the mealAmountData. Without using the meal amount data use the features in your assignment 2 to cluster the mealDataX.csv into 6 clusters. Use DBSCAN or KMeans. Try these two. 

#### Report your accuracy of clustering based on SSE and supervised cluster validity metrics.

#### Test script that does KNN classification choose K, choose distance metric

#### Given a test data, calculate distances of the test data from each of your training data point.Then do a K majority-based classification on  DBSCAN and K means

