#libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pickle

#reading dataset 
crop_df = pd.read_csv("D:/CODING/Projects/Crop_prediction_using_AI/data/Crop_recommendation.csv")

#data insights
print(crop_df.head())
print(crop_df.shape)
print(crop_df.isnull().sum())
print(crop_df.describe())

#visualizations

#nitrogen
plt.figure(figsize=(8,7))
sns.histplot(x='N', data=crop_df, color = 'b')
plt.title("Nitogen for crops", {'fontsize': 20})
plt.show()

#phosporus
plt.figure(figsize=(8,7))
sns.histplot(x='P', data=crop_df, color = 'b')
plt.title("Phosporus for crops", {'fontsize': 20})
plt.show()

#Potassium 
plt.figure(figsize=(8,7))
sns.histplot(x='K', data=crop_df, color = 'b')
plt.title("Potassium for crops", {'fontsize': 20})
plt.show()

#Temperature
plt.figure(figsize=(8,7))
sns.boxplot(x= crop_df.temperature)
plt.show()

#humidity
plt.figure(figsize=(8,7))
sns.boxplot(x= crop_df.humidity)
plt.show()

#PH
plt.figure(figsize=(8,7))
sns.histplot(x='ph', data=crop_df, color = 'b')
plt.title("PH for crops", {'fontsize': 20})
plt.show()

#Rainfall
plt.figure(figsize=(8,7))
sns.histplot(x='rainfall', data=crop_df, color = 'b')
plt.title("Rainfall for crops", {'fontsize': 20})
plt.show()

sns.displot(crop_df['ph'])
plt.show()

sns.scatterplot(crop_df['temperature'])
plt.title("Temperature")
plt.show()

plt.figure(figsize=(8,7))
g = sns.FacetGrid(crop_df, hue="label", height=5)
g.map(plt.scatter, "N", "humidity").add_legend()
plt.title("N vs Humidity by Label")
plt.show()



plt.figure(figsize=(10,8))
numeric_columns = crop_df.select_dtypes(include=['number']).columns
corrmat = crop_df[numeric_columns].corr()
corrmat.style.background_gradient('coolwarm')
plt.show()

sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(crop_df, hue="label", height=3)
plt.show()


#splitting data for training 

X= crop_df.drop(['label'], axis=1)
Y= pd.Categorical(crop_df.label)

X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size=0.2)

#K Nearest Neighbors Model
knnclassifier = KNeighborsClassifier(n_neighbors=9)
knnclassifier.fit(X_train, Y_train)
print("The accuracy of KNN classification is ", knnclassifier.score(X_train, Y_train), knnclassifier.score(X_test, Y_test))
knn= [knnclassifier.score(X_train, Y_train),knnclassifier.score(X_test, Y_test)]

#SVM Model
svm= SVC()
svm.fit(X_train, Y_train)
print("The accuracy of SVM is", svm.score(X_train, Y_train),svm.score(X_test, Y_test))
svm= [svm.score(X_train, Y_train),svm.score(X_test, Y_test)]

#Decision Tree Model
dtclassifier= DecisionTreeClassifier(max_depth=7)
dtclassifier.fit(X_train, Y_train)
print("The accuracy of decision tree is", dtclassifier.score(X_train, Y_train),dtclassifier.score(X_test, Y_test))
dt= [dtclassifier.score(X_train, Y_train),dtclassifier.score(X_test, Y_test)]

#Random forest model
rfclassifier = RandomForestClassifier()
rfclassifier.fit(X_train, Y_train)
print("The accuracy of random forest classifier is", rfclassifier.score(X_train, Y_train), rfclassifier.score(X_test, Y_test))
rf= [rfclassifier.score(X_train, Y_train), rfclassifier.score(X_test, Y_test)]

#evaluation metrics
#knn
knnclassifier= KNeighborsClassifier()
knnclassifier.fit(X_train, Y_train)
y_pred = knnclassifier.predict(X_test)
print(classification_report(Y_test, y_pred))

#dt
dtclassifier= DecisionTreeClassifier()
dtclassifier.fit(X_train, Y_train)
y_pred = dtclassifier.predict(X_test)
print(classification_report(Y_test, y_pred))

#rf
rfclassifier= RandomForestClassifier()
rfclassifier.fit(X_train, Y_train)
y_pred = rfclassifier.predict(X_test)
print(classification_report(Y_test, y_pred))

#From the table we can see that KNN and Random Forest models perform the best.
# After training the knn model we check the accuracies. Our model has achieved 97.7 % accuracy for the test data.
#After checking the performance, we decide to save the knn model.

#storing model in model.pkl
pickle.dump(knnclassifier, open('model.pkl', 'wb'))