# Online-Payments-Fraud-Detection-with-Machine-Learning
The introduction of online payment systems has helped a lot in the ease of payments. But, at the same time, it increased in payment frauds. Online payment frauds can happen with anyone using any payment system, especially while making payments using a credit card. That is why detecting online payment fraud is very important for credit card companies to ensure that the customers are not getting charged for the products and services they never paid.

## Online Payments Fraud Detection with Machine Learning
To identify online payment fraud with machine learning, we need to train a machine learning model for classifying fraudulent and non-fraudulent payments. For this, we need a dataset containing information about online payment fraud, so that we can understand what type of transactions lead to fraud. For this task, I collected a dataset from Kaggle, which contains historical information about fraudulent transactions which can be used to detect fraud in online payments. Below are all the columns from the dataset I’m using here:

step : represents a unit of time where 1 step equals 1 hour type : type of online transaction amount : the amount of the transaction nameOrig : customer starting the transaction oldbalanceOrg : balance before the transaction newbalanceOrig : balance after the transaction nameDest : recipient of the transaction oldbalanceDest : initial balance of recipient before the transaction newbalanceDest : the new balance of recipient after the transaction isFraud : fraud transaction Online Payments Fraud Detection using Python importing the necessary Python libraries and the dataset we need for this task:

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
Read and show Dataset

data = pd.read_csv( '/home/ryzenrtx/Prince/Projects/Online Payments Fraud Detection with Machine Learning/PS_20174392719_1491204439457_log.csv') data.head() data

Now, let’s have a look at whether this dataset has any null values or not:

print(data.isnull().sum())

null

So this dataset does not have any null values. Before moving forward, now, let’s have a look at the type of transaction mentioned in the dataset:

explore

type = data["type"].value_counts() transactions = type.index quantity = type.values

import plotly.express as px figure = px.pie(data, values=quantity, names=transactions,hole = 0.5, title="Distribution of Transaction Type") figure.show() graph

Now let’s have a look at the correlation between the features of the data with the isFraud column:

Checking correlation
correlation = data.corr() sns.heatmap(correlation, annot=True)

corr

Now let’s transform the categorical features into numerical. Here I will also transform the values of the isFraud column into No Fraud and Fraud labels to have a better understanding of the output:

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}) data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

Online Payments Fraud Detection Model Now let’s train a classification model to classify fraud and non-fraud transactions. Before training the model, I will split the data into training and test sets:

splitting the data
from sklearn.model_selection import train_test_split x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]) y = np.array(data[["isFraud"]])

Now let’s train the online payments fraud detection model:

training a machine learning model
from sklearn.tree import DecisionTreeClassifier xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size=0.10, random_state=42) model = DecisionTreeClassifier() model.fit(xtrain, ytrain) print(model.score(xtest, ytest)) Now let’s classify whether a transaction is a fraud or not by feeding about a transaction into the model:

prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig] features = np.array([[4, 9000.60, 9000.60, 0.0]]) print(model.predict(features))

Dump Model with the help of pickle

import pickle pickle.dump(mode|l, open("model.pkl", "wb"))

loading the model

model = pickle.load(open("model.pkl", "rb"))
