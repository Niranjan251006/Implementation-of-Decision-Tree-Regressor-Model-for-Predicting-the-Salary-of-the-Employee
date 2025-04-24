# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NIRANJAN S
RegisterNumber:  212224040221
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
r2=metrics.r2_score(y_test,y_pred)
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

![319283190-964c85c1-4627-45c1-b905-0f77c3c3b12d](https://github.com/user-attachments/assets/35f09b1f-65e3-4f0c-9780-8f00f854d6f5)

![319284497-b9d39342-e915-4ebd-906d-c7910f9bb566](https://github.com/user-attachments/assets/4d1ce0b4-856e-498c-8ef6-eab60ccd1c40)

![319283523-ddede0b2-c6d0-4630-bd54-16dac56a3b01](https://github.com/user-attachments/assets/0080d0b5-9d13-4910-af4d-d2e1dcd07b75)

![319284673-5cb9db1a-7819-42c7-be62-d137d6209c8c](https://github.com/user-attachments/assets/9da4577e-a3ae-42b0-af94-ea350e848dbf)

![319284919-acbff202-598d-4d01-a02f-7a7fb45741a0](https://github.com/user-attachments/assets/1969b5b0-8c27-4515-8b68-e0194cc7faa4)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
