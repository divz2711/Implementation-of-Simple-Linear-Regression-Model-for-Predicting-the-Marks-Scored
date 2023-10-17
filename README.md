# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your cell. 
2. Type the required program.
3. Print the program.
4. End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divya Sampath
RegisterNumber:  212221040042
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:

##  df.head():
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/eca53c72-3a89-4a19-bad4-df0e7da37daa)

## df.tail():
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/fab91082-a0bb-4e28-a551-10bfa3956f69)

## Array value of X:
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/aadff37f-747b-4562-8bf2-7888a5c234e4)

## Array value of Y:
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/a6296764-907f-46e6-aa9b-cd022d763624)

## Values of Y prediction:
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/3d43fd76-f36e-4b47-a74c-93bb7158f803)

## Values of Y test:
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/86af87ab-b29e-4faa-947c-1ac6e64a47e8)

## Training Set Graph:
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/8f52ae45-00fc-4b08-99d7-8c5aaaccf4ca)

## Test Set Graph:
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/5898ecf7-7106-4327-8170-4e04f10db893)

## Values of MSE, MAE and RMSE:
![image](https://github.com/divz2711/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121245222/1dd2597c-b51c-48da-ae13-af9f805e035c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
