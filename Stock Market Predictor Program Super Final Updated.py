#Importing Libraries For Computation 
import pandas as pd
import numpy as np
#Importing Stock Market Library
import yfinance as yf
#Importing Libraries For Machine Learning
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#Import matplot for plotting graphs of the stocks
import matplotlib.pyplot as plt
%matplotlib inline

#Assigning stock market and dates as inputs
Stock_Name = input("Enter a Stock Market:")
Start_Date = input("Enter the Start-Date(Y-M-D):")
End_Date = input("Enter the Final-Date(Y-M-D):") 

#Downloads the Finance/Stock Information
Stock_Table = yf.download(Stock_Name,Start_Date,End_Date)

#Prints the First Row
Stock_Table.head()

#Plots the Stock Closings to Time
plt.figure(figsize=(20,12))
plt.plot(Stock_Table['Close'], label='Close Price history')

#Creating DataFrame For Closing Prices
Closing_Table = pd.DataFrame(index=range(0,len(Stock_Table)),columns=["Date", "Close"])
Closing_Table['Date'][:] = Stock_Table.index[:]
Closing_Table['Close'][:] = Stock_Table['Close'][:]

Closing_Table.index = Closing_Table.Date
Closing_Table=Closing_Table.drop(["Date"],axis=1)

#Creation of Data for Training and Prediction Accuracy
Datas = Closing_Table.values

Training_Set = np.array(Datas[0:1510,:])
Testing_Set = np.array(Datas[1510:,:])
print(Datas.size)

#Normalizing Data For Accurate Computations
from sklearn.preprocessing import MinMaxScaler
Normalize = MinMaxScaler(feature_range=(0, 1))
Normalized_Data = Normalize.fit_transform(Datas)

#Creating X_Train and Y_Train
Prediction_Days=60
X_Train = []
Y_Train = []
for count in range(Prediction_Days,len(Training_Set)):
    X_Train.append(Normalized_Data[count-Prediction_Days:count,0])
    Y_Train.append(Normalized_Data[count,0])
X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)

#Reshape X_Train to Be Fit in Algorithm
X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1],1))
print(X_Train.shape)

model = Sequential()

#Layer 1
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_Train.shape[1],1)))

#Layer 2
model.add(LSTM(units=50))

#Output Layer
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_Train, Y_Train, epochs=2, batch_size=2, verbose=2)

inputs = Closing_Table[len(Closing_Table) - len(Testing_Set)-Prediction_Days:].values
inputs = inputs.reshape(-1,1)
inputs  = Normalize.transform(inputs)

X_Test = []
for T in range(Prediction_Days,inputs.shape[0]):
    X_Test.append(inputs[T-Prediction_Days:T,0])
X_Test = np.array(X_Test)

X_Test = np.reshape(X_Test, (X_Test.shape[0],X_Test.shape[1],1))
Closing_Price = model.predict(X_Test)
Closing_Price = Normalize.inverse_transform(Closing_Price)

#Plot the Tested Set Against the Real Data
Training_Set = Closing_Table[:1510]
Testing_Set = Closing_Table[1510:]
Testing_Set['Predictions'] = Closing_Price
print(Training_Set)
print(Testing_Set)
plt.plot(Training_Set['Close'])
plt.plot(Testing_Set[['Close','Predictions']])

#Checking Accuracy of Program
Subtract=Testing_Set['Close']-Testing_Set['Predictions']
for i in range(len(Subtract)):
    if Subtract[i] < 0:
        Subtract[i]=Subtract[i]*-1
Accuracy=(Subtract/Testing_Set['Close'])*100
print("Accuracy Percentage Error:")
print(round(Accuracy.mean(),5),"%")
