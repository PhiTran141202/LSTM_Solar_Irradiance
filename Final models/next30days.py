import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# READ THE DATASHEET (make panda recognize there's a date and not treat as normal string)
df = pd.read_csv( "C:/Users/Phipr/Desktop/Past Research/Completed Projects/LSTM_Solar_Prediction/Hanoi_daily.csv",  index_col='datetime', parse_dates=True)

# Print the first 5 values from list
print(df.head())

#plot the datasheet
#df.plot(figsize=(12,6))

#Time Series Decomposition, Seasonal and Trend Component Decomposition using Statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
#feed the data
results = seasonal_decompose(df['solarradiation'])
results.plot()
#plt.show()


data = pd.DataFrame(list(df['solarradiation']), columns=['solarradiation'])
print(data)



# specify number of data to use, 365 days
df = data[:365]

#print the shape of sheet
print(df.shape)

#print the number of null values in the list so we can subtract those
print(df.isnull().sum())
#dropping the null values from the data frame if theres any
df=df.dropna(axis=0)

#print the new sheet shape w/o null values
print(df.shape)


#get all the values into an array
df=df['solarradiation'].values

print(df[:5])

#turn the value array vertically
df=df.reshape(-1,1)


#PLOTTING THE DATA FRAME
#plt.figure(figsize=(25, 7))
#plt.plot(df, linewidth=1)
#plt.grid()
#plt.title("Time Series (daily radiation for 1 year)")
#plt.show()


#PREPROCESSING THE DATA, use minmaxscaler to convert the data into scale from 0 to 1 (install scikit-learn package)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#fitting then transforming the data
data_scaled = scaler.fit_transform(df)

#print the first 5 scaled values
print(data_scaled[:5])
#print the shape of the new scaled set
print(data_scaled.shape)


#MAKING TWO IN/OUT ARRAYS, putting the scaled values in each arrays, for 20 inputs we get 1 prediction
steps= 20
inp = []
out = []
for i in range(len(data_scaled) - (steps)):
    inp.append(data_scaled[i:i+steps])
    out.append(data_scaled[i+steps])

print(out[:10])

import numpy as np

#reshaping the input and output lists into numpy arrays
inp= np.asanyarray(inp)
out= np.asanyarray(out)

#taking 65% of the data from dataframe
print(len(df)*0.65)

#split the remaining data into 2 sets, training and testing
x_train = inp[:237]
x_test = inp[237:]
y_train = out[:237]
y_test= out[237:]

print(inp.shape)

print(x_train.shape)

print(x_test.shape)

# FIX RANDOM SEED FOR REPRODUCIBILITY
import numpy.random
import tensorflow as tf
import random
randomnum = 343
random.seed(randomnum)
#numpy.random.seed(randomnum)
tf.random.set_seed(randomnum)

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense,RepeatVector, LSTM, Dropout
from tensorflow.keras.models import Sequential


#MODEL BUILDING
neuronnum=95
# expected input data shape: (batch_size, time steps, data_dim)
model = Sequential() #make sure that the layers are in sequence, one after the other
model.add(LSTM(neuronnum, return_sequences= True, input_shape= (20,1))) #adding LSTM layer, 100 neurons, activation function relu, clarify the input shape of our data
model.add(LSTM(neuronnum, return_sequences=True)) # returns a sequence of vectors of dimension 100
model.add(LSTM(neuronnum)) # return a single vector of dimension 100
model.add(Dense(1)) #final output layer, make the prediction, dense = 1 means the layer generates 1 value
model.compile(loss = 'mean_squared_error', optimizer = 'adam') #compile using adam optimizer and mean square error as loss function
model = Sequential()
model.add(LSTM(neuronnum, return_sequences= True, input_shape= (20,1)))
model.add(LSTM(neuronnum, return_sequences=True))
model.add(LSTM(neuronnum))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

#print model summary
print(model.summary())

#training the model, call the input and output arrays of the training set, verbose=0 : Displays no logs. verbose=1 : Displays progress bar with logs (default). verbose=2 : Displays one line per epoch.
model.fit(x_train,y_train,epochs=100, verbose=1, )

#
#loss_per_epoch = model.history.history['loss']
#plt.plot(range(len(loss_per_epoch)), loss_per_epoch)

# model.evaluate(x_test, y_test)

#PREDICTION AND MODEL EVALUATION

#predictions on training set
print("Predicted Value",model.predict(x_train)[4][0])
print("Expected value",y_train[4][0])

#PREDICTING ON TESTING SET
predicted_values=model.predict(x_test)
expected_values=y_test

print("Predicted Value",predicted_values[2][0])
print("Expected Value",expected_values[2][0])

print('prediction shape: ', predicted_values.shape)
print('true values (y_test) shape: ', expected_values.shape)


#Making Numpy Dataframes to plot
pred_df=pd.DataFrame(predicted_values)
pred_df['TrueValues']=expected_values
pred_df_new = pred_df.rename(columns={ 0: 'Predictions'})
print('new prediction dataframe: \n', pred_df_new)

#Converting the data to original values
true_predictions = scaler.inverse_transform(pred_df)
print(true_predictions)

predict=pd.DataFrame(true_predictions)
print(predict)
predict = predict.rename(columns={ 1: 'TrueValues'})
predict = predict.rename(columns={ 0: 'Predictions'})
print(predict)


#Plotting the figures
plt.figure(figsize=(12,8))
sns.lineplot(data= predict)
plt.title("Predictions VS True Values on Testing Set")


#Calculate the root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(predicted_values, expected_values))
print("root mean square error: ", rmse)

#plt.show()

#PREDICTION FOR THE NEXT 30 DAYS, using last 10 days input for 1st day output

print(data_scaled.shape)
x_input=data_scaled[:20]
print(x_input.shape) #shape before reshaping
x_input = x_input.reshape(1, -1) #reshaping the data
print(x_input.shape) #shape after reshaping

radiation_input = list(x_input)
radiation_input = radiation_input[0].tolist()

# demonstrate prediction for next 10 days
from numpy import array

lst_output = []
n_steps = 20
i = 0
while (i < 30):

    if (len(radiation_input) > 20):
        # print(temp_input)
        x_input = np.array(radiation_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        radiation_input.extend(yhat[0].tolist())
        radiation_input = radiation_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        radiation_input.extend(yhat[0].tolist())
        #         print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1

print(lst_output)

true_output = scaler.inverse_transform(lst_output)

print(true_output)


#PLOTTING NEXT 10 DAYS PREDICTIONS
day_new = np.arange(1, 366)
day_pred = np.arange(365, 395)

print(day_new.shape)
print(data_scaled.shape)
print(day_pred.shape)
#print(lst_output.shape)

#Un-scaled the data
true_data = scaler.inverse_transform(data_scaled)


plt.figure(figsize=(12, 8))
plt.plot( day_new, true_data)
plt.plot(day_pred, true_output)


df3 = true_data.tolist()
df3.extend(true_output)
plt.figure(figsize=(12, 8))
plt.plot(df3)

plt.show()