# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

We create a simple dataset with one input and one output. This data is then divided into testing and training sets for our Neural Network Model to train and test on. The NN Model contains input layer, 2 nodes/neurons in the hidden layer which is then connected to the final output layer with one node/neuron. The Model is then compiled with an loss function and Optimizer, here we use MSE and rmsprop. The model is then trained for 2000 epochs.
We then perform an evaluation of the model with the test data. An user input is then predicted with the model. Finally, we plot the Training Loss VS Iteration graph for the given model.

## NEURAL NETWORK MODEL

![nn](https://user-images.githubusercontent.com/75234991/188797088-90a2a2ff-a38d-431f-9cce-f2f76358819b.svg)

## DESIGN STEPS

### Step 1:

Load the dataset.

### Step 2:

Split the dataset into training and testing data.

### Step 3:

Create MinMaxScalar object, fit the model and transform the data.

### Step 4:

Build the Neural Network Model and compile the model.

### Step 5:

Train the model with the training data.

### Step 6:

Plot the performance plot.

### Step 7:

Evaluate the model with the testing data.

## PROGRAM
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('data').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])

df.head()

df=df.astype({'X':'float'})
df=df.astype({'Y':'float'})
df.dtypes

X=df[['X']].values
Y=df[['Y']].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=33)

scaler=MinMaxScaler()
scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)

model=Sequential([
    Dense(2,activation='relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train_scaled,y=Y_train,epochs=20000)

loss_df=pd.DataFrame(model.history.history)
loss_df.plot()

x_Test1= Scaler.transform(x_test)
model.evaluate(x_test,y_test)

x_n1=[[4]]
x_n1_1=Scaler.transform(x_n1)

model.predict(x_n1_1)
```
## DATASET INFORMATION

![dl1](https://user-images.githubusercontent.com/75234588/226409462-d19150da-4266-44e7-9e85-4dbe2226af3b.PNG)

## OUTPUT

### Training Loss Vs Iteration Plot

![dl2](https://user-images.githubusercontent.com/75234588/226409486-82bcc472-a4ff-40d1-aea4-e81447a88b03.PNG)

### Test Data Root Mean Squared Error

![dl3](https://user-images.githubusercontent.com/75234588/226409540-3e6c0a96-68ec-4810-9b79-954820d232a9.PNG)

### New Sample Data Prediction

![dl4](https://user-images.githubusercontent.com/75234588/226409589-0047e69c-87f2-478c-ab01-574c1fb47f6d.PNG)


## RESULT
Thus, a Simple Neural Network Regression Model is developed successfully.
