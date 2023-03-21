# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

We create a simple dataset with one input and one output. This data is then divided into testing and training sets for our Neural Network Model to train and test on. The NN Model contains input layer, 3 hidden layer with 12, 6, 3 nodes/neurons in it, which is then connected to the final output layer with one node/neuron. The Model is then compiled with an loss function and Optimizer, here we use MSE and rmsprop. The model is then trained for 150 epochs.
We then perform an evaluation of the model with the test data. An user input is then predicted with the model. Finally, we plot the Training Loss VS Iteration graph for the given model.

## NEURAL NETWORK MODEL

![image](https://user-images.githubusercontent.com/75234588/226642673-16ab9a32-99b0-4f82-9eee-dd27b74bd784.png)

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
#### Developed By: J Vincent isaac jeyaraj
#### Register No: 212220230060

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
    Dense(12,activation='relu'),
    Dense(6),
    Dense(3),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train_scaled,y=Y_train,epochs=110)

loss_df=pd.DataFrame(model.history.history)
loss_df.plot()

x_Test1= Scaler.transform(x_test)
model.evaluate(x_test,y_test)

x_n1=[[50]]
x_n1_1=Scaler.transform(x_n1)

model.predict(x_n1)
```
## DATASET INFORMATION

![dl1](https://user-images.githubusercontent.com/75234588/226409462-d19150da-4266-44e7-9e85-4dbe2226af3b.PNG)

## OUTPUT

### Training Loss Vs Iteration Plot

![dl2](https://user-images.githubusercontent.com/75234588/226635294-16dfd685-749b-480a-8fb1-060ca34db86b.PNG)

### Test Data Root Mean Squared Error

![dl3](https://user-images.githubusercontent.com/75234588/226635335-e549dc9f-6b6b-457f-acd6-44e90dffab2c.PNG)

### New Sample Data Prediction

![dl4](https://user-images.githubusercontent.com/75234588/226635385-95fcf7b0-e0a4-4467-a611-b0deff0f1a41.PNG)


## RESULT
Thus, a Simple Neural Network Regression Model is developed successfully.
