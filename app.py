import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from neuralprophet import NeuralProphet
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import os
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.express as px

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Suppress TensorFlow info and warning messages
tf.get_logger().setLevel('ERROR')

# Customize title font color
st.markdown("<h1 style='text-align: center; color: red;'>Stock Trend Predictor</h1>", unsafe_allow_html=True)

# Add the developer information
st.markdown("<p style='text-align: center;  color: skyblue; '>Developed by SUBHRADEEP NATH</p>", unsafe_allow_html=True)



user_input = st.text_input('Enter Stock Ticker', 'AAPL')


# Fetch historical data for the specified stock
stock_data = yf.download(user_input, start='2015-01-01', end='2024-04-10')
# Reset index to turn 'Date' into a regular column
stock_data.reset_index(inplace=True)

# Drop the 'Date' column and the 'Adj Close' column
modified_stock_data = stock_data.drop(columns=['Adj Close'])

# Print the modified DataFrame
print(modified_stock_data.head())


#Describing Data
st.subheader('Stock Analysis Data from 2015 - 2024' )
st.dataframe(modified_stock_data)

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(stock_data.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = stock_data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = stock_data.Close.rolling(100).mean()
ma200 = stock_data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(stock_data.Close, 'b')
st.pyplot(fig)

#Splitting Data into Training and Testing
data_training = pd.DataFrame(stock_data['Close'][0:int(len(stock_data)*0.70)])
data_testing = pd.DataFrame(stock_data['Close'][int(len(stock_data)*0.70): int(len(stock_data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#Load my model
model = tf.keras.models.load_model('keras_model.h5')


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

if model.compiled_metrics is not None:
    st.write("Model has been compiled with metrics")
else:
    st.write("Model has not been compiled with metrics")

#Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
# Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original Chart')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'g', label= 'Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


#stocks= stock_data
#stocks['Date']= pd.to_datetime(stocks['Date'])
#stocks= stocks[['Date','Close']]
#stocks.columns = ['ds', 'y']

#model = NeuralProphet()
#model.fit(stocks)

#future = model.make_future_dataframe(stocks, periods = 300)
#forecast = model.predict(future)
#actual_prediction = model.predict(stocks)

#st.subheader('Future Trend Forecast')
#fig3 = plt.figure(figsize=(12,6))
#plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label = 'actual_prediction', c = 'r')
#plt.plot(forecast['ds'], forecast['yhat1'], label = 'future_prediction', c = 'b')
#plt.plot(stocks['ds'], stocks['y'], label = 'original', c = 'g')
#plt.legend()
#plt.pyplot(fig3)


# Assuming y_test and y_predicted are NumPy arrays or pandas Series

# Flatten y_test and y_predicted arrays if needed
y_test_flat = y_test.flatten()
y_predicted_flat = y_predicted.flatten()

# Create DataFrame
df = pd.DataFrame({'Time': range(len(y_test_flat)), 'Original Price': y_test_flat, 'Predicted Price': y_predicted_flat})

# Create figure
fig = go.Figure()

# Add original price
fig.add_trace(go.Scatter(x=df['Time'], y=df['Original Price'], mode='lines', name='Original Price', line=dict(color='green')))

# Add predicted price (initially empty)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Predicted Price', line=dict(color='red')))

# Update layout
fig.update_layout(title='Predictions vs Original Stock Trend', xaxis_title='Time', yaxis_title='Price', showlegend=True)

# Define frames for animation
frames = [go.Frame(data=[go.Scatter(x=df['Time'][:i+1], y=df['Original Price'][:i+1]),
                         go.Scatter(x=df['Time'][:i+1], y=df['Predicted Price'][:i+1])],
                   name=str(i)) for i in range(len(df))]

# Add frames to figure
fig.frames = frames

# Set frame duration to update faster
frame_duration = 15  # Adjust this value to change the speed of the animation
fig.layout.updatemenus = [dict(type='buttons', showactive=False, buttons=[dict(label='Predict',
                                                                              method='animate',
                                                                              args=[None, dict(frame=dict(duration=frame_duration, redraw=True), fromcurrent=True)])])]

# Display the animated plot
st.plotly_chart(fig)


