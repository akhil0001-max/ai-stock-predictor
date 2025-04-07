ai_stock_predictor.py

import yfinance as yf import pandas as pd import numpy as np import matplotlib.pyplot as plt import streamlit as st from sklearn.preprocessing import MinMaxScaler from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense, Dropout import datetime

st.set_page_config(page_title="AI Stock Predictor", layout="centered") st.title("AI-based Stock Predictor (Deep Learning)")

Step 1: User Input

symbol = st.text_input("Enter Company Symbol (e.g. INFY.BO for Infosys, TCS.BO, RELIANCE.BO)", "INFY.BO")

if st.button("Predict"): # Step 2: Get Historical Data end = datetime.datetime.today() start = end - datetime.timedelta(days=365 * 2) data = yf.download(symbol, start=start, end=end)

if data.empty:
    st.error("Invalid Symbol or No Data Found!")
else:
    # Step 3: Data Preprocessing
    df = data[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    sequence_len = 60
    x, y = [], []
    for i in range(sequence_len, len(scaled_data)):
        x.append(scaled_data[i-sequence_len:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Step 4: Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=3, batch_size=32, verbose=0)

    # Step 5: Prediction
    last_60_days = df[-60:].values
    last_60_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    # Step 6: Output Results
    st.subheader("Prediction")
    current_price = df.iloc[-1].values[0]
    predicted_price = pred_price[0][0]
    change = predicted_price - current_price
    confidence = 100 - abs(change) / current_price * 100

    if change > 0:
        st.success(f"UP: AI predicts stock may rise to ₹{predicted_price:.2f}")
    else:
        st.error(f"DOWN: AI predicts stock may fall to ₹{predicted_price:.2f}")

    st.info(f"Current Price: ₹{current_price:.2f}")
    st.warning(f"Confidence Score: {confidence:.2f}%")

    # Step 7: Show Chart
    df['Prediction'] = np.nan
    df.iloc[-1, df.columns.get_loc('Prediction')] = predicted_price

    st.line_chart(df[['Close', 'Prediction']])

