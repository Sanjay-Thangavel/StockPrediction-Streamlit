import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta



# Fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data["Close"]

# Prepare data for LSTM
def prepare_data(series, time_steps):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Load or train LSTM model
# Load or train LSTM model
def load_or_train_model(stock_ticker, start_date, end_date, time_steps=60, epochs=20, batch_size=32):
    model_filename = f"{stock_ticker}_lstm_model.h5"
    scaler_filename = f"{stock_ticker}_scaler.pkl"
    
    try:
        # Try to load the model and scaler if they exist
        model = load_model(model_filename)
        print(f"Loaded pre-trained model: {model_filename}")
        
        # Load the scaler from file
        import pickle
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded pre-trained scaler: {scaler_filename}")
    except:
        print("Model or scaler not found, training a new model...")
        # Fetch stock data
        stock_prices = fetch_stock_data(stock_ticker, start_date, end_date)
        stock_prices = stock_prices.values.reshape(-1, 1)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(stock_prices)
        
        # Prepare data for training
        X, y = prepare_data(scaled_prices, time_steps)
        X_train, y_train = X[:-int(len(X) * 0.2)], y[:-int(len(y) * 0.2)]  # 80% training data
        X_test, y_test = X[-int(len(X) * 0.2):], y[-int(len(y) * 0.2):]    # 20% testing data
        
        # Build LSTM model
        model = build_lstm_model(X_train.shape[1:])
        
        # Train model
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the trained model
        model.save(model_filename)
        print(f"Model trained and saved as {model_filename}")
        
        # Save the scaler to file using pickle
        import pickle
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved as {scaler_filename}")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()
    
    return model, scaler


# Predict stock price for next 7 days
# Predict stock price for next 7 days
def predict_next_7_days(model, scaler, stock_ticker, start_date, time_steps=60):
    # Fetch stock data
    stock_data = yf.download(stock_ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
    closing_prices = stock_data["Close"].values
    scaled_data = (closing_prices - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices))
    
    # Prepare data for prediction
    input_data = scaled_data[-time_steps:]
    input_data = np.expand_dims(input_data, axis=0)  # Reshape to (1, time_steps)

    # Predict the next 7 days
    predictions = []
    for _ in range(7):
        prediction = model.predict(input_data)
        predictions.append(prediction[0, 0])
        
        # Reshape prediction to match input_data's dimensions
        prediction_reshaped = np.reshape(prediction, (1, 1, 1))  # Shape (1, 1, 1)
        
        # Update input_data with the predicted value
        input_data = np.append(input_data[:, 1:, :], prediction_reshaped, axis=1)

    # Rescale predictions to original scale
    predictions = np.array(predictions)
    predictions = predictions * (np.max(closing_prices) - np.min(closing_prices)) + np.min(closing_prices)
    
    return predictions


# Streamlit App
st.title("Stock Price Prediction using LSTM")
st.markdown("""
This app predicts stock prices for the next 7 days using an LSTM model.  
Provide a stock ticker and a start date, and we will predict the stock price for the next 7 days.
""")

# Input for stock ticker and date
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="AAPL")
start_date = st.date_input("Enter Start Date:", value=datetime.today() - timedelta(days=365))

# if st.button("Predict"):
#     # Load or train the LSTM model
#     model, scaler = load_or_train_model(stock_ticker, start_date, datetime.today().strftime('%Y-%m-%d'))
    
#     # Predict the next 7 days stock prices
#     predictions = predict_next_7_days(model, scaler, stock_ticker, start_date)
    
#     # Display results
#     future_dates = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
#     prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})

#     st.write("### Predicted Stock Prices for the Next 7 Days")
#     st.write(prediction_df)

#     # Plotting predictions
#     stock_data = yf.download(stock_ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
#     closing_prices = stock_data["Close"].values
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(stock_data.index[-100:], closing_prices[-100:], label="Historical Prices")
#     plt.plot(future_dates, predictions, marker="o", label="Predicted Prices")
#     plt.xlabel("Date")
#     plt.ylabel("Stock Price")
#     plt.legend()
#     plt.grid()
#     st.pyplot(plt)


if st.button("Predict"):
    # Load or train the LSTM model
    model, scaler = load_or_train_model(stock_ticker, start_date, datetime.today().strftime('%Y-%m-%d'))
    
    # Predict the next 7 days stock prices
    predictions = predict_next_7_days(model, scaler, stock_ticker, start_date)
    
    # Display results
    future_dates = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})

    st.write("### Predicted Stock Prices for the Next 7 Days")
    st.write(prediction_df)

    # Plotting predictions
    stock_data = yf.download(stock_ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
    closing_prices = stock_data["Close"].values
    
    # Convert numpy.datetime64 to Python datetime for matplotlib compatibility
    future_dates = pd.to_datetime(future_dates)

    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index[-100:], closing_prices[-100:], label="Historical Prices")
    plt.plot(future_dates, predictions, marker="o", label="Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)