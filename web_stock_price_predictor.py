import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 📌 Streamlit UI Setup
st.set_page_config(page_title="Global Stock Price Predictor", layout="wide")
st.title("🌍 Global Stock Price Predictor with 10-Year Forecast")

# Dummy Company-to-Ticker Mapping
company_db = {
    "Apple Inc": "AAPL",
    "Tesla Inc": "TSLA",
    "Alphabet Inc (Google)": "GOOG",
    "Microsoft Corp": "MSFT",
    "Reliance Industries Ltd": "RELIANCE.NS",
    "Infosys Ltd": "INFY.NS",
    "Toyota Motor Corp": "7203.T",
    "HSBC Holdings": "HSBA.L",
    "BP Plc": "BP.L",
    "BMW AG": "BMW.DE",
}

# 🔍 Company Selection
company_name = st.selectbox("Select a Company", options=list(company_db.keys()))
stock = company_db[company_name]
st.success(f"✅ Selected: {company_name} → {stock}")

# ⏳ Date Range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# 📥 Download 20 Years of Stock Data
data = yf.download(stock, start=start, end=end)
if data.empty:
    st.error("No data found for this ticker. Try another.")
    st.stop()

st.subheader("📊 Full Historical Stock Data (20 Years)")
st.write(data)
st.caption(f"Records: {len(data)} | From: {data.index.min().date()} → {data.index.max().date()}")

# 📈 Plot with Moving Averages
data['MA_100'] = data['Close'].rolling(100).mean()
data['MA_200'] = data['Close'].rolling(200).mean()

def plot_ma():
    fig = plt.figure(figsize=(15,6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['MA_100'], label='100-day MA')
    plt.plot(data['MA_200'], label='200-day MA')
    plt.title(f"{company_name} Stock Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    return fig

st.subheader("📉 Stock Price Chart with MAs")
st.pyplot(plot_ma())

# 🧠 Load Model
model = load_model("Latest_stock_price_model.keras")

# 🧪 Prepare Data
split_len = int(len(data) * 0.7)
test_data = data['Close'][split_len:]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(np.array(test_data).reshape(-1,1))

x_test, y_test = [], []
for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i-100:i])
    y_test.append(scaled_data[i])

x_test = np.array(x_test)
y_test = np.array(y_test)

# 🧪 Predict on Past Test Set
predicted = model.predict(x_test)
predicted_inv = scaler.inverse_transform(predicted)
actual_inv = scaler.inverse_transform(y_test.reshape(-1,1))

compare_df = pd.DataFrame({
    "Actual Price": actual_inv.flatten(),
    "Predicted Price": predicted_inv.flatten()
}, index=data.index[split_len + 100:])

compare_df = compare_df.round(2)

st.subheader("🧾 Actual vs Predicted Prices (Full Test Set)")
st.dataframe(compare_df, height=600)

# ⬇️ Download CSV for past prediction comparison
csv_compare = compare_df.to_csv().encode('utf-8')
st.download_button("⬇️ Download Actual vs Predicted CSV", data=csv_compare, file_name=f"{stock}_actual_vs_predicted.csv", mime='text/csv')

# 📈 Plot comparison
fig2 = plt.figure(figsize=(15, 6))
plt.plot(data['Close'][:split_len + 100], label="Training Data")
plt.plot(compare_df["Actual Price"], label="Actual Test Data")
plt.plot(compare_df["Predicted Price"], label="Predicted")
plt.title(f"{company_name} Past Prediction Performance")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# 🔮 Predict Next 10 Years
st.subheader("🔮 Forecast: Next 10 Years of Stock Prices")
future_days = 3650  # fixed to 10 years

last_100 = scaled_data[-100:]
future_input = list(last_100)
future_predictions = []

progress_bar = st.progress(0)
status_text = st.empty()

for i in range(future_days):
    input_seq = np.array([
        x[0] if isinstance(x, (np.ndarray, list)) else x
        for x in future_input[-100:]
    ]).reshape(1, 100, 1)

    try:
        pred = model.predict(input_seq, verbose=0)
        next_val = pred[0][0]
    except Exception as e:
        st.error(f"Error in prediction at step {i}: {e}")
        break

    future_predictions.append(next_val)
    future_input.append(next_val)

    if future_days > 100:
        progress_bar.progress((i + 1) / future_days)
        if i % 100 == 0 or i == future_days - 1:
            status_text.text(f"Generating day {i + 1} of {future_days}")

st.success(f"✅ Prediction complete: {future_days} days")

# 🧾 Inverse scale
future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 📅 Dates for forecast
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

future_df = pd.DataFrame({'Predicted Price': future_predictions_inv.flatten()}, index=future_dates)
future_df['Predicted Price'] = future_df['Predicted Price'].round(2)

# 📅 First 5 forecast values
st.write("📅 Forecast Data (First 5 Days)")
st.write(future_df.head())

# 📉 Full forecast chart
st.subheader("📉 10-Year Forecast Chart")
st.line_chart(future_df)

# 📘 Full forecast table
st.subheader("📘 Forecast for Next 10 Years (3650 Days)")
st.dataframe(future_df, height=800)

# ⬇️ Download full forecast
csv_10yr = future_df.to_csv().encode('utf-8')
st.download_button("⬇️ Download Full 10-Year Forecast CSV", data=csv_10yr, file_name=f"{stock}_10_year_forecast.csv", mime='text/csv')





