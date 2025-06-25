import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Prévision CA", layout="wide")
st.title("📈 Application de Prévision du Chiffre d'Affaires")

uploaded_file = st.sidebar.file_uploader("Importer le fichier CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=",", decimal=",")
    mois_dict = {
        "janvier":1, "février":2, "mars":3, "avril":4, "mai":5, "juin":6,
        "juillet":7, "août":8, "septembre":9, "octobre":10, "novembre":11, "décembre":12
    }
    df["Mois_num"] = df["Mois"].map(mois_dict)
    df["Date"] = pd.to_datetime(dict(year=df["Année"], month=df["Mois_num"], day=1))
    df_final = df[["Date", "Chiffre d'affaire"]].sort_values("Date")
    df_final.set_index("Date", inplace=True)
    df_final = df_final.asfreq("MS")

    st.subheader("📊 Évolution mensuelle du chiffre d'affaires")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df_final.index, df_final["Chiffre d'affaire"], marker='o')
    ax1.axvspan(pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30"), color="red", alpha=0.2, label="Chute COVID")
    ax1.set_title("Évolution du chiffre d'affaires")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("CA")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    train = df_final.loc["2020-01-01":"2020-11-30"]
    test = df_final.loc["2020-12-01":"2021-03-31"]

    run_arima = st.sidebar.checkbox("🔶 Lancer ARIMA")
    run_prophet = st.sidebar.checkbox("🔷 Lancer Prophet")
    run_lstm = st.sidebar.checkbox("🔵 Lancer LSTM")

    if run_arima:
        model_arima = ARIMA(train["Chiffre d'affaire"], order=(2, 1, 1))
        model_arima_fit = model_arima.fit()

        pred_test_arima = model_arima_fit.forecast(steps=len(test))
        pred_test_arima.index = test.index

        mae_arima = mean_absolute_error(test["Chiffre d'affaire"], pred_test_arima)
        rmse_arima = np.sqrt(mean_squared_error(test["Chiffre d'affaire"], pred_test_arima))

        future_dates = pd.date_range(start="2021-04-01", periods=6, freq="MS")
        forecast_future_arima = model_arima_fit.forecast(steps=6)
        forecast_future_arima.index = future_dates

        st.subheader("📈 Prévision ARIMA")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df_final.index, df_final["Chiffre d'affaire"], label="Historique", marker='o')
        ax2.plot(pred_test_arima.index, pred_test_arima, label="Prévision ARIMA (test)", color="orange", linestyle="--", marker="x")
        ax2.plot(forecast_future_arima.index, forecast_future_arima, label="Prévision ARIMA (futur)", color="green", marker="s")
        ax2.axvspan(pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30"), color="red", alpha=0.1)
        ax2.set_title("Prévision ARIMA")
        ax2.legend()
        st.pyplot(fig2)

        st.success(f"MAE : {mae_arima:,.2f}".replace(",", " ").replace(".", ","))
        st.success(f"RMSE : {rmse_arima:,.2f}".replace(",", " ").replace(".", ","))

    if run_prophet:
        df_prophet = train.reset_index()[["Date", "Chiffre d'affaire"]].rename(columns={"Date": "ds", "Chiffre d'affaire": "y"})
        model_prophet = Prophet()
        model_prophet.fit(df_prophet)

        future_test = test.reset_index().rename(columns={"Date": "ds"})
        forecast_test_prophet = model_prophet.predict(future_test)

        mae_prophet = mean_absolute_error(test["Chiffre d'affaire"], forecast_test_prophet["yhat"])
        rmse_prophet = np.sqrt(mean_squared_error(test["Chiffre d'affaire"], forecast_test_prophet["yhat"]))

        future_dates_prophet = pd.DataFrame({"ds": pd.date_range(start="2021-04-01", periods=6, freq="MS")})
        forecast_future_prophet = model_prophet.predict(future_dates_prophet)

        st.subheader("📈 Prévision Prophet")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(df_final.index, df_final["Chiffre d'affaire"], label="Historique", marker='o')
        ax3.plot(test.index, forecast_test_prophet["yhat"], label="Prévision Prophet (test)", color="orange", linestyle="--", marker="x")
        ax3.plot(future_dates_prophet["ds"], forecast_future_prophet["yhat"], label="Prévision Prophet (futur)", color="green", marker="s")
        ax3.axvspan(pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30"), color="red", alpha=0.1)
        ax3.set_title("Prévision Prophet")
        ax3.legend()
        st.pyplot(fig3)

        st.success(f"MAE : {mae_prophet:,.2f}".replace(",", " ").replace(".", ","))
        st.success(f"RMSE : {rmse_prophet:,.2f}".replace(",", " ").replace(".", ","))

    if run_lstm:
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)

        def create_sequences(data, window_size=1):
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size])
                y.append(data[i+window_size])
            return np.array(X), np.array(y)

        window_size = 1
        X_train, y_train = create_sequences(train_scaled, window_size)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0)

        test_scaled = scaler.transform(test)
        input_seq = train_scaled[-window_size:]
        predictions_test_scaled = []

        for i in range(len(test)):
            pred = model.predict(input_seq.reshape(1, window_size, 1), verbose=0)[0,0]
            predictions_test_scaled.append(pred)
            input_seq = np.array([[pred]])

        predictions_test = scaler.inverse_transform(np.array(predictions_test_scaled).reshape(-1,1)).flatten()

        mae_lstm = mean_absolute_error(test["Chiffre d'affaire"], predictions_test)
        rmse_lstm = np.sqrt(mean_squared_error(test["Chiffre d'affaire"], predictions_test))

        input_seq = np.array([[predictions_test_scaled[-1]]])
        predictions_future_scaled = []
        for _ in range(6):
            pred = model.predict(input_seq.reshape(1, window_size, 1), verbose=0)[0,0]
            predictions_future_scaled.append(pred)
            input_seq = np.array([[pred]])

        predictions_future = scaler.inverse_transform(np.array(predictions_future_scaled).reshape(-1,1)).flatten()
        future_dates = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

        st.subheader("📈 Modèle LSTM (Réseau de neurones récurrent)")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.plot(df_final.index, df_final["Chiffre d'affaire"], label="Historique", marker='o')
        ax4.plot(test.index, predictions_test, label="Prévision LSTM (test)", color="orange", linestyle="--", marker="x")
        ax4.plot(future_dates, predictions_future, label="Prévision LSTM (futur)", color="green", marker="s")
        ax4.axvspan(pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30"), color="red", alpha=0.1)
        ax4.set_title("Prévision LSTM")
        ax4.legend()
        st.pyplot(fig4)

        st.success(f"MAE : {mae_lstm:,.2f}".replace(",", " ").replace(".", ","))
        st.success(f"RMSE : {rmse_lstm:,.2f}".replace(",", " ").replace(".", ","))

else:
    st.warning("Veuillez importer un fichier CSV contenant les données.")
