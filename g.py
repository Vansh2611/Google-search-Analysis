import pandas as pd
from pytrends.request import TrendReq
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA from statsmodels
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox
from datetime import datetime
from textblob import TextBlob

# Function to fetch Google Trends data for multiple search terms
def fetch_trends_data(search_terms, start_date, end_date):
    pytrends = TrendReq()
    pytrends.build_payload(kw_list=search_terms, timeframe=f'{start_date} {end_date}')
    data = pytrends.interest_over_time()
    return data

# Function for machine learning modeling
def perform_machine_learning(data, search_terms, method='linear_regression'):
    for search_term in search_terms:
        # Assuming 'date' is in datetime format, if not, convert it
        data['date'] = pd.to_datetime(data.index)

        # Feature engineering: Extracting month and day as features
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day

        # Define features and target
        features = ['month', 'day']
        target = search_term

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Choose the machine learning method
        if method == 'linear_regression':
            model = LinearRegression()
        elif method == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == 'svr':
            model = SVR()
        else:
            raise ValueError("Invalid machine learning method specified.")

        # Create and train the model
        model.fit(train_data[features], train_data[target])

        # Predict on the test set
        predictions = model.predict(test_data[features])

        # Evaluate the model
        mse = mean_squared_error(test_data[target], predictions)
        print(f'Mean Squared Error ({method}) for {search_term}: {mse}')

        # Plot the actual vs predicted values
        plt.scatter(test_data['date'], test_data[target], label=f'Actual - {search_term}')
        plt.scatter(test_data['date'], predictions, label=f'Predicted ({method}) - {search_term}')
    
    plt.xlabel('Date')
    plt.ylabel('Interest')
    plt.legend()
    plt.title(f'Comparative Analysis of Interest Trends')
    plt.show()

# Function for time series forecasting using ARIMA
def perform_time_series_forecasting(data, search_terms):
    for search_term in search_terms:
        # Assuming 'date' is in datetime format, if not, convert it
        data['date'] = pd.to_datetime(data.index)

        # Feature engineering: Extracting month and day as features
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day

        # Define features and target
        features = ['month', 'day']
        target = search_term

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Create a time series model (ARIMA)
        model = ARIMA(train_data[target], order=(5, 1, 0))  
        fitted_model = model.fit()

        # Forecast future values
        forecast = fitted_model.forecast(steps=len(test_data))

        # Evaluate the model
        mse = mean_squared_error(test_data[target], forecast)
        print(f'Mean Squared Error (ARIMA) for {search_term}: {mse}')

        # Plot the actual vs forecasted values
        plt.plot(test_data['date'], test_data[target], label=f'Actual - {search_term}')
        plt.plot(test_data['date'], forecast, label=f'Forecast (ARIMA) - {search_term}')
    
    plt.xlabel('Date')
    plt.ylabel('Interest')
    plt.legend()
    plt.title(f'Time Series Forecasting with ARIMA')
    plt.show()


# Function to handle the search button click
def search():
    try:
        search_terms = entry_search.get().split(',')
        start_date = entry_start_date.get()
        end_date = entry_end_date.get()

        if not all([search_terms, start_date, end_date]):
            messagebox.showinfo("Error", "Please enter search terms and select start and end dates.")
            return

        # Fetch Google Trends data
        trends_data = fetch_trends_data(search_terms, start_date, end_date)

        # Perform machine learning and display comparative results
        perform_machine_learning(trends_data, search_terms, method='linear_regression')
        perform_machine_learning(trends_data, search_terms, method='random_forest')
        perform_machine_learning(trends_data, search_terms, method='svr')

        # Perform time series forecasting with ARIMA
        perform_time_series_forecasting(trends_data, search_terms)

    except Exception as e:
        messagebox.showinfo("Error", f"An error occurred: {str(e)}")

# Create main application window
root = Tk()
root.title("Google Trends Analysis")

# Labels and entry for user input
label_search = Label(root, text="Enter search terms (comma-separated):")
label_search.pack()

entry_var_search = StringVar()
entry_search = Entry(root, textvariable=entry_var_search)
entry_search.pack()

label_start_date = Label(root, text="Enter start date (YYYY-MM-DD):")
label_start_date.pack()

entry_var_start_date = StringVar()
entry_start_date = Entry(root, textvariable=entry_var_start_date)
entry_start_date.pack()

label_end_date = Label(root, text="Enter end date (YYYY-MM-DD):")
label_end_date.pack()

entry_var_end_date = StringVar()
entry_end_date = Entry(root, textvariable=entry_var_end_date)
entry_end_date.pack()

# Button to trigger the search
search_button = Button(root, text="Search", command=search)
search_button.pack()

# Run the GUI application
root.mainloop()