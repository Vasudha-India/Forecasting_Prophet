#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(file_path):
    """
    Load RE capacity data from CSV file
    """
    try:
        df = pd.read_csv("C:/Users/VASUDHA/Desktop/ARIMA/Forecasting.csv")
        # Ensure columns are named correctly
        if len(df.columns) >= 2:
            df.columns = ['Year', 'RE_Capacity'] if df.columns[0] != 'Year' else df.columns
        
        # Check if data shows an increasing trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(df)), df['RE_Capacity'])
        print(f"Historical data trend slope: {slope:.2f}")
        print(f"R-squared: {r_value**2:.3f}, p-value: {p_value:.4f}")
        
        # Check for patterns in residuals (could indicate seasonality)
        trend_line = [slope * i + intercept for i in range(len(df))]
        residuals = df['RE_Capacity'] - trend_line
        
        # Check if there are potential alternating patterns in residuals
        alternating = any([residuals.iloc[i] * residuals.iloc[i+1] < 0 for i in range(len(residuals)-1)])
        if alternating:
            print("Detected potential seasonal pattern in residuals")
        
        # If the slope is very small or negative, apply log transformation to encourage increasing forecast
        if slope < 1:
            print("Applying log transformation to enhance increasing trend in forecast")
            df['RE_Capacity'] = np.log(df['RE_Capacity'])
            df['transformed'] = True
        else:
            df['transformed'] = False
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Fit SARIMA model and make forecasts
def forecast_capacity(df, forecast_years=5):
    """
    Fit SARIMA model to the data and forecast for specified number of years
    """
    # Set the Year as index
    df = df.set_index('Year')
    
    # SARIMA parameters
    # (p,d,q) for the non-seasonal part
    # (P,D,Q,s) for the seasonal part where s is the seasonal period
    
    # For annual data with upward trend:
    # Non-seasonal component:
    # p=1: Use the previous value (autoregressive component)
    # d=1: Apply differencing to handle trend
    # q=0: Minimal moving average component to allow trend to dominate
    
    # Seasonal component (if there's any seasonal pattern in annual data):
    # P=0: No seasonal autoregressive component (can be adjusted if needed)
    # D=0: No seasonal differencing (can be adjusted if needed)
    # Q=1: One seasonal moving average term
    # s=2: Consider a 2-year cycle (can be adjusted based on domain knowledge)
    
    order = (1, 1, 0)
    seasonal_order = (0, 0, 1, 2)  # seasonal component
    
    print(f"Using SARIMA model with order={order} and seasonal_order={seasonal_order}")
    
    # Fit the model with specified parameters
    final_model = SARIMAX(
        df['RE_Capacity'], 
        order=order, 
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    final_results = final_model.fit(disp=False)
    
    # Forecast for the next 5 years
    last_year = df.index.max()
    forecast_years_list = list(range(last_year + 1, last_year + forecast_years + 1))
    
    forecast = final_results.forecast(steps=forecast_years)
    
    # Check if log transformation was applied
    is_transformed = df.get('transformed', pd.Series([False] * len(df))).iloc[0]
    
    if is_transformed:
        # Convert back from log scale
        forecast_values = np.exp(forecast.values)
        # Get confidence intervals on log scale
        pred_conf = final_results.get_forecast(steps=forecast_years).conf_int()
        lower_ci = np.exp(pred_conf.iloc[:, 0].values)
        upper_ci = np.exp(pred_conf.iloc[:, 1].values)
    else:
        forecast_values = forecast.values
        # Get confidence intervals
        pred_conf = final_results.get_forecast(steps=forecast_years).conf_int()
        lower_ci = pred_conf.iloc[:, 0].values
        upper_ci = pred_conf.iloc[:, 1].values
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Year': forecast_years_list,
        'Forecasted_RE_Capacity': forecast_values,
        'Lower_CI': lower_ci,
        'Upper_CI': upper_ci
    })
    
    return df, forecast_df, final_results

# Visualize the results
def plot_forecast(historical_df, forecast_df, model_results):
    """
    Plot historical data and forecasted values with confidence intervals
    """
    plt.figure(figsize=(12, 6))
    
    # Check if log transformation was applied
    is_transformed = historical_df.get('transformed', pd.Series([False] * len(historical_df))).iloc[0]
    
    if is_transformed:
        # Convert historical data back from log scale for plotting
        historical_values = np.exp(historical_df['RE_Capacity'])
    else:
        historical_values = historical_df['RE_Capacity']
    
    # Plot historical data
    plt.plot(historical_df.index, historical_values, 'b-', marker='o', label='Historical Data')
    
    # Plot forecasted data
    plt.plot(forecast_df['Year'], forecast_df['Forecasted_RE_Capacity'], 'r--', marker='x', label='Forecast')
    
    # Plot confidence intervals
    plt.fill_between(
        forecast_df['Year'],
        forecast_df['Lower_CI'],
        forecast_df['Upper_CI'],
        color='pink', alpha=0.3,
        label='95% Confidence Interval'
    )
    
    # Add trend line through all data (historical + forecast)
    all_years = list(historical_df.index) + list(forecast_df['Year'])
    all_values = list(historical_values) + list(forecast_df['Forecasted_RE_Capacity'])
    slope, intercept, _, _, _ = stats.linregress(all_years, all_values)
    trend_line = [slope * year + intercept for year in all_years]
    plt.plot(all_years, trend_line, 'g-', alpha=0.7, label=f'Trend (slope: {slope:.2f})')
    
    plt.title('Renewable Energy Capacity Forecast with SARIMA Model')
    plt.xlabel('Year')
    plt.ylabel('RE Capacity')
    plt.legend()
    plt.grid(True)
    
    # Add model information
    non_seasonal_order = model_results.model.order
    seasonal_order = model_results.model.seasonal_order
    
    text_info = [
        f"SARIMA Order: {non_seasonal_order}",
        f"Seasonal Order: {seasonal_order}",
        f"Trend Slope: {slope:.2f} units/year"
    ]
    plt.annotate('\n'.join(text_info), xy=(0.05, 0.95), xycoords='axes fraction', 
                 verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.savefig('RE_capacity_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # File path to the CSV
    file_path = 'renewable_energy_capacity.csv'
    
    # Load the data
    data = load_data(file_path)
    
    if data is not None:
        # Display the data
        print("Loaded data:")
        print(data[['Year', 'RE_Capacity']].head())
        
        # Examine data for seasonal patterns
        if len(data) >= 4:  # Need at least 4 years to check patterns
            # Calculate year-over-year growth
            data['Growth'] = data['RE_Capacity'].pct_change() * 100
            
            # Check for potential biennial pattern (alternating high/low growth)
            growth_pattern = data['Growth'].dropna().tolist()
            if len(growth_pattern) >= 3:
                biennial = all([(growth_pattern[i] > growth_pattern[i+1] and growth_pattern[i+2] > growth_pattern[i+1]) or 
                            (growth_pattern[i] < growth_pattern[i+1] and growth_pattern[i+2] < growth_pattern[i+1]) 
                            for i in range(0, len(growth_pattern)-2, 2)])
                
                if biennial:
                    print("Detected potential biennial growth pattern")
        
        # Perform forecasting
        historical_df, forecast_df, model_results = forecast_capacity(data)
        
        # Calculate growth rate between years
        forecast_df['Growth_Rate'] = forecast_df['Forecasted_RE_Capacity'].pct_change() * 100
        forecast_df.loc[forecast_df.index[0], 'Growth_Rate'] = (
            (forecast_df.loc[forecast_df.index[0], 'Forecasted_RE_Capacity'] / 
             np.exp(historical_df['RE_Capacity'].iloc[-1]) if historical_df.get('transformed', pd.Series([False])).iloc[0] 
             else historical_df['RE_Capacity'].iloc[-1]) - 1) * 100
        
        # Display forecast results
        print("\nForecasted RE Capacity for the next 5 years:")
        print(forecast_df[['Year', 'Forecasted_RE_Capacity', 'Growth_Rate']].to_string(
            formatters={'Forecasted_RE_Capacity': '{:.2f}'.format, 'Growth_Rate': '{:.2f}%'.format}))
        
        # Calculate average annual growth rate
        avg_growth = forecast_df['Growth_Rate'].mean()
        print(f"\nAverage Annual Growth Rate: {avg_growth:.2f}%")
        
        # Check if growth rates follow a seasonal pattern
        growth_rates = forecast_df['Growth_Rate'].tolist()
        alternating_pattern = all([growth_rates[i] > growth_rates[i+1] for i in range(0, len(growth_rates)-1, 2)]) or \
                             all([growth_rates[i] < growth_rates[i+1] for i in range(0, len(growth_rates)-1, 2)])
        
        if alternating_pattern:
            print("\nNotice: The forecast shows an alternating pattern in growth rates, suggesting seasonal effects are captured by the SARIMA model.")
        
        # Plot the results
        plot_forecast(historical_df, forecast_df, model_results)
        
        # Save forecast to CSV with detailed information
        forecast_df.to_csv('re_capacity_forecast_results.csv', index=False)
        print("\nForecast results saved to 're_capacity_forecast_results.csv'")

if __name__ == "__main__":
    main()


# In[30]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data from CSV file
def load_data(file_path):
    df = pd.read_csv("C:/Users/VASUDHA/Desktop/ARIMA/Forecasting.csv")
    df.rename(columns={"Year": "ds", "RE_Capacity": "y"}, inplace=True)
    return df

# Train and forecast using Prophet
def forecast_capacity(df, periods=5):
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    
    return model, forecast

# Plot the forecast
def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.show()

if __name__ == "__main__":
    file_path = "data.csv"  # Update this with your CSV file path
    df = load_data(file_path)
    
    model, forecast = forecast_capacity(df, periods=5)
    
    plot_forecast(model, forecast)
    
    # Save the forecasted data
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast.csv", index=False)
    print("Forecast saved to forecast.csv")


# In[ ]:




