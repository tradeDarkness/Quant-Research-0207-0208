import pandas as pd
import numpy as np

def calculate_Price_Momentum_5D():
    # Load the data
    data = pd.read_hdf('daily_pv_debug.h5')
    
    # Ensure the index is sorted by instrument and then datetime for correct shifting
    # Qlib data usually comes as (datetime, instrument)
    data = data.sort_index(level=['instrument', 'datetime'])
    
    # Extract adjusted close price
    # In Qlib, $close is often the raw close, and $factor is the cumulative adjustment factor
    # Adjusted Close = $close * $factor
    adj_close = data['$close'] * data['$factor']
    
    # Calculate the price 5 days ago per instrument
    # groupby('instrument') ensures we don't shift values between different stocks
    adj_close_t_5 = adj_close.groupby('instrument').shift(5)
    
    # Calculate Momentum: (Close_t - Close_t-5) / Close_t-5
    # This is equivalent to (Close_t / Close_t-5) - 1
    momentum = (adj_close / adj_close_t_5) - 1
    
    # Convert to DataFrame with the specific factor name
    result = momentum.to_frame(name='Price_Momentum_5D')
    
    # Qlib expects the index to be (datetime, instrument)
    result = result.swaplevel('instrument', 'datetime').sort_index()
    
    # Save the result to result.h5
    result.to_hdf('result.h5', key='df')

if __name__ == '__main__':
    calculate_Price_Momentum_5D()