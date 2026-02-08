import pandas as pd
import numpy as np

def calculate_Volume_Scaled_Volatility_5D():
    # Load data from the specified HDF5 file
    data = pd.read_hdf('daily_pv_debug.h5')
    
    # Ensure the index is sorted by instrument and datetime for correct rolling calculations
    data = data.sort_index(level=['instrument', 'datetime'])
    
    # Extract necessary columns
    close = data['$close']
    volume = data['$volume']
    
    # Group by instrument to perform time-series calculations per stock
    grouped = data.groupby('instrument')
    
    # 1. Calculate daily returns: (Close_t - Close_{t-1}) / Close_{t-1}
    # Note: pct_change() computes (s[t] - s[t-1]) / s[t-1]
    returns = grouped['$close'].pct_change()
    
    # 2. Calculate rolling 5-day mean volume
    mean_volume_5 = grouped['$volume'].rolling(window=5).mean().reset_index(level=0, drop=True)
    
    # 3. Calculate the volume scaling factor: Volume_t / Mean(Volume, 5)
    volume_scale = volume / mean_volume_5
    
    # 4. Calculate the product: Return * Volume_Scale
    product = returns * volume_scale
    
    # 5. Calculate the 5-day standard deviation of the product
    # We need to re-group product because it's a Series with the original index
    factor_series = product.groupby(level='instrument').rolling(window=5).std().reset_index(level=0, drop=True)
    
    # Create the final DataFrame
    result_df = pd.DataFrame(index=data.index)
    result_df['Volume_Scaled_Volatility_5D'] = factor_series
    
    # Save the result to result.h5
    result_df.to_hdf('result.h5', key='df')

if __name__ == '__main__':
    calculate_Volume_Scaled_Volatility_5D()