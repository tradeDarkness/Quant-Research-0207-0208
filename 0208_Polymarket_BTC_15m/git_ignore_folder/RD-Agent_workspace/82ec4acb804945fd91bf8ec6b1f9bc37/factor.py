import pandas as pd
import numpy as np

def calculate_Price_Momentum_5D():
    # Load the source data
    # The data is expected to be an HDF5 file with MultiIndex ['datetime', 'instrument']
    df = pd.read_hdf('daily_pv_debug.h5')
    
    # Sort index to ensure temporal order for shift operation
    df = df.sort_index(level=['instrument', 'datetime'])
    
    # Calculate the factor: (Close_t - Close_{t-5}) / Close_{t-5}
    # We use groupby on instrument to ensure shifts don't cross-contaminate between stocks
    close = df['$close']
    close_lag5 = df.groupby('instrument')['$close'].shift(5)
    
    # Formulation: (Close - Close_lag5) / Close_lag5
    momentum = (close - close_lag5) / close_lag5
    
    # Create the result dataframe
    result = momentum.to_frame(name='Price_Momentum_5D')
    
    # Save the result to result.h5
    result.to_hdf('result.h5', key='df')

if __name__ == '__main__':
    calculate_Price_Momentum_5D()