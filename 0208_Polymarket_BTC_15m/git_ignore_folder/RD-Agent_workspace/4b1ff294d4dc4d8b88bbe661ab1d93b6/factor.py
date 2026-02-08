import pandas as pd
import numpy as np

def calculate_PriceMomentum20D():
    # Load the source data
    # The file is expected to have a MultiIndex ['datetime', 'instrument']
    df = pd.read_hdf('daily_pv_debug.h5')
    
    # Ensure the index is sorted by instrument and then datetime to perform time-series operations correctly
    df = df.sort_index(level=['instrument', 'datetime'])
    
    # Calculate the 20-day momentum
    # Formulation: (Close_t - Close_{t-20}) / Close_{t-20}
    # This is equivalent to (Close_t / Close_{t-20}) - 1
    
    # Group by instrument to ensure the shift operation stays within each asset's time series
    close_t_minus_20 = df.groupby('instrument')['$close'].shift(20)
    
    # Calculate the factor
    momentum = (df['$close'] - close_t_minus_20) / close_t_minus_20
    
    # Create the result dataframe
    result = pd.DataFrame(index=df.index)
    result['PriceMomentum20D'] = momentum.astype(np.float64)
    
    # Save the result to result.h5
    result.to_hdf('result.h5', key='df', mode='w')

if __name__ == '__main__':
    calculate_PriceMomentum20D()