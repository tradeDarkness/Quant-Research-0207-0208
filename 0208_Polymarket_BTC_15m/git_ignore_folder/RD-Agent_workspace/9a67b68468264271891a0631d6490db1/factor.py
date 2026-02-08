import pandas as pd
import numpy as np

def calculate_Vol_CV_5D():
    # Load the data from the HDF5 file
    # The data contains a MultiIndex: ['datetime', 'instrument']
    data = pd.read_hdf('daily_pv_debug.h5')
    
    # Extract the volume column
    # We need to calculate the rolling mean and standard deviation per instrument
    volume = data['$volume']
    
    # Group by instrument to ensure rolling calculations are isolated to each stock
    grouped = volume.groupby('instrument')
    
    # Calculate the 5-day rolling mean
    rolling_mean = grouped.rolling(window=5).mean()
    
    # Calculate the 5-day rolling standard deviation
    rolling_std = grouped.rolling(window=5).std()
    
    # Calculate the Coefficient of Variation (CV): Std / Mean
    # We reset the index levels to match the original dataframe structure if necessary,
    # but pandas rolling on groupby usually returns a series with the same MultiIndex (instrument, datetime)
    # We need to align it back to (datetime, instrument)
    cv = rolling_std / rolling_mean
    
    # The groupby rolling returns index as (instrument, datetime), 
    # we swap it to (datetime, instrument) to match the input format
    cv = cv.reorder_levels(['datetime', 'instrument']).sort_index()
    
    # Create a DataFrame for the result
    result_df = cv.to_frame(name='Vol_CV_5D')
    
    # Save the result to result.h5
    result_df.to_hdf('result.h5', key='df')

if __name__ == '__main__':
    calculate_Vol_CV_5D()