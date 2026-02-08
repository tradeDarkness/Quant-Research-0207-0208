import pandas as pd
import numpy as np

def calculate_VolumeVolatility5D():
    # Load the data from the HDF5 file
    # The file contains a MultiIndex: ['datetime', 'instrument']
    data = pd.read_hdf('daily_pv_debug.h5')
    
    # Extract the volume column
    # We use $volume according to the source data description
    volume = data['$volume']
    
    # Group by instrument to calculate the rolling standard deviation per stock
    # The window size is 5 as specified in the factor formulation and name
    # ddof=0 is used to match the population standard deviation formula provided in the formulation
    factor = volume.groupby('instrument').rolling(window=5).std(ddof=0)
    
    # The rolling operation on a grouped series returns a multi-indexed series 
    # where the first level is the grouping key ('instrument') and the second is the original index.
    # We need to reorder the index to match ['datetime', 'instrument'] and name the column.
    factor = factor.reset_index(level=0, drop=True).reindex(data.index)
    
    # Create a DataFrame for the result
    result = pd.DataFrame(index=data.index)
    result['VolumeVolatility5D'] = factor
    
    # Save the result to result.h5
    result.to_hdf('result.h5', key='df')

if __name__ == '__main__':
    calculate_VolumeVolatility5D()