import pandas as pd
import numpy as np

def calculate_VolAdjMomentum():
    # Load the data
    data = pd.read_hdf('daily_pv_debug.h5')
    
    # Sort index to ensure rolling operations are correct
    data = data.sort_index(level=['instrument', 'datetime'])
    
    # Define hyperparameters
    momentum_window = 20
    vol_volatility_window = 5
    epsilon = 1e-6
    
    # Calculate PriceMomentum20D
    # Formulation: (Price_t / Price_{t-20}) - 1
    # Using $close as the price representative
    close = data['$close']
    price_momentum = close.groupby(level='instrument').apply(lambda x: x / x.shift(momentum_window) - 1)
    
    # Calculate VolumeVolatility5D
    # Formulation: Standard deviation of volume over the last 5 days
    volume = data['$volume']
    vol_volatility = volume.groupby(level='instrument').apply(lambda x: x.rolling(window=vol_volatility_window).std())
    
    # Calculate VolAdjMomentum
    # Formulation: PriceMomentum20D / (VolumeVolatility5D + epsilon)
    factor_values = price_momentum / (vol_volatility + epsilon)
    
    # Create the result DataFrame
    result = factor_values.to_frame(name='VolAdjMomentum')
    
    # Save the result to result.h5
    result.to_hdf('result.h5', key='df')

if __name__ == '__main__':
    calculate_VolAdjMomentum()