import pandas as pd
try:
    df = pd.read_hdf("daily_pv_all.h5", key="data")
    print("Instruments:", df.index.get_level_values("instrument").unique())
    print("Min Date:", df.index.get_level_values("datetime").min())
    print("Max Date:", df.index.get_level_values("datetime").max())
    print("Sample:\n", df.head())
except Exception as e:
    print(e)
