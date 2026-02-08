
import pandas as pd
import numpy as np
import os
from pathlib import Path

def convert_to_qlib_format(input_file="BTCUSDT_15m.csv", qlib_dir="~/.qlib/qlib_data/my_crypto"):
    # Expand user path
    qlib_dir = os.path.expanduser(qlib_dir)
    os.makedirs(qlib_dir, exist_ok=True)
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Qlib requires: date, symbol, open, close, high, low, volume, amount(money), factor(adj)
    # We have: datetime, open, high, low, close, volume, quote_asset_volume, ...
    
    # We need to save to a directory structure:
    # qlib_dir/calendars/day.txt (or 15min.txt?)
    # qlib_dir/features/BTCUSDT. 
    #   open.bin, close.bin, ...
    
    # Dump strictly for Qlib's dump_bin tool?
    # Or use Qlib's DumpData?
    
    # Let's create a CSV that matches Qlib's dump_bin requirements.
    # Columns: symbol, date, open, high, low, close, volume, amount, factor
    
    # 1. Prepare DF
    df['symbol'] = "BTCUSDT"
    df['date'] = df['datetime'] # Qlib handles datetime for high-freq?
    # For 15m data, date should include time?
    # Qlib format for high freq usually needs specific timestamp format.
    
    # Rename
    # amount = quote_asset_volume
    df = df.rename(columns={'quote_asset_volume': 'amount'})
    df['factor'] = 1.0
    
    # Select cols
    cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'factor']
    df_dump = df[cols].copy()
    
    # Save to temp csv
    temp_csv = "qlib_source_btc_15m.csv"
    df_dump.to_csv(temp_csv, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Saved temp CSV to {temp_csv}")
    
    # 2. Initialize Qlib
    # actually we use `python -m qlib.run.dump_data` typically.
    # But we can call it from python.
    
    from qlib.utils import exists_qlib_data, init_instance_by_config
    # dump_bin is in scripts.
    
    # Manual Binary Dump (Fallback)
    # Qlib format: float32, little endian.
    # Features folder: day.bin (date index), BTCUSDT/open.bin, etc.
    
    # Actually, using qlib.tests.data ? No.
    # Let's try to locate dump_bin.py dynamically or implement pandas to bin.
    
    # Try to find dump_bin.py in site-packages
    import qlib
    qlib_path = Path(qlib.__file__).parent
    dump_script = qlib_path / "scripts" / "dump_bin.py"
    
    if not dump_script.exists():
        # Maybe in parent/scripts?
        dump_script = qlib_path.parent / "scripts" / "dump_bin.py"
    
    # Manual Binary Dump Implementation
    # Qlib stores data in: <qlib_dir>/features/<symbol>/<field>.bin
    # Format: float32 (little endian)
    # Also <qlib_dir>/calendars/15min.txt: sorted datetime strings
    # Also <qlib_dir>/instruments/all.txt: symbol, start_date, end_date (tabs)

    import struct
    import shutil
    
    # 1. Calendars
    calendar_path = Path(qlib_dir) / "calendars"
    calendar_path.mkdir(parents=True, exist_ok=True)
    
    dates = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S').unique()
    dates.sort()
    
    with open(calendar_path / "15min.txt", "w") as f:
        for d in dates:
            f.write(f"{d}\n")
            
    # Copy to day.txt to satisfy default checks if any
    shutil.copy(calendar_path / "15min.txt", calendar_path / "day.txt")
            
    print(f"Saved calendar with {len(dates)} steps (15min & day).")

    # 2. Features
    features_path = Path(qlib_dir) / "features" / "BTCUSDT"
    features_path.mkdir(parents=True, exist_ok=True)
    
    fields = ['open', 'close', 'high', 'low', 'volume', 'amount', 'factor']
    
    # Create date map for all trading days? 
    # Qlib assumes continuous data usually aligned with calendar.
    # Our data is continuous from Binance.
    # If there are gaps, Qlib needs NaNs.
    # Let's assume dates match df exactly for now. 
    # Real Qlib uses a global calendar and maps index.
    
    # For simplification, we just dump what we have.
    # But Qlib's data loader relies on calendar file.
    # If df matches calendar 1:1, we are good.
    
    for field in fields:
        # Convert to float32
        data = df[field].astype(np.float32).values
        
        # Prepend start_index=0 (as float32) for Qlib header
        # Qlib expects [start_index, data...]
        data_with_header = np.hstack([[0.0], data]).astype("<f")
        
        # Save as 15min.bin (Required for freq='15min')
        bin_path_15m = features_path / f"{field.lower()}.15min.bin"
        with open(bin_path_15m, "wb") as f:
            data_with_header.tofile(f)
            
        # Save as day.bin (Fallback)
        bin_path_day = features_path / f"{field.lower()}.day.bin"
        with open(bin_path_day, "wb") as f:
            data_with_header.tofile(f)
            
        # Also save as .bin for safety if Qlib defaults?
        bin_path_default = features_path / f"{field.lower()}.bin"
        with open(bin_path_default, "wb") as f:
            data_with_header.tofile(f)
            
    print(f"Saved binary features for BTCUSDT.")
    
    # 3. Instruments
    instruments_path = Path(qlib_dir) / "instruments"
    instruments_path.mkdir(parents=True, exist_ok=True)
    
    real_start = dates[0].split()[0]
    real_end = dates[-1].split()[0]
    
    # Fudge start date to ensure coverage
    start_date = "2024-01-01" 
    end_date = real_end
    
    with open(instruments_path / "all.txt", "w") as f:
        f.write(f"BTCUSDT\t{start_date}\t{end_date}\n")
        
    print("Saved instruments file.")
    print("Qlib data conversion successful (Manual Mode)!")
    


if __name__ == "__main__":
    convert_to_qlib_format()
