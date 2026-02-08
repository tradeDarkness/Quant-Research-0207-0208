
import pandas as pd
import numpy as np
import os
from pathlib import Path

def convert_to_qlib_format(input_file="BTCUSDT_15m_tb.csv", qlib_dir="~/.qlib/qlib_data/my_crypto"):
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
    df['symbol'] = "btcusdt"
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
    
    # 2. Calendars
    calendar_path = Path(qlib_dir) / "calendars"
    calendar_path.mkdir(parents=True, exist_ok=True)
    
    dates = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S').unique()
    dates.sort()
    
    with open(calendar_path / "15min.txt", "w") as f:
        for d in dates:
            f.write(f"{d}\n")
            
    # day.txt MUST only contain unique date strings (YYYY-MM-DD)
    unique_days = sorted(df['date'].dt.strftime('%Y-%m-%d').unique())
    with open(calendar_path / "day.txt", "w") as f:
        for d in unique_days:
            f.write(f"{d}\n")
            
    print(f"Saved calendar with {len(dates)} steps (15min & day).")

    # 3. Features
    # Naming convention: Qlib usually stores in <qlib_dir>/features/<symbol>/<field>.bin
    # For high freq, it may look for <qlib_dir>/features/<freq>/<symbol>/<field>.bin
    # 2. Features
    # Naming convention for high freq in Qlib: 
    # <qlib_dir>/features/<freq>/<symbol_lower>/<field>.bin
    # Note: Some Qlib versions are sensitive to symbol casing in the folder name.
    symbol_lower = "btcusdt"
    features_path_main = Path(qlib_dir) / "features" / symbol_lower
    features_path_15min = Path(qlib_dir) / "features" / "15min" / symbol_lower
    
    for p in [features_path_main, features_path_15min]:
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    
    fields = ['open', 'close', 'high', 'low', 'volume', 'amount', 'factor', 'lb_tb']
    
    for field in fields:
        # Convert to float32
        data = df[field].astype(np.float32).values
        data_with_header = np.hstack([[0.0], data]).astype("<f")
        
        # Save to both paths
        for p in [features_path_main, features_path_15min]:
            bin_path = p / f"{field.lower()}.bin"
            with open(bin_path, "wb") as f:
                data_with_header.tofile(f)
                    
    print(f"Saved binary features for btcusdt (lowercased) in root and 15min folders.")
    
    # 4. Instruments
    instruments_path = Path(qlib_dir) / "instruments"
    instruments_path.mkdir(parents=True, exist_ok=True)
    
    real_start = dates[0].split()[0]
    real_end = dates[-1].split()[0]
    
    # Use actual data range for instrument file
    start_date = real_start
    end_date = real_end
    
    with open(instruments_path / "all.txt", "w") as f:
        # Extend bounds for safety
        f.write(f"{symbol_lower}\t2023-01-01\t2026-12-31\n")
        
    print(f"Qlib data conversion successful for {symbol_lower}!")
    print("Saved instruments file.")
    print("Qlib data conversion successful (Manual Mode)!")
    


if __name__ == "__main__":
    convert_to_qlib_format()
