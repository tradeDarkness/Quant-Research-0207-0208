
import numpy as np
import pandas as pd
from pathlib import Path
import struct
import os

qlib_dir = os.path.expanduser("~/.qlib/qlib_data/my_crypto")
symbol = "BTCUSDT"

def check_data():
    # Check Calendar
    cal_path = Path(qlib_dir) / "calendars" / "15min.txt"
    if not cal_path.exists():
        print("Calendar 15min.txt missing!")
        return
        
    with open(cal_path, "r") as f:
        dates = [line.strip() for line in f]
    print(f"Calendar steps: {len(dates)}")
    print(f"First: {dates[0]}, Last: {dates[-1]}")
    
    # Check Feature
    feat_path = Path(qlib_dir) / "features" / symbol / "close.bin"
    if not feat_path.exists():
        print(f"Feature close.bin missing for {symbol}!")
        return
        
    # Read float32
    with open(feat_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
        
    print(f"Feature close.bin length: {len(data)}")
    print(f"First 5: {data[:5]}")
    print(f"Last 5: {data[-5:]}")
    
    if len(dates) != len(data):
        print(f"MISMATCH: Calendar {len(dates)} vs Data {len(data)}")
    else:
        print("Alignment OK.")
        
    # Check Instruments
    inst_path = Path(qlib_dir) / "instruments" / "all.txt"
    if inst_path.exists():
        with open(inst_path, "r") as f:
            print(f"Instruments all.txt:\n{f.read()}")

if __name__ == "__main__":
    check_data()
