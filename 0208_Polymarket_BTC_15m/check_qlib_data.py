
import qlib
from qlib.data import D
import os
import pandas as pd

QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/my_crypto")
qlib.init(provider_uri=QLIB_DIR)

print("\n--- Qlib Detailed Diagnostic ---")

# 1. Calendar
cal = D.calendar(freq='15min')
print(f"Calendar: {len(cal)} steps. First: {cal[0]}, Last: {cal[-1]}")

# 2. Instruments
inst_config = D.instruments(market='all')
inst_list = D.list_instruments(inst_config)
print(f"Instruments in 'all': {inst_list}")

# 3. Check specific instrument range from Qlib
for sym, ranges in inst_list.items():
    print(f"Symbol: {sym}, Ranges: {ranges}")

# 4. Check binary files
base_req = "features/15min/btcusdt/close.bin"
full_path = os.path.join(QLIB_DIR, base_req)
if os.path.exists(full_path):
    size = os.path.getsize(full_path)
    print(f"File {base_req} exists. Size: {size} bytes")
else:
    print(f"File {base_req} NOT FOUND!")

# 5. Try loading with relative indices instead of dates
try:
    print("Attempting to load using relative indices (first 5 steps)...")
    # Using lowercase btcusdt
    df = D.features(['btcusdt'], ['$close'], start_time=0, end_time=5, freq='15min')
    print("Success with indices!")
    print(df)
except Exception as e:
    print(f"Failed with indices: {e}")

# 6. Check Feature Names manually
path_check = os.path.join(QLIB_DIR, 'features', '15min', 'btcusdt')
print(f"Features folder list: {os.listdir(path_check) if os.path.exists(path_check) else 'N/A'}")
