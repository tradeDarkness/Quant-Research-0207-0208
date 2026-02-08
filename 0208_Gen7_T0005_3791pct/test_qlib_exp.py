
import qlib
from qlib.data import D
import pandas as pd

provider_uri = '/Users/zhangzc/.qlib/qlib_data/crypto_10m'
qlib.init(provider_uri=provider_uri, region='us')

fields = [
    "Corr($close/Ref($close,1)-1, $volume/Ref($volume,1)-1, 10)",
    "Std($close, 10) / Mean($close, 10)"
]
names = ["test_corr", "test_vol"]

print("Testing expressions...")
try:
    df = D.features(["ETHUSDT"], fields, start_time="2025-01-01", end_time="2025-01-05")
    print("Success!")
    print(df.head())
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
